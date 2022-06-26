import os
import pickle
import tempfile
import time
from typing import List, Any

import ray
from opentelemetry import trace
from opentelemetry.trace import Tracer

from __refactored__.replay_buffer.ReplayBufferCachedData import ReplayBufferCachedData
from core.reanalyze_cpu_worker import BatchWorker_CPU
from core.reanalyze_gpu_worker import BatchWorker_GPU
from core.replay_buffer import ReplayBuffer
from core.selfplay_worker import DataWorker
from core.storage import SharedStorage, QueueStorage
from core.test import _test


class RayActors:
    _workers: List[Any]
    module_tracer: Tracer
    batch_storage: QueueStorage
    mcts_storage: QueueStorage

    @property
    def remote_workers(self) -> List[Any]:
        return self._workers

    @property
    def remote_storage(self):
        return self.storage

    @property
    def remote_replay_buffer(self):
        return self.replay_buffer

    @property
    def remote_batch_storage(self) -> QueueStorage:
        return self.batch_storage

    @property
    def remote_mcts_storage(self) -> QueueStorage:
        return self.mcts_storage

    def __init__(self, config, model, target_model):
        self.config = config

        self.module_tracer = trace.get_tracer(__name__)
        with self.module_tracer.start_as_current_span("instantiate remote actors"):
            self.storage = SharedStorage.remote(model, target_model)

            # prepare the batch and mctc context storage
            self.batch_storage = QueueStorage('batch_storage', 150, 200)
            self.mcts_storage = QueueStorage('mcts_storage', 180, 250)

            # prepare the replay buffer
            self.persisted_replay_buffer_file = tempfile.gettempdir() + '/replay_buffer.pkl'
            self.replay_buffer = self.make_replay_buffer_actor(config)

            # other workers
            self._workers = []

            # reanalyze workers
            self.cpu_workers = self.make_reanalyze_cpu_workers(config)
            self.gpu_workers = self.make_reanalyze_gpu_workers(config)

            self.data_workers = self.make_self_play_workers(config)

    def run_remote_workers(self):
        self._workers += [cpu_worker.run.remote() for cpu_worker in self.cpu_workers]
        self._workers += [gpu_worker.run.remote() for gpu_worker in self.gpu_workers]
        self._workers += [worker.run.remote() for worker in self.data_workers]

        # test workers
        self._workers += [_test.remote(self.config, self.storage)]

    def make_replay_buffer_actor(self, config) -> ReplayBuffer:
        """
        Make a new remote ray actor for the replay buffer

        If a file for a persisted replay buffer data exist,
        we use it to pre-warming the replay buffer
        (reducing the booting time)

        :param config:
        :return ReplayBuffer:
        """
        replay_buffer_cached_data = None

        if os.path.isfile(self.persisted_replay_buffer_file):
            replay_buffer_cached_data = pickle.load(open(self.persisted_replay_buffer_file, "rb"))

            for game_history in replay_buffer_cached_data.buffer:
                game_history.obs_history = ray.put(game_history.obs_history)

        cached_data_handler = ray.put(replay_buffer_cached_data)

        return ReplayBuffer \
            .options(name="replay_buffer") \
            .remote(config, cached_data_handler)

    def make_self_play_workers(self, config):
        return [
            DataWorker
            .options(max_concurrency=2, name="self-play-worker-{}".format(rank), lifetime="detached",
                     get_if_exists=True)
            .remote(rank, self.replay_buffer, self.storage, config)
            for rank in range(0, config.num_actors)
        ]

    def make_reanalyze_gpu_workers(self, config):
        return [
            BatchWorker_GPU
            .options(max_concurrency=2, name="reanalyze-gpu-worker-{}".format(idx))
            .remote(
                idx,
                self.replay_buffer,
                self.storage,
                self.batch_storage,
                self.mcts_storage,
                config
            ) for
            idx in range(config.gpu_actor)]

    def make_reanalyze_cpu_workers(self, config):
        return [
            BatchWorker_CPU
            .options(max_concurrency=2, name="reanalyze-cpu-worker-{}".format(idx))
            .remote(
                idx,
                self.replay_buffer,
                self.storage,
                self.batch_storage,
                self.mcts_storage,
                config
            ) for
            idx in range(config.cpu_actor)]

    def kill_actors(self):
        cacheable_replay_buffer: ReplayBufferCachedData = ray.get(self.replay_buffer.get_data_to_cache.remote())

        pickle.dump(cacheable_replay_buffer, open(self.persisted_replay_buffer_file, "wb"))

        for worker in [*self.data_workers, *self.cpu_workers, *self.gpu_workers]:
            worker.terminate.remote()
            time.sleep(2)
            ray.kill(worker)
