import time

import ray
from opentelemetry import trace
from opentelemetry.propagate import inject
from ray import ObjectRef

from core.reanalyze_worker import BatchWorker_CPU, BatchWorker_GPU
from core.replay_buffer import ReplayBuffer
from core.selfplay_worker import DataWorker
from core.storage import SharedStorage, QueueStorage
from core.test import _test


class RayActors:
    batch_storage: QueueStorage
    mcts_storage: QueueStorage

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
        module_tracer = trace.get_tracer(__name__)
        with module_tracer.start_as_current_span("instantiate remote actors"):
            self.storage = SharedStorage.remote(model, target_model)

            # prepare the batch and mctc context storage
            self.batch_storage = QueueStorage(15, 20)
            self.mcts_storage = QueueStorage(18, 25)
            self.replay_buffer = ReplayBuffer \
                .options(name="replay_buffer", lifetime="detached", get_if_exists=True) \
                .remote(config=config)

            # other workers
            self.workers = []

            # reanalyze workers
            cpu_workers = [
                BatchWorker_CPU.remote(idx, self.replay_buffer, self.storage, self.batch_storage, self.mcts_storage,
                                       config) for
                idx in range(config.cpu_actor)]
            self.workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
            gpu_workers = [
                BatchWorker_GPU.remote(idx, self.replay_buffer, self.storage, self.batch_storage, self.mcts_storage,
                                       config) for
                idx in range(config.gpu_actor)]
            self.workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]

            # self-play workers
            with module_tracer.start_as_current_span("running self-play workers"):
                self_play_context = {}
                inject(self_play_context)
                self.data_workers = [
                    DataWorker
                    .options(max_concurrency=2)
                    .remote(rank, self.replay_buffer, self.storage, config, self_play_context)
                    for rank in range(0, config.num_actors)
                ]
                self.workers += [worker.run.remote() for worker in self.data_workers]
            # test workers
            self.workers += [_test.remote(config, self.storage)]

    def kill_actors(self):
        for worker in self.data_workers:
            worker.close_open_spans.remote()
            time.sleep(2)
            ray.kill(worker)
