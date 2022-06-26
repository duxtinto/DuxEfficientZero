import time

import numpy as np
import ray
import torch

from __refactored__.tracing.opentelemetry import make_tracer_provider
from core.utils import prepare_observation_lst, LinearSchedule


@ray.remote
class BatchWorker_CPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):
        """CPU Batch Worker for reanalyzing targets, see Appendix.
        Prepare the context concerning CPU overhead
        Parameters
        ----------
        worker_id: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        batch_storage: Any
            The batch storage (batch queue)
        mcts_storage: Ant
            The mcts-related contexts storage
        """
        # pydevd_pycharm.settrace('localhost', port=5677, stdoutToServer=True, stderrToServer=True)

        self.tracer_provider = make_tracer_provider("smartfighters-efficientZero-reanalyze-CPU-worker-{}".format(worker_id))
        self.service_tracer = self.tracer_provider.get_tracer(__name__)

        with self.service_tracer.start_as_current_span("Initialize reanalyze CPU worker"):
            self.worker_id = worker_id
            self.replay_buffer = replay_buffer
            self.storage = storage
            self.batch_storage = batch_storage
            self.mcts_storage = mcts_storage
            self.config = config

            self.last_model_index = -1
            self.batch_max_num = 20
            self.beta_schedule = LinearSchedule(config.training_steps + config.last_steps, initial_p=config.priority_prob_beta, final_p=1.0)

    def _prepare_reward_value_context(self, indices, games, state_index_lst, total_transitions):
        """prepare the context of rewards and values for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        total_transitions: int
            number of collected transitions
        """
        zero_obs = games[0].zero_obs()
        config = self.config
        value_obs_lst = []
        # the value is valid or not (out of trajectory)
        value_mask = []
        rewards_lst = []
        traj_lens = []

        td_steps_lst = []
        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            # off-policy correction: shorter horizon of td steps
            delta_td = (total_transitions - idx) // config.auto_td_steps
            td_steps = config.td_steps - delta_td
            td_steps = np.clip(td_steps, 1, 5).astype(np.int)

            # prepare the corresponding observations for bootstrapped values o_{t+k}
            game_obs = game.obs(state_index + td_steps, config.num_unroll_steps)
            rewards_lst.append(game.rewards)
            for current_index in range(state_index, state_index + config.num_unroll_steps + 1):
                td_steps_lst.append(td_steps)
                bootstrap_index = current_index + td_steps

                if bootstrap_index < traj_len:
                    value_mask.append(1)
                    beg_index = bootstrap_index - (state_index + td_steps)
                    end_index = beg_index + config.stacked_observations
                    obs = game_obs[beg_index:end_index]
                else:
                    value_mask.append(0)
                    obs = zero_obs

                value_obs_lst.append(obs)

        value_obs_lst = ray.put(value_obs_lst)
        reward_value_context = [value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst]
        return reward_value_context

    def _prepare_policy_non_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for non-reanalyzing part, just return the policy in self-play
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        child_visits = []
        traj_lens = []

        for game, state_index, idx in zip(games, state_index_lst, indices):
            traj_len = len(game)
            traj_lens.append(traj_len)

            child_visits.append(game.child_visits)

        policy_non_re_context = [state_index_lst, child_visits, traj_lens]
        return policy_non_re_context

    def _prepare_policy_re_context(self, indices, games, state_index_lst):
        """prepare the context of policies for reanalyzing part
        Parameters
        ----------
        indices: list
            transition index in replay buffer
        games: list
            list of game histories
        state_index_lst: list
            transition index in game
        """
        zero_obs = games[0].zero_obs()
        config = self.config

        with torch.no_grad():
            # for policy
            policy_obs_lst = []
            policy_mask = []  # 0 -> out of traj, 1 -> new policy
            rewards, child_visits, traj_lens = [], [], []
            for game, state_index in zip(games, state_index_lst):
                traj_len = len(game)
                traj_lens.append(traj_len)
                rewards.append(game.rewards)
                child_visits.append(game.child_visits)
                # prepare the corresponding observations
                game_obs = game.obs(state_index, config.num_unroll_steps)
                for current_index in range(state_index, state_index + config.num_unroll_steps + 1):

                    if current_index < traj_len:
                        policy_mask.append(1)
                        beg_index = current_index - state_index
                        end_index = beg_index + config.stacked_observations
                        obs = game_obs[beg_index:end_index]
                    else:
                        policy_mask.append(0)
                        obs = zero_obs
                    policy_obs_lst.append(obs)

        policy_obs_lst = ray.put(policy_obs_lst)
        policy_re_context = [policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens]
        return policy_re_context

    def make_batch(self, batch_context, ratio, weights=None):
        """prepare the context of a batch
        reward_value_context:        the context of reanalyzed value targets
        policy_re_context:           the context of reanalyzed policy targets
        policy_non_re_context:       the context of non-reanalyzed policy targets
        inputs_batch:                the inputs of batch
        weights:                     the target model weights
        Parameters
        ----------
        batch_context: Any
            batch context from replay buffer
        ratio: float
            ratio of reanalyzed policy (value is 100% reanalyzed)
        weights: Any
            the target model weights
        """
        with self.service_tracer.start_as_current_span("obtain the batch context from replay buffer"):
            # obtain the batch context from replay buffer
            game_lst, game_pos_lst, indices_lst, weights_lst, make_time_lst = batch_context
            batch_size = len(indices_lst)
            obs_lst, action_lst, mask_lst = [], [], []

        with self.service_tracer.start_as_current_span("prepare the inputs of a batch") as span:
            # prepare the inputs of a batch
            for i in range(batch_size):
                span.add_event("preparing game {} of {}".format(i, batch_size))

                # with self.service_tracer.start_as_current_span("preparing game {} of {}".format(i, batch_size)):
                game = game_lst[i]
                game_pos = game_pos_lst[i]

                _actions = game.actions[game_pos:game_pos + self.config.num_unroll_steps].tolist()
                # add mask for invalid actions (out of trajectory)
                _mask = [1. for i in range(len(_actions))]
                _mask += [0. for _ in range(self.config.num_unroll_steps - len(_mask))]

                _actions += [np.random.randint(0, game.action_space_size) for _ in range(self.config.num_unroll_steps - len(_actions))]

                # obtain the input observations
                obs_lst.append(game_lst[i].obs(game_pos_lst[i], extra_len=self.config.num_unroll_steps, padding=True))
                action_lst.append(_actions)
                mask_lst.append(_mask)

        with self.service_tracer.start_as_current_span("formalize the inputs"):
            re_num = int(batch_size * ratio)
            # formalize the input observations
            obs_lst = prepare_observation_lst(obs_lst)

            # formalize the inputs of a batch
            inputs_batch = [obs_lst, action_lst, mask_lst, indices_lst, weights_lst, make_time_lst]
            for i in range(len(inputs_batch)):
                inputs_batch[i] = np.asarray(inputs_batch[i])

        total_transitions = ray.get(self.replay_buffer.get_total_len.remote())

        # obtain the context of value targets
        reward_value_context = self._prepare_reward_value_context(indices_lst, game_lst, game_pos_lst, total_transitions)

        # 0:re_num -> reanalyzed policy, re_num:end -> non reanalyzed policy
        # reanalyzed policy
        if re_num > 0:
            # obtain the context of reanalyzed policy targets
            policy_re_context = self._prepare_policy_re_context(indices_lst[:re_num], game_lst[:re_num], game_pos_lst[:re_num])
        else:
            policy_re_context = None

        # non reanalyzed policy
        if re_num < batch_size:
            # obtain the context of non-reanalyzed policy targets
            policy_non_re_context = self._prepare_policy_non_re_context(indices_lst[re_num:], game_lst[re_num:], game_pos_lst[re_num:])
        else:
            policy_non_re_context = None

        countext = reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, weights
        self.mcts_storage.push(countext)

    def terminate(self):
        self.close_open_spans()

    def close_open_spans(self):
        self.tracer_provider.force_flush()

    def wait_if_mcts_queue_is_full(self):
        mcts_storage_len = self.mcts_storage.get_len()
        if mcts_storage_len <= 20:
            return

        with self.service_tracer.start_as_current_span("wait until mcts queue is not full") as span:
            while True:
                mcts_storage_len = self.mcts_storage.get_len()
                if mcts_storage_len <= 20:
                    break

                span.add_event('queue is full', {'num_elements': mcts_storage_len})
                time.sleep(1)

    def wait_for_the_start_signal(self):
        with self.service_tracer.start_as_current_span("waiting for the start signal"):
            while True:
                if ray.get(self.storage.get_start_signal.remote()):
                    break

                time.sleep(1)

    def run(self):
        self.wait_for_the_start_signal()

        batch_number = 0

        # start making mcts contexts to feed the GPU batch maker
        while True:
            batch_number = batch_number + 1
            with self.service_tracer.start_as_current_span("process batch #{}".format(batch_number)):
                with self.service_tracer.start_as_current_span("prepare the context") as span:
                    ray_data_lst = [self.storage.get_counter.remote(), self.storage.get_target_weights.remote()]
                    trained_steps, target_weights = ray.get(ray_data_lst)

                    beta = self.beta_schedule.value(trained_steps)
                    # obtain the batch context from replay buffer
                    batch_context = ray.get(self.replay_buffer.prepare_batch_context.remote(self.config.batch_size, beta))

                    span.set_attribute("trained_steps", trained_steps)
                    span.set_attribute("batch size", self.config.batch_size)

                # break
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    with self.service_tracer.start_as_current_span("training is done"):
                        time.sleep(30)
                        break

                new_model_index = trained_steps // self.config.target_model_interval
                if new_model_index > self.last_model_index:
                    with self.service_tracer.start_as_current_span("update the model index"):
                        self.last_model_index = new_model_index
                else:
                    target_weights = None

                with self.service_tracer.start_as_current_span("create the batch"):
                    self.wait_if_mcts_queue_is_full()

                    # Observation will be deleted if replay buffer is full. (They are stored in the ray object store)
                    try:
                        self.make_batch(batch_context, self.config.revisit_policy_search_rate, weights=target_weights)
                    except Exception as err:
                        with self.service_tracer.start_as_current_span("an exception was triggered while making batch") as span:
                            span.record_exception(err)

                            print('Data is deleted...')
                            time.sleep(0.1)
