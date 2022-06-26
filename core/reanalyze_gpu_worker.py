import time

import core.ctree.cytree as cytree
import numpy as np
import pydevd_pycharm
import ray
import torch
from torch.cuda.amp import autocast as autocast

from __refactored__.tracing.opentelemetry import make_tracer_provider
from core.mcts import MCTS
from core.model import concat_output, concat_output_value
from core.utils import prepare_observation_lst


@ray.remote
class BatchWorker_GPU(object):
    def __init__(self, worker_id, replay_buffer, storage, batch_storage, mcts_storage, config):
        """GPU Batch Worker for reanalyzing targets, see Appendix.
        receive the context from CPU maker and deal with GPU overheads
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
        # pydevd_pycharm.settrace('localhost', port=5678, stdoutToServer=True, stderrToServer=True)

        self.tracer_provider = make_tracer_provider("smartfighters-efficientZero-reanalyze-GPU-worker-{}".format(worker_id))
        self.service_tracer = self.tracer_provider.get_tracer(__name__)

        with self.service_tracer.start_as_current_span("Initialize reanalyze GPU worker"):
            self.replay_buffer = replay_buffer
            self.config = config
            self.worker_id = worker_id

            self.model = config.get_uniform_network()
            self.model.to(config.device)
            self.model.eval()

            self.mcts_storage = mcts_storage
            self.storage = storage
            self.batch_storage = batch_storage

            self.last_model_index = 0

    def _prepare_reward_value(self, reward_value_context):
        """prepare reward and value targets from the context of rewards and values
        """
        value_obs_lst, value_mask, state_index_lst, rewards_lst, traj_lens, td_steps_lst = reward_value_context
        value_obs_lst = ray.get(value_obs_lst)
        device = self.config.device
        batch_size = len(value_obs_lst)

        batch_values, batch_value_prefixs = [], []
        with torch.no_grad():
            value_obs_lst = prepare_observation_lst(value_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)
                m_obs = torch.from_numpy(value_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            # concat the output slices after model inference
            if self.config.use_root_value:
                # use the root values from MCTS
                # the root values have limited improvement but require much more GPU actors;
                _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
                value_prefix_pool = value_prefix_pool.squeeze().tolist()
                policy_logits_pool = policy_logits_pool.tolist()
                roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
                noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
                roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

                roots_values = roots.get_values()
                value_lst = np.array(roots_values)
            else:
                # use the predicted values
                value_lst = concat_output_value(network_output)

            # get last state value
            value_lst = value_lst.reshape(-1) * (np.array([self.config.discount for _ in range(batch_size)]) ** td_steps_lst)
            value_lst = value_lst * np.array(value_mask)
            value_lst = value_lst.tolist()

            horizon_id, value_index = 0, 0
            for traj_len_non_re, reward_lst, state_index in zip(traj_lens, rewards_lst, state_index_lst):
                # traj_len = len(game)
                target_values = []
                target_value_prefixs = []

                value_prefix = 0.0
                base_index = state_index
                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    bootstrap_index = current_index + td_steps_lst[value_index]
                    # for i, reward in enumerate(game.rewards[current_index:bootstrap_index]):
                    for i, reward in enumerate(reward_lst[current_index:bootstrap_index]):
                        value_lst[value_index] += reward * self.config.discount ** i

                    # reset every lstm_horizon_len
                    if horizon_id % self.config.lstm_horizon_len == 0:
                        value_prefix = 0.0
                        base_index = current_index
                    horizon_id += 1

                    if current_index < traj_len_non_re:
                        target_values.append(value_lst[value_index])
                        # Since the horizon is small and the discount is close to 1.
                        # Compute the reward sum to approximate the value prefix for simplification
                        value_prefix += reward_lst[current_index]  # * config.discount ** (current_index - base_index)
                        target_value_prefixs.append(value_prefix)
                    else:
                        target_values.append(0)
                        target_value_prefixs.append(value_prefix)
                    value_index += 1

                batch_value_prefixs.append(target_value_prefixs)
                batch_values.append(target_values)

        batch_value_prefixs = np.asarray(batch_value_prefixs)
        batch_values = np.asarray(batch_values)
        return batch_value_prefixs, batch_values

    def _prepare_policy_re(self, policy_re_context):
        """prepare policy targets from the reanalyzed context of policies
        """
        batch_policies_re = []
        if policy_re_context is None:
            return batch_policies_re

        policy_obs_lst, policy_mask, state_index_lst, indices, child_visits, traj_lens = policy_re_context
        policy_obs_lst = ray.get(policy_obs_lst)
        batch_size = len(policy_obs_lst)
        device = self.config.device

        with torch.no_grad():
            policy_obs_lst = prepare_observation_lst(policy_obs_lst)
            # split a full batch into slices of mini_infer_size: to save the GPU memory for more GPU actors
            m_batch = self.config.mini_infer_size
            slices = np.ceil(batch_size / m_batch).astype(np.int_)
            network_output = []
            for i in range(slices):
                beg_index = m_batch * i
                end_index = m_batch * (i + 1)

                m_obs = torch.from_numpy(policy_obs_lst[beg_index:end_index]).to(device).float() / 255.0
                if self.config.amp_type == 'torch_amp':
                    with autocast():
                        m_output = self.model.initial_inference(m_obs)
                else:
                    m_output = self.model.initial_inference(m_obs)
                network_output.append(m_output)

            _, value_prefix_pool, policy_logits_pool, hidden_state_roots, reward_hidden_roots = concat_output(network_output)
            value_prefix_pool = value_prefix_pool.squeeze().tolist()
            policy_logits_pool = policy_logits_pool.tolist()

            roots = cytree.Roots(batch_size, self.config.action_space_size, self.config.num_simulations)
            noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() for _ in range(batch_size)]
            roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
            # do MCTS for a new policy with the recent target model
            MCTS(self.config).search(roots, self.model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()
            policy_index = 0
            for state_index, game_idx in zip(state_index_lst, indices):
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    distributions = roots_distributions[policy_index]

                    if policy_mask[policy_index] == 0:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                    else:
                        # game.store_search_stats(distributions, value, current_index)
                        sum_visits = sum(distributions)
                        policy = [visit_count / sum_visits for visit_count in distributions]
                        target_policies.append(policy)

                    policy_index += 1

                batch_policies_re.append(target_policies)

        batch_policies_re = np.asarray(batch_policies_re)
        return batch_policies_re

    def _prepare_policy_non_re(self, policy_non_re_context):
        """prepare policy targets from the non-reanalyzed context of policies
        """
        batch_policies_non_re = []
        if policy_non_re_context is None:
            return batch_policies_non_re

        state_index_lst, child_visits, traj_lens = policy_non_re_context
        with torch.no_grad():
            # for policy
            policy_mask = []  # 0 -> out of traj, 1 -> old policy
            # for game, state_index in zip(games, state_index_lst):
            for traj_len, child_visit, state_index in zip(traj_lens, child_visits, state_index_lst):
                # traj_len = len(game)
                target_policies = []

                for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
                    if current_index < traj_len:
                        target_policies.append(child_visit[current_index])
                        policy_mask.append(1)
                    else:
                        target_policies.append([0 for _ in range(self.config.action_space_size)])
                        policy_mask.append(0)

                batch_policies_non_re.append(target_policies)
        batch_policies_non_re = np.asarray(batch_policies_non_re)
        return batch_policies_non_re

    def _prepare_target_gpu(self):
        input_countext = self.mcts_storage.pop()
        if input_countext is None:
            with self.service_tracer.start_as_current_span("no context is present on the mcts storage"):
                time.sleep(1)
        else:
            with self.service_tracer.start_as_current_span("prepare target batch"):
                reward_value_context, policy_re_context, policy_non_re_context, inputs_batch, target_weights = input_countext
                if target_weights is not None:
                    with self.service_tracer.start_as_current_span("evaluating model"):
                        self.model.load_state_dict(target_weights)
                        self.model.to(self.config.device)
                        self.model.eval()

                with self.service_tracer.start_as_current_span("prepare rewards"):
                    # target reward, value
                    batch_value_prefixs, batch_values = self._prepare_reward_value(reward_value_context)

                with self.service_tracer.start_as_current_span("prepare target policy"):
                    # target policy
                    batch_policies_re = self._prepare_policy_re(policy_re_context)
                    batch_policies_non_re = self._prepare_policy_non_re(policy_non_re_context)
                    batch_policies = np.concatenate([batch_policies_re, batch_policies_non_re])

                targets_batch = [batch_value_prefixs, batch_values, batch_policies]
                # a batch contains the inputs and the targets; inputs is prepared in CPU workers

                with self.service_tracer.start_as_current_span("push batch to batch storage") as push_span:
                    self.batch_storage.push([inputs_batch, targets_batch])

                    push_span.set_attribute('sf.batch_storage.num_elements', self.batch_storage.get_len())

    def terminate(self):
        self.close_open_spans()

    def close_open_spans(self):
        self.tracer_provider.force_flush()

    def wait_for_the_start_signal(self):
        with self.service_tracer.start_as_current_span("waiting for the start signal"):
            while True:
                if ray.get(self.storage.get_start_signal.remote()):
                    break

                time.sleep(1)

    def run(self):
        self.wait_for_the_start_signal()

        with self.service_tracer.start_as_current_span("running the reanalyze gpu worker "):
            while True:
                trained_steps = ray.get(self.storage.get_counter.remote())
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    with self.service_tracer.start_as_current_span("the max number of trained steps has been reached") as span:
                        span.set_attribute("trained_steps", trained_steps)
                        span.set_attribute("max_steps", self.config.training_steps + self.config.last_steps)

                        time.sleep(30)
                        break

                with self.service_tracer.start_as_current_span("work load for num. trained steps: {}".format(trained_steps)):
                    self._prepare_target_gpu()
