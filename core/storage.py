from typing import Optional

import ray
from opentelemetry.context import Context
from opentelemetry.trace import Status, StatusCode

from __refactored__.queue.queue import Queue
from __refactored__.tracing.opentelemetry import make_tracer_provider


class QueueStorage(object):
    def __init__(self, name='', threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        # pydevd_pycharm.settrace('localhost', port=5674, stdoutToServer=True, stderrToServer=True)

        # name = random.choices(string.ascii_lowercase + string.digits, k=5)
        self.name = name
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch, _ray_trace_ctx: Optional[Context] = None):
        tracer_provider = make_tracer_provider("smartfighters-efficientZero-{}-queue-storage".format(self.name))
        service_tracer = tracer_provider.get_tracer(__name__)

        with service_tracer.start_as_current_span("trying to push batch to queue") as span:
            num_elements = self.queue.qsize()
            is_queue_full = num_elements > self.threshold

            span.set_attributes({
                'sf.queue.is_full': is_queue_full,
                'num_elements': num_elements
            })

            if is_queue_full:
                span.set_status(Status(StatusCode.ERROR, 'queue is full'))
                span.set_attribute('sf.queue.num_elements', num_elements)
                return

            self.queue.put(batch)

    def pop(self, _ray_trace_ctx: Optional[Context] = None):
        tracer_provider = make_tracer_provider("smartfighters-efficientZero-{}-queue-storage".format(self.name))
        service_tracer = tracer_provider.get_tracer(__name__)

        with service_tracer.start_as_current_span("trying to pop batch from queue") as span:
            is_queue_empty = self.queue.qsize() <= 0

            span.set_attribute('sf.queue.is_empty', is_queue_empty)

            if is_queue_empty:
                return None

            return self.queue.get()

    def get_len(self, _ray_trace_ctx: Optional[Context] = None):
        return self.queue.qsize()


@ray.remote
class SharedStorage(object):
    def __init__(self, model, target_model):
        """Shared storage for models and others
        Parameters
        ----------
        model: any
            models for self-play (update every checkpoint_interval)
        target_model: any
            models for reanalyzing (update every target_model_interval)
        """
        # pydevd_pycharm.settrace('localhost', port=5674, stdoutToServer=True, stderrToServer=True)

        self.tracer_provider = make_tracer_provider("smartfighters-efficientZero-shared-storage")
        self.service_tracer = self.tracer_provider.get_tracer(__name__)

        self.step_counter = 0
        self.test_counter = 0
        self.model = model
        self.target_model = target_model
        self.ori_reward_log = []
        self.reward_log = []
        self.reward_max_log = []
        self.test_dict_log = {}
        self.eps_lengths = []
        self.eps_lengths_max = []
        self.temperature_log = []
        self.visit_entropies_log = []
        self.priority_self_play_log = []
        self.distributions_log = {}
        self.start = False

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def get_target_weights(self):
        return self.target_model.get_weights()

    def set_target_weights(self, weights):
        return self.target_model.set_weights(weights)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_len_max, eps_ori_reward, eps_reward, eps_reward_max, temperature,
                             visit_entropy, priority_self_play, distributions):
        self.eps_lengths.append(eps_len)
        self.eps_lengths_max.append(eps_len_max)
        self.ori_reward_log.append(eps_ori_reward)
        self.reward_log.append(eps_reward)
        self.reward_max_log.append(eps_reward_max)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)
        self.priority_self_play_log.append(priority_self_play)

        for key, val in distributions.items():
            if key not in self.distributions_log.keys():
                self.distributions_log[key] = []
            self.distributions_log[key] += val

    def add_test_log(self, test_counter, test_dict):
        self.test_counter = test_counter
        for key, val in test_dict.items():
            if key not in self.test_dict_log.keys():
                self.test_dict_log[key] = []
            self.test_dict_log[key].append(val)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            ori_reward = sum(self.ori_reward_log) / len(self.ori_reward_log)
            reward = sum(self.reward_log) / len(self.reward_log)
            reward_max = sum(self.reward_max_log) / len(self.reward_max_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            eps_lengths_max = sum(self.eps_lengths_max) / len(self.eps_lengths_max)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)
            priority_self_play = sum(self.priority_self_play_log) / len(self.priority_self_play_log)
            distributions = self.distributions_log

            self.ori_reward_log = []
            self.reward_log = []
            self.reward_max_log = []
            self.eps_lengths = []
            self.eps_lengths_max = []
            self.temperature_log = []
            self.visit_entropies_log = []
            self.priority_self_play_log = []
            self.distributions_log = {}

        else:
            ori_reward = None
            reward = None
            reward_max = None
            eps_lengths = None
            eps_lengths_max = None
            temperature = None
            visit_entropy = None
            priority_self_play = None
            distributions = None

        if len(self.test_dict_log) > 0:
            test_dict = self.test_dict_log

            self.test_dict_log = {}
            test_counter = self.test_counter
        else:
            test_dict = None
            test_counter = None

        return ori_reward, reward, reward_max, eps_lengths, eps_lengths_max, test_counter, test_dict, temperature, visit_entropy, priority_self_play, distributions
