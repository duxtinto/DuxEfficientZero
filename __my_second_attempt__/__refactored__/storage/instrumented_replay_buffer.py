from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract
from opentelemetry.trace import Tracer
from ray.actor import ActorClass

from __refactored__.storage.replay_buffer import ReplayBuffer
from __refactored__.tracing.opentelemetry import make_trace_provider


class InstrumentedReplayBuffer(object):
    """
    Instrumented version of the replay buffer

    As the original replay buffer actor can be used in multiple runs,
    we need to make sure that we always send the span to the right trace.
    """
    span_context: Context
    service_tracer: Tracer
    remote_replay_buffer: ActorClass

    def __init__(self, span_context: dict, remote_replay_buffer: ActorClass):
        # pydevd_pycharm.settrace('localhost', port=5676, stdoutToServer=True, stderrToServer=True)

        self.span_context: Context = extract(span_context)

        self.service_tracer = trace.get_tracer(
            __name__,
            tracer_provider=(make_trace_provider("smartfighters-efficientZero-replay-buffer"))
        )

        with self.service_tracer.start_as_current_span(
                "initialize replay buffer",
                context=self.span_context
        ):
            self.remote_replay_buffer = remote_replay_buffer

    def save_pools(self, pools, gap_step):
        with self.service_tracer.start_as_current_span(
                "save game pools",
                context=self.span_context
        ):
            self.remote_replay_buffer.save_pools.remote(pools, gap_step)

    def save_game(self, game, end_tag, gap_steps, priorities=None):
        """Save a game history block
        Parameters
        ----------
        game: Any
            a game history block
        end_tag: bool
            True -> the game is finished. (always True)
        gap_steps: int
            if the game is not finished, we only save the transitions that can be computed
        priorities: list
            the priorities corresponding to the transitions in the game history
        """
        with self.service_tracer.start_as_current_span(
                "save game",
                context=self.span_context
        ):
            self.remote_replay_buffer.save_game.remote(game, end_tag, gap_steps, priorities)

    def get_game(self, idx):
        with self.service_tracer.start_as_current_span(
                "get game",
                context=self.span_context
        ) as span:
            game = self.remote_replay_buffer.get_game.remote(idx)

            span.set_attribute("game_idx", idx)
            span.set_attribute("game", str(game))
            
            return game

    def prepare_batch_context(self, batch_size, beta):
        """Prepare a batch context that contains:
        game_lst:               a list of game histories
        game_pos_lst:           transition index in game (relative index)
        indices_lst:            transition index in replay buffer
        weights_lst:            the weight concering the priority
        make_time:              the time the batch is made (for correctly updating replay buffer when data is deleted)
        Parameters
        ----------
        batch_size: int
            batch size
        beta: float
            the parameter in PER for calculating the priority
        """
        with self.service_tracer.start_as_current_span(
                "prepare batch context",
                context=self.span_context
        ):
            return self.remote_replay_buffer.prepare_batch_context.remote(batch_size, beta)

    def update_priorities(self, batch_indices, batch_priorities, make_time):
        with self.service_tracer.start_as_current_span(
                "update priorities of the existing transitions",
                context=self.span_context
        ):
            self.remote_replay_buffer.update_priorities.remote(batch_indices, batch_priorities, make_time)

    def remove_to_fit(self):
        with self.service_tracer.start_as_current_span(
                "remove old transitions",
                context=self.span_context
        ):
            self.remote_replay_buffer.remove_to_fit.remote()

    def clear_buffer(self):
        with self.service_tracer.start_as_current_span(
                "clear buffer",
                context=self.span_context
        ):
            self.remote_replay_buffer.clear_buffer.remote()

    def size(self):
        with self.service_tracer.start_as_current_span(
                "get size",
                context=self.span_context
        ):
            self.remote_replay_buffer.size.remote()

    def episodes_collected(self):
        with self.service_tracer.start_as_current_span(
                "episodies collected",
                context=self.span_context
        ):
            return self.remote_replay_buffer.episodes_collected.remote()

    def get_batch_size(self):
        return self.remote_replay_buffer.get_batch_size.remote()

    def get_priorities(self):
        return self.remote_replay_buffer.get_priorities.remote()

    def get_total_len(self):
        return self.remote_replay_buffer.get_total_len.remote()
