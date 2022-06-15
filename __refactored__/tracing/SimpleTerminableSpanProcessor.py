import logging
import typing

from opentelemetry.context import Context, attach, set_value, _SUPPRESS_INSTRUMENTATION_KEY, detach
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class SimpleTerminableSpanProcessor(SpanProcessor):
    """Simple SpanProcessor implementation.

    SimpleSpanProcessor is an implementation of `SpanProcessor` that
    passes ended spans directly to the configured `SpanExporter`.
    """

    def __init__(self, span_exporter: SpanExporter):
        self.span_exporter = span_exporter

        self.open_spans = {}

    def on_start(
            self, span: Span, parent_context: typing.Optional[Context] = None
    ) -> None:
        self.open_spans[span.get_span_context().span_id] = span

    def on_end(self, span: ReadableSpan) -> None:
        if not span.context.trace_flags.sampled:
            return
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            self.span_exporter.export((span,))
            del self.open_spans[span.get_span_context().span_id]
        # pylint: disable=broad-except
        except Exception:
            logger.exception("Exception while exporting Span.")
        detach(token)

    def shutdown(self) -> None:
        """
        Close any open spans before shutting down the processor, so they are not lost.
        """
        for open_span in self.open_spans.copy().values():
            open_span.end()

        self.span_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # pylint: disable=unused-argument
        for open_span in self.open_spans.copy().values():
            open_span.end()

        return True
