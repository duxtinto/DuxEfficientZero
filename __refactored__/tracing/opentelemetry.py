from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from __refactored__.tracing.SimpleTerminableSpanProcessor import SimpleTerminableSpanProcessor
from opentelemetry import trace


def setup_ray_tracing():
    trace_provider = trace.get_tracer_provider()

    if trace_provider is None:
        tracer_provider = make_tracer_provider('smartfighters-efficientZero-ray')
        trace.set_tracer_provider(tracer_provider)


def make_tracer_provider(service_name: str) -> TracerProvider:
    jaeger_exporter = JaegerExporter(
        agent_host_name='127.0.0.1',
        agent_port=6831,
        collector_endpoint='http://localhost:14268/api/traces?format=jaeger.thrift',
    )

    span_processor = SimpleTerminableSpanProcessor(jaeger_exporter)
    # span_processor = BatchSpanProcessor(jaeger_exporter)

    resource = Resource.create({SERVICE_NAME: service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)

    return tracer_provider
