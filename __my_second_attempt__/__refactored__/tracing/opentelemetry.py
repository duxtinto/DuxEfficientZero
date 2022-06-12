from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, SimpleSpanProcessor
)


def setup_ray_tracing():
    trace_provider = trace.get_tracer_provider()

    if trace_provider is None:
        tracer_provider = make_trace_provider('smartfighters-efficientZero-ray')
        trace.set_tracer_provider(tracer_provider)


def make_trace_provider(service_name: str) -> TracerProvider:
    jaeger_exporter = JaegerExporter(
        # configure agent
        agent_host_name='127.0.0.1',
        agent_port=6831,
        # optional: configure also collector
        collector_endpoint='http://localhost:14268/api/traces?format=jaeger.thrift',
        # username=xxxx, # optional
        # password=xxxx, # optional
    )

    # span_processor = SimpleSpanProcessor(jaeger_exporter)
    span_processor = BatchSpanProcessor(jaeger_exporter)

    resource = Resource.create({SERVICE_NAME: service_name})

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)

    return tracer_provider
