import functools


def instrumented_class(cls):
    """Instrument a class with a tracing decorator."""

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        return cls(*args, **kwargs)

    return wrapper
