# This file defines a decorator '@log_to()' that logs every call to a
# function, along with the arguments that function was called with. It
# takes a logging function, which is any function that accepts a
# string and does something with it. A good choice is the debug
# function from the logging module. A second decorator '@logdebug' is
# provided that uses 'logging.debug' as the logger.

from functools import wraps
from inspect import getcallargs, getargspec, getfullargspec
from collections import OrderedDict, Iterable
from itertools import *
import logging
import time

# from logdecorator import log_on_start, log_on_end, log_on_error
# from logging import DEBUG, ERROR, INFO


def flatten(l):
    """Flatten a list (or other iterable) recursively"""
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def getargnames(func):
    """Return an iterator over all arg names, including nested arg names and varargs.

    Goes in the order of the functions argspec, with varargs and
    keyword args last if present."""
    (argnames, varargname, kwargname, _, _, _, _) = getfullargspec(func)
    return chain(flatten(argnames), filter(None, [varargname, kwargname]))


def getcallargs_ordered(func, *args, **kwargs):
    """Return an OrderedDict of all arguments to a function.

    Items are ordered by the function's argspec."""
    argdict = getcallargs(func, *args, **kwargs)
    return OrderedDict((name, argdict[name]) for name in getargnames(func))


def describe_call(func, *args, **kwargs):
    try:
        yield "Calling %s with args:" % func.__name__
        for argname, argvalue in getcallargs_ordered(func, *args, **kwargs).items():
            yield "\t%s = %s" % (argname, repr(argvalue))
    except:
        yield None


def log_to(logger_func):
    """A decorator to log every call to function (function name and arg values).

    logger_func should be a function that accepts a string and logs it
    somewhere. The default is logging.debug.

    If logger_func is None, then the resulting decorator does nothing.
    This is much more efficient than providing a no-op logger
    function: @log_to(lambda x: None).
    """
    if logger_func is not None:

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for line in describe_call(func, *args, **kwargs):
                    logger_func(line)
                return func(*args, **kwargs)

            return wrapper

    else:
        decorator = lambda x: x
    return decorator


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            logging.info(
                "End of {}. Elapsed time: {:0.2f} sec.".format(
                    method.__name__, (te - ts)
                )
            )
        return result

    return timed


def debug(fn):
    def wrapper(*args, **kwargs):
        logger.debug("Entering {:s}...".format(fn.__name__))
        result = fn(*args, **kwargs)
        logger.debug("Finished {:s}.".format(fn.__name__))
        return result

    return wrapper


logging_arg = log_to(logging.info)


@logging_arg
@timeit
def myfunc(a, b, c, *args, **kwargs):
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        filename="testing.log",
        filemode="w",
    )
    myfunc(1, 2, 3, 4, 5, 6, x=7, y=8, z=9, g="blarg", f=lambda x: x + 2)
