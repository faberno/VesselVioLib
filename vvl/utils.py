import enum
import logging
import platform
from time import perf_counter
import os
import sys

logger = logging.getLogger(__name__)

class GraphType(enum.Enum):
    """Enum to represent different types of graphs."""
    CENTERLINE = "centerline"
    BRANCH = "branch"

def measure_time(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        logger.info(f"Calling function {func.__name__} ...")

        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()

        logger.debug(f"Function {func.__name__} executed in {end_time - start_time:.3f}s")
        return result
    return wrapper


def get_cwd():
    """Get current working directory and find out whether the program is running from the app or terminal"""
    try:
        wd = sys._MEIPASS
    except AttributeError:
        wd = os.getcwd()
    return wd

def get_os():
    sys_os = platform.system()
    return sys_os


def is_unix():
    sys_os = get_os()
    return sys_os != "Windows"