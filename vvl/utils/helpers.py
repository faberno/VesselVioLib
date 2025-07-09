import platform
from multiprocessing import cpu_count, get_context
from math import floor
from itertools import chain
import concurrent.futures as cf
import os
import sys


def get_OS():
    sys_os = platform.system()
    return sys_os


def unix_check():
    sys_os = get_OS()
    return sys_os != "Windows"

def multiprocessing_input(function, list_size, workers: int = None, sublist=False):
    """
    The variables used in the called function need to be global.

    function: The function to be used for multiprocessing

    list_size: The size of the list that is to be processed.
        This list will be processed in steps based on worker count.

    workers: The amount of CPUs to be called on. Defaults to maximum available based on mp.cpu_count()
    """
    if not workers:
        workers = cpu_count()

    futures = []
    steps = floor(list_size / workers)

    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("fork")
    ) as executor:
        for i in range(workers):
            bottom = i * steps
            top = bottom + steps if i != workers - 1 else list_size
            futures.append(executor.submit(function, bottom, top))

    if sublist:
        results = chain.from_iterable([f.result()] for f in cf.as_completed(futures))
    else:
        results = chain.from_iterable(f.result() for f in cf.as_completed(futures))

    return list(results)


def get_ext(file):
    ext = os.path.splitext(file)[-1]
    return ext


def simplify_graph(
    g,
    reduced_edges,
    volumes,
    surface_areas,
    lengths,
    tortuosities,
    radii_avg,
    radii_max,
    radii_min,
    radii_SD,
    vis_radii,
    coords_lists,
    radii_lists,
):
    g.delete_edges(g.es())
    g.add_edges(
        reduced_edges,
        {
            "volume": volumes,
            "surface_area": surface_areas,
            "length": lengths,
            "tortuosity": tortuosities,
            "radius_avg": radii_avg,
            "radius_max": radii_max,
            "radius_min": radii_min,
            "radius_SD": radii_SD,
            "coords_list": coords_lists,
            "radii_list": radii_lists,
            "vis_radius": vis_radii,
        },
    )
    g.delete_vertices(g.vs.select(_degree=0))
    return

# Find out whether the program is running from the app or terminal
def get_cwd():
    try:
        wd = sys._MEIPASS
    except AttributeError:
        wd = os.getcwd()
    return wd
