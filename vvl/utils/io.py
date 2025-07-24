import os

import numpy as np

def save_graph(
    g, filename, results_dir, main_thread=True, caching=False, verbose=False
):
    if verbose:
        print("Saving graph...", end="\r")

    if not g.vs():
        return

    # Save Coords as XYZ values
    points = np.array(g.vs["v_coords"])
    g.vs["X"] = points[:, 2]
    g.vs["Y"] = points[:, 1]
    g.vs["Z"] = points[:, 0]
    

    if main_thread:
        del g.vs["v_radius"]
        del g.vs["vis_radius"]
    del g.es["radii_list"]
    del g.es["coords_list"]
    del g.es["original_edge_positions"]
    del g.es["original_edge_paths"]
    del g.vs["v_coords"]

    if caching:
        return g

    # Get the dir and name for our graph.
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    results_dir = os.path.join(results_dir, "Graphs")
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    file = os.path.join(results_dir, filename + "." + "graphml")

    # save the graph
    g.write(file)

    return
