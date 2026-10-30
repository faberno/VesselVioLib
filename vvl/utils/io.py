import os
import pickle
import gc
import numpy as np
import networkx as nx

def save_graph(
    g, filename, results_dir, main_thread=True, caching=False, verbose=False, save_pickle=False
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
    

    # Full igraph object incl. edge positions, needed to reconstruct volumes from the graph later
    if save_pickle:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, filename + ".pkl"), "wb") as f:
            pickle.dump(g, f)


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
    # try:
    print("Vertices:", g.vcount(), "Edges:", g.ecount())
    try:
        if g.vcount() > 0:
            with open(file, "wb") as f:
                g.write_graphml(f)
        else:
            print("Empty graph, skipped:", file)
    except Exception as e:
        print("Save failed:", e)
    # except:
    #     g_ = g.to_networkx()
    #     nx.write_graphml(g_, file)
    finally:
        gc.collect()
    return
