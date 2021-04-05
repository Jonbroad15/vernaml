import argparse
import os
import sys
import networkx as nx
from numpy import random
from tqdm import tqdm
from random import shuffle

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.slice import *
from prepare_data.interfaces import *

from tools.graph_utils import bfs_expand, dangle_trim
from custom_pytools.os_tools import listdir_fullpath
from utils.graph_io import load_json, dump_json

# def balance(graph_dir, interaction):
    # # Compute number of nodes
    # num_nodes = 0
    # binding_nodes = 0
    # for graph_file in tqdm(listdir_fullpath(args.input_dir)):
        # g = load_graph(graph_file)
        # num_nodes += len(g.nodes)
        # binding_nodes += len([n for n, d in g.nodes.data()\
                                        # if d['binding_' + interaction] is not None])

def undersample_graph_dir(graph_dir, interaction,
                            threshold=1.2,
                            remove_size = 0.3):
    """
    Remove graphs from graph_dir if they do not have a given interaction

    :param threshold: (int) threshold for how much bigger the complement set is.
    """

    print("Removing graphs without binding interactions")
    NB_nodes, binding_nodes = remove_graphs(graph_dir, interaction)

    if binding_nodes*threshold > NB_nodes:
        return
    print("Remove nodes without binding interactions")
    remove_nodes(graph_dir, interaction, NB_nodes, binding_nodes,
            threshold, remove_size=remove_size)

def remove_graphs(graph_dir, interaction):

    num_nodes = 0
    binding_nodes = 0
    for graph_file in tqdm(listdir_fullpath(graph_dir)):
        g = load_json(graph_file)
        g_binding_nodes = len([n for n, d in g.nodes.data()\
                            if d['binding_' + interaction] is not None])
        binding_nodes += g_binding_nodes
        if g_binding_nodes == 0:
            os.remove(graph_file)
            continue
        num_nodes += len(g.nodes)

    return num_nodes - binding_nodes, binding_nodes

def remove_nodes(graph_dir, interaction, num_NB, num_binding, threshold,
                    remove_size=0.3):
    """
    Undersample nodes from graphs to produce a balanced dataset
    """
    # Count number of nodes needed to remove

    remove_size = int( (num_NB - num_binding) / len(os.listdir(graph_dir)) )
    remove_size -= 10

    while num_binding*threshold < num_NB:
        print(f"Binding: {num_binding} \t Non-Binding {num_NB}")
        small = 0
        for graph_file in tqdm(listdir_fullpath(graph_dir)):
            g = load_json(graph_file)
            NB_nodes = [n for n, d in g.nodes.data()\
                    if d['binding_' + interaction] is None]
            if len(g.nodes) < 20 or len(NB_nodes) == 0:
                small += 1
                continue
            shuffle(NB_nodes)
            trash = NB_nodes[:remove_size]
            g.remove_nodes_from(trash)
            num_NB -= len(trash)
            dump_json(graph_file, g)
        if small + 10 > len(os.listdir(graph_dir)):
            break



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('interaction')
    parser.add_argument('-t', '--threshold', type=float, default=1.2)
    parser.add_argument('-r', '--remove_size', type=float, default=0.3)
    args = parser.parse_args()
    undersample_graph_dir(args.dir, args.interaction,
            threshold=args.threshold, remove_size=args.remove_size)


if __name__ == '__main__':
    main()
