import argparse
import os
import sys
import networkx as nx

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.interfaces import *
from tools.graph_utils import bfs_expand, dangle_trim

def connect_components(g):
    """
    Check if the graph contains disconnected components, connect them if it can be done
    with just a few edges. else return components as their own graphs
    :param g: networkx graph
    :return graphs: list of connected graphs
    """
    graphs = []

    if nx.is_connected(g):
        print(list(g.nodes)[0][0], ': connected')
        return [g]
    else:
        print(f'\n\nedges: {list(g.nodes)[0][0]} \n')
        for e in g.edges:
            print(e)
        print('\n\n', list(g.nodes)[0][0], ': not connected \t components:', nx.number_connected_components(g))

    for n, nbrs in g.adj.items():
        for nbr, eattr in nbrs.items():
            print(eattr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-graph')
    parser.add_argument('-dir')
    args = parser.parse_args()

    i = 0
    if args.dir:
        for graph_file in os.listdir(args.dir):
            if i == 10: break
            i += 1
            if '.nx' not in graph_file: continue
            g = nx.read_gpickle(os.path.join(args.dir, graph_file))
            connect_components(g)
    elif args.graph:
        g = nx.read_gpickle(args.graph)
        connect_components(g)

if __name__ == '__main__':
    main()
