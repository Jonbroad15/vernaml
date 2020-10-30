import argparse
import os
import sys
import networkx as nx
from numpy import random

scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.interfaces import *
from tools.graph_utils import bfs_expand, dangle_trim

def get_pbid(g):

    return list(g.nodes)[0][0]

def get_num_nodes(g):

    return len(list(g.nodes))

def print_component_info(g, i):


    pbid = list(g.nodes)[0][0]
    # print(f'\n\nedges: {list(g.nodes)[0][0]} \n')
    # for e in g.edges:
    # print(e)


    if nx.is_connected(g):
        # print(pbid, i, ': connected')
        pass
    else:
        print("\n\n WARNING \n",
                pbid, i, ': not connected \t components:', nx.number_connected_components(g))

    # for n, nbrs in g.adj.items():
        # for nbr, eattr in nbrs.items():
            # print(eattr)

   # Remove nodes that have no other interactions other than backbone

def balance_complement(interface, complement):
    """
    Count the number of nodes in the interface
    Do BFS_expand on a random node in the complement until number of nodes is equal
    :param interface: nx graph
    :param complement: nx graph
    :return balanced_complement: nx_graph the same size as the interface
    """

    num_nodes = len(list(interface.nodes))

    random_node_idx = int(random.rand()*len(list(complement.nodes)))

    balanced_complement_nodes = [list(complement.nodes)[random_node_idx]]

    while(len(balanced_complement_nodes) < num_nodes):
        balanced_complement_nodes = bfs_expand(complement, balanced_complement_nodes, depth=1)

    balanced_complement = complement.subgraph(balanced_complement_nodes).copy()

    size_difference = abs( get_num_nodes(balanced_complement) - get_num_nodes(interface) )
    if size_difference > 10:
        print("\n WARNING Graph: ", get_pbid(interface),
                "has size_difference: ", size_difference, '\n\n')

    return balanced_complement

def balance_complement_all(interface_dir, complement_dir, output_dir):

    i = 0
    for graph_file in os.listdir(interface_dir):
        # Loop control
        # if i == 30: break
        # i += 1
        if '.nx' not in graph_file: continue

        # read input and compute function
        try:
            interface = nx.read_gpickle(os.path.join(interface_dir, graph_file))
            complement = nx.read_gpickle(os.path.join(complement_dir, graph_file))
            print('Balancing', graph_file, '...')
        except FileNotFoundError:
            print('\nWARNING, complement graph not found for: ', graph_file, '\n\n')
            continue
        balanced_complement = balance_complement(interface, complement)

        # Write output
        nx.write_gpickle(balanced_complement, os.path.join(output_dir, graph_file))

def connect_components(g, g_native, depth=2):
    """
    Check if the graph contains disconnected components, connect them if it can be done
    with just a few edges. else return components as their own graphs
    Only coded for depth = 2.
    for bigger depths use recursive calls
    for a depth of just one do not use a nested neighbour search
    :param g: networkx graph
    :param g_native: native graph before interface was chopped
    :return graphs: list of connected graphs
    """
    graphs = []

    depth = int(depth/2)
    # if depth % 2 != 0:
        # inner_depth = int(depth/2)
        # outer_depth = int(depth/2) + 1
    # else:
        # inner_depth, outer_depth = depth/2



    if nx.is_connected(g):
        return [g]
    else:

        connected_components = list(nx.connected_components(g))
        num_initial_components = len(connected_components)
        i = 0
        while(len(connected_components) > 1):

            # Remove a random component and gets its neighbours
            connected_nodes = component_outer = connected_components.pop()
            expanded_outer = bfs_expand(g_native, connected_nodes, depth=depth)

            # Search if neighbours overlap with other components neighbors
            for component_inner in connected_components:
                if component_inner == component_outer: continue
                expanded_inner = bfs_expand(g_native, component_inner, depth=depth)

                # If the neighbours of components intersect connect them
                neighbours_intersection = expanded_inner.intersection(expanded_outer)
                if len(neighbours_intersection) > 0:
                    connected_nodes = connected_nodes.union(neighbours_intersection)
                    connected_nodes = connected_nodes.union(component_inner)

            # Remove components that got merged
            for component_inner in connected_components:
                if component_inner.issubset(connected_nodes):
                    connected_components.remove(component_inner)

            # Add the new merge connected component
            connected_components.append(connected_nodes)

            # Create exit for inifinite loop (components cannot be connected)
            i += 1
            if i > (num_initial_components*10):
                break

    for component in connected_components:
        h = g_native.subgraph(list(component)).copy()
        dangle_trim(h)
        if len(list(h.nodes)) != 0:
            graphs.append(h)

    return graphs

def connect_all(input_dir, native_dir, output_dir):
    """
    runs connect_components on all graphs in input_dir and outputs
    resulting connected graphs to output_dir
    """
    i = 0
    for graph_file in os.listdir(input_dir):
        # Loop control
        # if i == 30: break
        # i += 1
        if '.nx' not in graph_file: continue

        # read input and compute function
        g = nx.read_gpickle(os.path.join(input_dir, graph_file))
        g_native = nx.read_gpickle(os.path.join(native_dir, graph_file))
        connected_graphs = connect_components(g, g_native)

        # Write output
        pbid = graph_file[:4]
        for i, h in enumerate(connected_graphs):
            nx.write_gpickle(h, os.path.join(output_dir, (pbid + '_' + str(i) + '.nx') ))
            print_component_info(h, i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help= 'Directory containing interface graphs')
    parser.add_argument('-C', '--complement_dir',
                        help= 'Directory containing complement interface graphs')
    parser.add_argument('-N', '--native_dir',
                        help='directory containing native unchopped graphs',
                        default = os.path.join(scriptdir, '..', 'data', 'graphs', 'native'))
    parser.add_argument('output',
                        help='output directory to store cleaned up graphs')
    args = parser.parse_args()

    #connect_all(args.dir, args.native_dir, args.output)

    balance_complement_all(args.dir, args.complement_dir, args.output)

if __name__ == '__main__':
    main()
