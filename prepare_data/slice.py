import networkx as nx
import argparse
import os
import pandas as pd
import csv

def slice_graph(g, subset):
    """
    :param g:       graph
    :param subset:  set of nodes

    :return g_subgraph: subgraph of nodes in subset
    :return g_prime:    complement of subgraph
    """
    g_prime = g.copy()

    if len(subset) > 0:
        g_subgraph = g_prime.subgraph(subset).copy()
        g_prime.remove_nodes_from([n for n in g_prime if n in set(subset)])
    else:
        g_subgraph = None

    return g_subgraph, g_prime

def slice_all(input_dir, output_dir, subset):
    """
    Slice all graphs in input_dir and writes them in the output_dir

    :param input_dir:
    :param output_dir:
    :param subset: dictionary of pbid : [list of interfaces nodes]

    :return:
    """
    try:
        os.mkdir(os.path.join(output_dir, 'complement'))
    except FileExistsError:
        print('complement directory already exists! make sure you are not overwriting')
    output_prime_dir = os.path.join(output_dir, 'complement')

    for f in os.listdir(input_dir):
        print(f'Slicing RNA graph: \t {f}')
        graph_file = os.path.join(input_dir, f)
        g = nx.read_gpickle(os.path.join(input_dir, f))
        pbid = list(g.nodes)[0][0]
        if pbid in subset.keys():
            sub = subset[pbid]
            g_subgraph, g_prime = slice_graph(g, sub)
            nx.write_gpickle(g, os.path.join(output_dir, f))
        else:
            g_prime = g.copy()

        nx.write_gpickle(g_prime, os.path.join(output_prime_dir, f))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input directory', default ='../data/unchopped_v4_nr')
    parser.add_argument('subset', help='CSV file of subset of interface residues')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('-t','--type',
                        help='interaction type = {protein, ion, rna, ligand, all}',
                        default='all')
    args = parser.parse_args()

    input_dir = args.i
    output_dir = args.output_dir
    interaction_type = args.type.split(' ')

    return input_dir, output_dir, interaction_type

def parse_subset(interface_residues, interaction_type):
    """
    Initialize subset from list of tuples of interface residues
    :param interface residues: list of tuples (pbid, position, chain, type)
    :param interaction_type: list of strings of interaction type
                            options = {protein, rna, ligand, ion, all}
    :return subset: dictionary of pbid : [list of nodes]
    """
    subset = {}
    print('Building interface subset dictionary...')
    for pbid, position, chain, curr_type in interface_residues:
        if curr_type not in interaction_type and 'all' not in interaction_type: continue
        if pbid in subset.keys():
            subset[pbid].append((pbid, (chain, int(position))))
        else:
            subset[pbid] = [(pbid, (chain, int(position)))]

    return subset

def parse_subset_fromcsv(subset_file, interaction_type):
    """
    Initialize subset from subset csv input argument
    :param subset_file:
    :param interaction_type: list of strings of interaction type
                            options = {protein, rna, ligand, ion, all}
    :return subset: dictionary of pbid : [list of nodes]
    """
    subset = {}
    print('Building interface subset dictionary...')
    with open(subset_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = True
        for items in reader:
            if header:
                header = False
                continue
            if items[3] not in interaction_type and 'all' not in interaction_type: continue
            pbid = f'{items[0]}.nx'
            if pbid in subset.keys():
                subset[pbid].append((pbid, (items[2], int(items[1]))))
            else:
                subset[pbid] = [(pbid, (items[2], int(items[1])))]

    return subset


