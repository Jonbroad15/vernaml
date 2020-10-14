import networkx as nx
import argparse
import os
import pandas as pd
import csv
from subset_graphs import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input directory', default ='../data/unchopped_v4_nr')
    parser.add_argument('subset', help='CSV file of subset of interface residues')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('-t', help='interaction type = {protein, ion, rna, ligand, all}', default='all')
    args = parser.parse_args()

    input_dir = args.i
    output_dir = args.output_dir
    try:
        output_prime_dir = os.mkdir(f'{output_dir}complement')
    except FileExistsError:
        output_prime_dir = f'{output_dir}complement'
    interaction_type = args.t

    # Initialize subset from subset csv input argument
    subset = {}
    with open(args.subset, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for items in reader:
            if items[3] != interaction_type and interaction_type != 'all': continue
            if items[1] == 'position': continue
            pbid = f'{items[0]}.nx'
            if pbid in subset.keys():
                subset[pbid].append((pbid, (items[2], int(items[1]))))
            else:
                subset[pbid] = [(pbid, (items[2], int(items[1])))]

    # slice graphs
    for f in os.listdir(input_dir):
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

if __name__ == '__main__':
    main()
