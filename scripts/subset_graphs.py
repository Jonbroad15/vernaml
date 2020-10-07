import networkx as nx
import argparse
import os
import pandas as pd
import csv

def main():
    parser =argparse.ArgumentParser()
    parser.add_argument('-i', help='input directory', default ='../data/unchopped_v4_nr')
    parser.add_argument('--subset', help='CSV file of subset of interface residues')
    parser.add_argument('-o', help='output directory')
    args = parser.parse_args()

    # Initialize subset from subset csv input argument
    subset = {}
    with open(args.subset, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for items in reader:
            if items[1] == 'postion': continue
            pbid = f'{items[0]}.nx'
            if pbid in subset.keys():
                subset[pbid].append((pbid, (items[2], int(items[1]))))
            else:
                subset[pbid] = [(pbid, (items[2], int(items[1])))]

    # slice graphs
    for f in os.listdir(args.i):
        g = nx.read_gpickle(os.path.join(args.i, f))
        pbid = list(g.nodes)[0][0]
        if pbid not in subset.keys(): continue
      #  print(f'number of nodes in {pbid}: {len(list(g.nodes))}')
        g = g.subgraph(subset[pbid]).copy()
      #  print(f'POST SUBSET number of nodes in {pbid}: {len(list(g.nodes))}')
        nx.write_gpickle(g, os.path.join(args.o, f))

if __name__ == '__main__':
    main()
