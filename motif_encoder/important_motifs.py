import os
import networkx as nx
import sys
import argparse
import pickle

script_dir = os.path.realpath(os.path.dirname(__file__))
graph_dir = os.path.join(script_dir, '..', 'data', 'graphs')
sys.path.append(os.path.join(script_dir, '..'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('top_motifs', help='dictionary of motifs and their occurences')
    parser.add_argument('-O', '--output_dir', help='output dir for motif graphs',
                            default=os.path.join(graph_dir, 'motifs', 'protein'))

    args = parser.parse_args()

    with open(args.top_motifs, 'rb') as f:
        top_motifs = pickle.load(f)

    for i, (motif, occurences) in enumerate(top_motifs.items()):
        motif_dir = os.path.join(args.output_dir, str(i))
        os.mkdir(motif_dir)
        for graph, nodes in occurences.items():
            g = nx.read_gpickle(os.path.join(graph_dir, 'native', graph))
            motif_graph = g.subgraph(nodes).copy()
            nx.write_gpickle(motif_graph, os.path.join(motif_dir, graph))


if __name__ == '__main__':
    main()
