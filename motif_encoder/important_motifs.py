import os
import networkx as nx
import sys
import argparse
import pickle
from shutil import copyfile

script_dir = os.path.realpath(os.path.dirname(__file__))
graph_dir = os.path.join(script_dir, '..', 'data', 'graphs')
sys.path.append(os.path.join(script_dir, '..'))

def subset_graphs(pbid_list, output_dir):
    """
    create a new dataset of a subset of pbids from the PDB website
    """
    with open(pbid_list, 'r') as f:
        pbids = f.readline().strip().lower()
        pbids = pbids.split(',')
        pbids = sorted(pbids)

    print(pbids)

    all_dir = os.path.join(graph_dir, 'interfaces_cutoff10', 'protein')
    for graph in os.listdir(all_dir):
        if any([pbid in graph for pbid in pbids]):
            copyfile(os.path.join(all_dir, graph), os.path.join(output_dir, graph))


def slice_motif_graphs(top_motifs_file, output_dir):
    """
    slice graphs from motif dict
    """

    with open(top_motifs_file, 'rb') as f:
        top_motifs = pickle.load(f)

    for i, (motif, occurences) in enumerate(top_motifs.items()):
        motif_dir = os.path.join(output_dir, str(i))
        os.mkdir(motif_dir)
        for graph, nodes in occurences.items():
            g = nx.read_gpickle(os.path.join(graph_dir, 'native', graph))
            motif_graph = g.subgraph(nodes).copy()
            nx.write_gpickle(motif_graph, os.path.join(motif_dir, graph))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-top_motifs', help='dictionary of motifs and their occurences')
    parser.add_argument('-O', '--output_dir', help='output dir for motif graphs',
                            default=os.path.join(graph_dir, 'motifs', 'ligand'))

    args = parser.parse_args()

    slice_motif_graphs('data/top_ten_motifs_vernal_transcription.p',
                        'data/graphs/motifs/transcription')

    # subset_graphs(os.path.join(script_dir, '..', 'data', 'transcription_pdbs.txt'),
                    # os.path.join(graph_dir, 'interfaces_cutoff10', 'transcription'))


if __name__ == '__main__':
    main()
