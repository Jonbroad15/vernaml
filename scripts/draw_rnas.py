"""
Draw RNA graphs for all files in a directory
"""


from drawing import *
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input directory')
    parser.add_argument('-o', help='output_directory')
    args = parser.parse_args()

    for graph_file in os.listdir(args.i):
        if '.nx' not in graph_file: continue
        g = nx.read_gpickle(os.path.join(args.i, graph_file))
        if len(list(g.nodes)) < 1: continue
        rna_draw(g, save=f'{os.path.join(args.o, graph_file[:-3])}.pdf')




if __name__ == '__main__':
    main()
