"""
Draw RNA graphs for all files in a directory
"""


from drawing import *
import argparse
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help='input directory')
    parser.add_argument('o', help='output_directory')
    args = parser.parse_args()
    output_dir = args.o

    #TODO: fix FileNotFoundError
    print(output_dir)
    for graph_file in os.listdir(args.i):
        if '.nx' not in graph_file: continue
        g = nx.read_gpickle(os.path.join(args.i, graph_file))
        if len(list(g.nodes)) < 1: continue
        savefile = os.path.join(output_dir, graph_file[:-3]) + '.pdf'
        rna_draw(g, save=savefile)
        continue




if __name__ == '__main__':
    main()
