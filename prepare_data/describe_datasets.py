import os


script_dir = os.path.dirname(os.path.realpath(__file__))

def main():

    datasets_container = os.path.join(script_dir, '..', 'data', 'Interface_graph_subsets')
    for directory in os.listdir(datasets_container):
        # Count the number of graphs in each set
        interface = os.path.join(datasets_container, directory)
        complement = os.path.join(datasets_container, directory, 'complement')
        for dataset in [interface, complement]:
            num_graphs = len(os.listdir(dataset)) - 1
            print(directory, '\t num graphs: ', num_graphs)

        # Count the number of nodes
        

if __name__ == '__main__':
    main()
