import os
import sys
import pickle

from tqdm import tqdm
import networkx as nx
import dgl
import numpy as np
import torch


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from torch.utils.data import Dataset, DataLoader, Subset

EDGE_MAP = {'B53': 0, 'CHH': 1, 'CHS': 2, 'CHW': 3, 'CSH': 2, 'CSS': 4, 'CSW': 5, 'CWH': 3, 'CWS': 5, 'CWW': 6, 'THH': 7, 'THS': 8, 'THW': 9, 'TSH': 8, 'TSS': 10, 'TSW': 11, 'TWH': 9, 'TWS': 11, 'TWW': 12}

def get_labels(g, mode=True):

    one_count = 0
    zero_count = 0
    labels = {}
    for node in g.nodes:
        for interaction in ['rna', 'protein', 'ion', 'ligand']:
            if g.nodes[node][interaction] is not None:
                # Interface
                labels[node] = 1
                one_count += 1
                break
        else:
            zero_count += 1
            labels[node] = 0

    if mode:
        if one_count >= zero_count:
            return {key: 1 for key in labels.keys()}
        else:
            return {key: 0 for key in labels.keys()}
    else:
        return labels


class V1(Dataset):
    def __init__(self,
                 edge_map,
                 graphs_path='../data/graphs/interfaces_cutoff10/',
                 debug=False,
                 shuffled=False,
                 ):

        self.path = graphs_path
        self.all_graphs = sorted(os.listdir(graphs_path))


        self.edge_map = edge_map
        # This is len() so we have to add the +1
        self.num_edge_types = max(self.edge_map.values()) + 1
        print(f"Found {self.num_edge_types} relations")

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        g_path = os.path.join(self.path, self.all_graphs[idx])
        try:
            graph = nx.read_gpickle(g_path)
        except:
            print("ERROR could not read graph file:\n", g_path)

        graph = nx.to_undirected(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}
        interface = get_labels(graph)
        nx.set_node_attributes(graph, name='interface', values = interface)
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['interface'])


        return g_dgl

def collate_fn(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    return dgl.batch(samples)

class Loader():
    def __init__(self,
                 graphs_path='data/graphs/interfaces_cutoff10',
                 batch_size=5,
                 num_workers=20,
                 debug=False,
                 shuffled=False,
                 edge_map=EDGE_MAP):
        """

        :param graphs_path:
        :param batch_size:
        :param num_workers:
        :param debug:
        :param shuffled:
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = V1(graphs_path=graphs_path,
                          debug=debug,
                          shuffled=shuffled,
                          edge_map=edge_map)

        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self):
        n = len(self.dataset)
        indices = list(range(n))
        # np.random.shuffle(indices)

        np.random.seed(0)
        split_train, split_valid = 0.7, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        all_set = Subset(self.dataset, indices)

        print(f"training items: ", len(train_set))
        sample = next(iter(train_set))
        print(sample)
        print(type(sample))
        sample = next(iter(train_set))
        print(sample)
        print(type(sample))
        sample = next(iter(train_set))
        print(sample)
        print(type(sample))


        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_fn)
        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_fn)
        all_loader = DataLoader(dataset=all_set, shuffle=True, batch_size=self.batch_size,
                                num_workers=self.num_workers, collate_fn=collate_fn)

        batch = next(iter(train_loader))
        print(f"batch len: ", len(batch))
        print(batch)
        print(type(batch))
        batch = next(iter(train_loader))
        print(f"batch len: ", len(batch))
        print(batch)
        print(type(batch))
        batch = next(iter(train_loader))
        print(f"batch len: ", len(batch))
        print(batch)
        print(type(batch))
        return train_loader, test_loader, all_loader

def loader_from_hparams(graphs_path, hparams):
    """
        :params
    """
    loader = Loader(graphs_path=graphs_path,
                    batch_size=hparams.get('argparse', 'batch_size'),
                    num_workers=hparams.get('argparse', 'workers'))
    return loader

