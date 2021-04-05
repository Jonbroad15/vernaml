import os
import sys
import pickle
import json
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
        #,'TW.': 13, 'T.W':14, 'TS.':15, 'T.S':16, 'T.H': 17, 'TH.': 18, 'C.W':19, 'CW.': 20, 'CH.': 21, 'C.H': 22, 'CS.': 23, 'C.S': 23}


def read_graph(g_path):
    with open(g_path, 'r') as f:
        d = json.load(f)
    return nx.readwrite.json_graph.node_link_graph(d)

def get_labels(g, interaction, mode=False):

    one_count = 0
    zero_count = 0
    labels = {}
    for node in g.nodes:
        try:
            if interaction == 'any':
                for interaction in ['protein', 'ion', 'small-molecule']:
                    if g.nodes[node]['binding_' + interaction] is not None:
                        # Interface
                        labels[node] = 1
                        one_count += 1
                        break
                else:
                    zero_count += 1
                    labels[node]= 0
            else:
                if g.nodes[node]['binding_' + interaction] is not None:
                    # Interface
                    labels[node] = 1
                    one_count += 1
                else:
                    zero_count += 1
                    labels[node]= 0

        except KeyError:
            print("ERROR interaction not found for", node)
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
                 interaction = 'any',
                 debug=False,
                 shuffled=False,
                 use_mode=False
                 ):

        self.path = graphs_path
        self.all_graphs = sorted(os.listdir(graphs_path))
        self.use_mode = use_mode
        self.interaction = interaction

        self.edge_map = edge_map
        # This is len() so we have to add the +1
        self.num_edge_types = max(self.edge_map.values()) + 1
        print(f"Found {self.num_edge_types} relations")

    def __len__(self):
        return len(self.all_graphs)

    def __getitem__(self, idx):
        g_path = os.path.join(self.path, self.all_graphs[idx])
        try:
            graph = read_graph(g_path)
        except Exception as e:
            print(e)
            print("ERROR could not read graph file:\n", g_path)

        # graph = nx.to_undirected(graph)
        # graph = nx.Graph(graph)
        one_hot = {}
        for edge, label in (nx.get_edge_attributes(graph, 'LW')).items():
            if '.' in label:
                graph.remove_edge(edge[0], edge[1])
                continue
            try:
                one_hot[edge] = torch.tensor(self.edge_map[label.upper()])
            except KeyError as e:
                # print('ERROR: unrecognized edge label:')
                # print(e)
                graph.remove_edge(edge[0], edge[1])

        interface = get_labels(graph, interaction=self.interaction,
                mode=self.use_mode)
        nx.set_node_attributes(graph, name='interface', values = interface)
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['interface'])


        return g_dgl, [idx]

def collate_fn(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, idx = map(list, zip(*samples))
    batch = dgl.batch(graphs)
    idx = np.array(idx)

    return batch, torch.from_numpy(idx)

class Loader():
    def __init__(self,
                 graphs_path='data/graphs/interfaces_cutoff10',
                 batch_size=5,
                 num_workers=20,
                 interaction = 'any',
                 debug=False,
                 shuffled=False,
                 use_mode=False,
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
                          interaction = interaction,
                          shuffled=shuffled,
                          use_mode=use_mode,
                          edge_map=edge_map)

        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self):
        n = len(self.dataset)
        indices = list(range(n))
        # np.random.shuffle(indices)

        np.random.seed(0)
        split_train, split_valid = 0.5, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        all_set = Subset(self.dataset, indices)



        train_loader = DataLoader(dataset=train_set, shuffle=True,
                                    batch_size=self.batch_size,
                                    num_workers=self.num_workers,
                                    collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=valid_set, shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_set, shuffle=True,
                batch_size=self.batch_size,
                num_workers=self.num_workers, collate_fn=collate_fn)
        all_loader = DataLoader(dataset=all_set, shuffle=True,
                batch_size=self.batch_size, num_workers=self.num_workers,
                collate_fn=collate_fn)
        # i=0
        # num_batches = len(train_loader)
        # t = iter(train_loader)
        # for batch in train_loader:
            # # print(batch)
            # i+=1
        # for j in range(num_batches):
            # print(j)
            # try:
                # print(next(t))
            # except StopIteration:
                # t = iter(train_loader)
                # print(next(t))
            # # if j == 25:
                # # break


        # print('num_batches', num_batches)
        # print('i', i)

        # raise Exception

        return train_loader, valid_loader, all_loader

def loader_from_hparams(graphs_path, hparams):
    """
        :params
    """
    loader = Loader(graphs_path=graphs_path,
                    batch_size=hparams.get('argparse', 'batch_size'),
                    num_workers=hparams.get('argparse', 'workers'),
                    interaction = hparams.get('argparse', 'interaction'),
                    use_mode=hparams.get('argparse', 'use_mode'))
    return loader

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

def main():

    for g_path in listdir_fullpath('../data/practice_graphs/'):
        g = read_graph(g_path)

        for node, data in g.nodes.data():
            if data['binding_protein'] is not None:
                print(node)

        labels = get_labels(g, 'protein')
        print(labels)

if __name__ == '__main__':
    main()

