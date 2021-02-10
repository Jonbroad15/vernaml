import argparse
import json
import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm
import csv
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from tools.meta_graph import MGraph, MGraphAll, MGraphNC
from tools.graph_utils import bfs_expand
from prepare_data.slice import get_pbid

def get_binary_node_labels(graph_dir, task):
    """
    Get labels at the node level, return a dictionary

    :param graph_dir: directory of nx graphs
    :return labels: dictionary of labels
    """

    labels = {}
    keyerrors = []

    for graph_file in os.listdir(graph_dir):
        if '.nx' not in graph_file: continue
        # try:
        g = nx.read_gpickle(os.path.join(graph_dir, graph_file))
        # except ValueError:
            # print('could not read graph: ', graph_file)
            # continue
        for node in g.nodes:
            try:
                if g.nodes[node][task]:
                    labels[node] = 1
                else:
                    labels[node] = 0
            except KeyError:
                keyerrors.append(node)

    if len(keyerrors) > 0: print('key errors found in the following nodes:')
    for line in keyerrors:
        print(line)

    return labels

def get_node_labels(graph_dir, task):
    """
    Get labels at the node level, return a dictionary

    :param graph_dir: directory of nx graphs
    :return labels: dictionary of labels
    """

    labels = {}
    keyerrors = []

    for graph_file in os.listdir(graph_dir):
        if '.nx' not in graph_file: continue
        # try:
        g = nx.read_gpickle(os.path.join(graph_dir, graph_file))
        # except ValueError:
            # print('could not read graph: ', graph_file)
            # continue
        for node in g.nodes:
            try:
                labels[node] = g.nodes[node][task]
            except KeyError:
                keyerrors.append(node)

    if len(keyerrors) > 0: print('key errors found in the following nodes:')
    for line in keyerrors:
        print(line)

    return labels


def get_binary_labels(graph_dir):
    """
    For graphs in a directory with a complement subdirectory, get binary interface labels
    """
    labels = {}

    # Interface graphs
    for graph_file in os.listdir(graph_dir):
        if '.nx' not in graph_file: continue
        if 'C' in graph_file[4:]:
            labels[graph_file] = 0
        else:
            labels[graph_file] = 1


    return labels

def get_motifs_metagraph(meta_graph_path,
                        maximal_only=True,
                        load_from_cache = True,
                        motif_size = None):
    """
    Parse metagraph to get a dictionary of
                    keys = nodes
                    values = list of motifs that node belongs to

    Will try to load from cache if it exists

    :param meta_graph_path:
    :param maximal_only: if True only get the largest motif
    :return node_to_motifs: dictionary of node to motif mappings
    :motif_set: set of motifs contained in node_to_motifs
    """
    # Cache node_to_motifs mapping
    onehot_cache = os.path.join(script_dir, '..', 'data', '.onehot_cache')

    if not os.path.exists(onehot_cache):
        os.mkdir(onehot_cache)

    if motif_size:
        node_to_motifs_file = os.path.join(onehot_cache,
                                        'node_to_motifs_size' + str(motif_size) + '.p')
        motif_set_file = os.path.join(onehot_cache,
                                     'motif_set_size' + str(motif_size) + '.p')
    else:
        node_to_motifs_file = os.path.join(onehot_cache, 'node_to_motifs.p')
        motif_set_file = os.path.join(onehot_cache, 'motif_set.p')

    if os.path.exists(node_to_motifs_file)\
    and os.path.exists(motif_set_file) \
    and load_from_cache:
        print('Loading node_to_motifs from cache')
        with open(node_to_motifs_file, 'rb') as f:
            node_to_motifs = pickle.load(f)
        with open(motif_set_file, 'rb') as f:
            motif_set = pickle.load(f)
    else:
        print('Building node_to_motifs from metagraph')
        # map graph files to dict of motif occurrences
        node_to_motifs = defaultdict(set)

        maga_graph = pickle.load(open(meta_graph_path, 'rb'))

        meta_nodes = sorted(maga_graph.maga_graph.nodes(data=True),
                            key=lambda x: len(x[0]),
                            reverse=True)
        motif_set = set()
        for motif, d in tqdm(meta_nodes):
            if motif_size:
                if len(motif) != motif_size: continue
            try:
                tuples = enumerate(d['node_set'])
            except KeyError:
                print('\nKeyError found at:\nmotif:', motif, '\nd:', d)
                continue
            for i, instance in tuples:
                for node_id in instance:

                    # Node_id are integer ids that map back to nx nodes
                    node = maga_graph.reversed_node_map[node_id]

                    # add motif to the node
                    if maximal_only:
                        # make sure larger motif not already counted
                        for larger in node_to_motifs[node]:
                            if motif.issubset(larger):
                                break
                        else:
                            node_to_motifs[node].add(motif)
                            motif_set.add(motif)
                    else:
                        node_to_motifs[node].add(motif)
                        motif_set.add(motif)

        with open(node_to_motifs_file, 'wb') as f:
            pickle.dump(node_to_motifs, f)
        with open(motif_set_file, 'wb') as f:
            pickle.dump(motif_set, f)


    # print('node_to_motifs: ', node_to_motifs.keys())
    # for node, motif in node_to_motifs.items():
        # print('node: ', node)
        # print('motif: ', motif)
    # print(motif_set)
    return node_to_motifs, motif_set


def parse_json_node(node):
    """
    parse a node from a JSON (made of lists) into a normal node (of tuples)
    """

    node[1] = tuple(node[1])
    node = tuple(node)

    return node

def node_to_json(node):
    """
    change the tuples to list for a node to serialise to json
    """
    new_node = [node[0], list(node[1])]

    return new_node

def parse_str_node(node):
    """
    parse a node from a string to a tuple
    """
    # node: (xxxx.nx, (C, N))
    node = node.translate({ord(i): None for i in '()\''})

    node = node.split(',')
    new_node = (node[0], (node[1], int(node[2])))

    return new_node

def get_motifs_json(json_dict, native_dir):
    """
    sCreate mapping of nodes to motifs from json dict

    :param json_file: path to json file containing motifs
    :return node_to_motifs: dictionary mapping nodes to motifs
    :return motif_set: set of motifs found
    """

    graphs = os.listdir(native_dir)

    node_to_motifs = defaultdict(set)
    motif_set = set()

    for motif, instances in json_dict.items():
        for instance in instances:
            for node_dict in instance:
                node = parse_json_node(node_dict['node'])

                graph = node[0]
                if graph not in graphs:
                    continue

                node_to_motifs[node].add(motif)
                motif_set.add(motif)

    return node_to_motifs, motif_set

def build_onehot_nodes(node_labels, native_dir,
                                 meta_graph_path=None,
                                 json_motifs=None,
                                 load_from_cache=True,
                                 motif_size = None,
                                 extend = False,
                                 maximal_only=True):
    """
    Extracts one_hots for each node in the set interface_dir

    :param node_labels: JSON dictionary of target labels
    :param meta_graph_path: path to meta graph
    :param interface_dir: directory of graphs
    :param maximal_only: if True, only keeps largest motif if superset.

    :return X: one hot array of number of nodes in all graphs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    if meta_graph_path:
        node_to_motifs, motif_set = get_motifs_metagraph(meta_graph_path,
                                                        load_from_cache=load_from_cache,
                                                        motif_size = motif_size)
    elif json_motifs:
        node_to_motifs, motif_set = get_motifs_json(json_motifs, native_dir)
    else:
        raise Exception("Build_onehot_nodes needs a metagraph or json_motifs")

    if extend: node_to_motifs = extend_motifs(node_to_motifs, native_dir)

    print('Building onehot')
    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(node_labels), len(hot_map)))
    target_labels = []
    for i, (node, label) in enumerate(node_labels.items()):
        target_labels.append(str(label))
        for motif in node_to_motifs[node]:
            X[i][hot_map[motif]] = 1

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    y = [target_encode[label] for label in target_labels]

    return X, y

def get_neighbours(graph_dir,
                        load_from_cache=True,
                        depth = 1):
    """
    get a list of all neighbours of all nodes, return the result and JSON serialise
    it into a cache

    :param graph_dir: directory of native graphs
    :return neighbours: dictionary list of neighbours for every node
    """


    # caching
    onehot_cache = os.path.join(script_dir, '..', 'data', '.onehot_cache')

    if not os.path.exists(onehot_cache):
        os.mkdir(onehot_cache)

    neighbours_file = os.path.join(onehot_cache,'neighbours_depth'+ str(depth) + '.json')

    if os.path.exists(neighbours_file)\
    and load_from_cache:
        print('Loading neighbours from cache')
        with open(neighbours_file, 'r') as f:
            neighbours = json.load(f)
    else:
        print('Finding neighbours for all nodes')
        neighbours = defaultdict(list)
        for graph_file in tqdm(os.listdir(graph_dir)):
            if '.nx' not in graph_file: continue
            graph_path = os.path.join(graph_dir, graph_file)
            g = nx.read_gpickle(graph_path)
            for node in g.nodes:
                for neighbour in bfs_expand(g, [node], depth = depth):
                    neighbours[str(node)].append(str(neighbour))
        # save to cache
        with open(neighbours_file, 'w') as f:
            json.dump(neighbours, f, indent=3)
    # parse JSON nodes to tuples
    tuple_neighbours = defaultdict(list)
    for node, neighbour_list in neighbours.items():
        for neighbour in neighbour_list:
            tuple_neighbours[parse_str_node(node)].append(parse_str_node(neighbour))

    return tuple_neighbours

def extend_motifs(node_to_motifs, graph_dir):
    """
    extend motif mapping by collectin motifs from all nodes connected to the given node
    """

    nodes = list(node_to_motifs.keys())

    neighbours = get_neighbours(graph_dir)

    print('extending motifs')
    for node in tqdm(nodes):
        for neighbour in neighbours[node]:
            node_to_motifs[node] = node_to_motifs[neighbour] | node_to_motifs[node]

    return node_to_motifs


def kfold(X, y):

    from sklearn.model_selection import cross_val_score

    print('Data shape: ', X.shape)

    scores = cross_val_score(SGDClassifier(), X, y)
    dummy_scores = cross_val_score(DummyClassifier(strategy='stratified'),
                                    X, y)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Dummy: %0.2f (+/- %0.2f)" % (dummy_scores.mean(), dummy_scores.std() * 2))

    return (scores.mean(), dummy_scores.mean())

def draw_roc(roc_data, save_fig):
    """
    Draw ROC curve for binary classification task
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    plt.clf()
    plt.figure(figsize=(15,10))
    task_to_name = {'rna': 'RNA',
                    'ion': 'Ion',
                    'ligand': 'Small Molecule',
                    'protein': 'Protein'}

    weights = defaultdict(dict)
    for i, (task, data) in enumerate(roc_data.items()):
        rcurve_not_made = True
        plt.subplot(2, 2, i+1)
        for X, y, label in data:
            print(f"{task}, {label}: shape = {X.shape}")

            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                test_size=.2,
                                                                random_state=0)

            # Plot random curve
            if rcurve_not_made:
                r_probs = [0 for _ in range(len(y_test))]
                r_auc = roc_auc_score(y_test, r_probs)
                r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
                plt.plot(r_fpr, r_tpr, linestyle='--',
                        label='Random (AUROC = %0.3f)' % r_auc)
                rcurve_not_made = False
            # fit a model
            classifier = SGDClassifier()
            model = classifier.fit(X_train, y_train)

            # get prediction probabilities
            model_probs = model.decision_function(X_test)

            # Compute AUROC and draw ROC
            model_auc = roc_auc_score(y_test, model_probs)

            # print scores:
            print(f'{task}, {label}: AUROC = %.3f' % (model_auc))
            # save coefficients
            weights[label][task] = model.coef_

            # Calculate ROC curve
            model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

            # Plot ROC curve
            plt.plot(model_fpr, model_tpr,
                       label=f'{label} (AUROC = %0.3f)' % model_auc)


            # Title
            plt.title(f'RNA-{task_to_name[task]}')
            if i == 2:
                # Axis labels
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')

        # Show legend
        plt.legend(loc='lower right')
    # Save plot
    plt.tight_layout()
    plt.savefig(save_fig)

    return weights

def plot_weights(weights, save):
    """
    plot a pcolormesh graph
    """


    tasks = list(weights.keys())

    for task in tasks:
        top_weights = sorted([i for i in weights[task][0]], reverse=True)[:50]
        print(task, top_weights)

    x = [i for i in range(len(weights[tasks[0]][0]))]
    y = [i for i in range(len(tasks))]
    print(list(enumerate(tasks)))

    Z = np.array([w[0] for w in weights.values()])

    plt.clf()
    print(Z.shape)
    plt.pcolormesh(x, y, Z, shading='auto',
                    cmap = 'GnBu',
                    label='motif importance')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save)

def compute_accuracy(X, y):
    """
    :param task, which task to predict on.
    """

    print(f"Data shape {X.shape}.")

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=.2,
                                                        random_state=0
                                                        )

    dummy = DummyClassifier(strategy='stratified').fit(X_train, y_train)
    dummy_acc = dummy.score(X_test, y_test)

    model = SGDClassifier().fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    print("accuracy ", acc)
    print("dummy ", dummy_acc)

    return (round(acc, 3), round(dummy_acc, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--graph_dir',
                        help='directory containing graphs to predict on',
                        default=os.path.join(script_dir, '..', 'data', 'graphs',
                                            'interfaces_cutoff10'))
    parser.add_argument('-N', '--native_dir',
                        help='directory containing native graphs',
                        default = os.path.join(script_dir, '..', 'data', 'graphs', 'native'))
    parser.add_argument('-onehot_data_output',
                        help='JSON output for onehot data',
                        default=os.path.join(script_dir, '..', 'data', 'onehot_data.json'))
    parser.add_argument('-m', '--metagraph',
                        help = 'Metagraph of Motifs from vernal')
                        #default = os.path.join(script_dir, '..', 'data', 'general_fuzzy.p'))
    parser.add_argument('-j', '--json_motifs',
                        help='motifs given in a json serialization instead of metagraph')
    parser.add_argument('-n', action='store_false',
                        help='option to build onehots at the node level',
                        default=True)
    parser.add_argument('-c', action='store_false',
                        help='clear cache', default=True)
    parser.add_argument('-o', '--accuracy_output',
                        help='output accuracy table to csv file')
    parser.add_argument('-t', '--tasks', default = 'rna ligand protein ion')
    parser.add_argument('-f', '--fig_save',
                        help='output file for roc plots',
                        default = os.path.join(script_dir, '..', 'images', 'rocs'))
    parser.add_argument('-e', '--extend_motifs',
                        help='flag to extend motifs with node hops',
                        action='store_true')
    parser.add_argument('-s', '--motif_size',
                        help = 'Fix motifs to only use those of fixed size for vernal motifs',
                        type = int)
    parser.add_argument('-w', '--plot_weights',
                        help = 'save file for plot of feature importance')
    args = parser.parse_args()

    tasks = args.tasks.split()
    accuracy = defaultdict(dict)
    roc_data = defaultdict(list)

    for task in tasks:
        print('Computing accuracy for: ', task)
        print('Doing node_level predictions')
        labels = {}
        labels = get_binary_node_labels(os.path.join(args.graph_dir, task), task)

        if args.metagraph:
            name = 'vernal'
            print('Motif Set:', name)
            X, y = build_onehot_nodes(labels, args.native_dir,
                                        load_from_cache=args.c,
                                        meta_graph_path=args.metagraph,
                                        extend = args.extend_motifs)
            # accuracy['vernal'][task] = kfold(X, y)
            roc_data[task].append((X, y, name))

            if args.motif_size:
                name = 'vernal-' + str(args.motif_size) + 'mers'
                print('Motif Set:', name)
                X, y = build_onehot_nodes(labels, args.native_dir,
                                            load_from_cache=args.c,
                                            meta_graph_path=args.metagraph,
                                            motif_size = args.motif_size,
                                            extend = args.extend_motifs)
                # accuracy['vernal'][task] = kfold(X, y)
                roc_data[task].append((X, y, name))

            args.c = True

        if args.json_motifs:
            with open(args.json_motifs, 'r') as f:
                data = json.load(f)
            for name, motif_set in data.items():
                print('Motif Set: ', name)
                X, y = build_onehot_nodes(labels, args.native_dir,
                                            load_from_cache=args.c,
                                            json_motifs=motif_set,
                                            motif_size = args.motif_size,
                                            extend = args.extend_motifs)
               # accuracy[name][task] = kfold(X, y)
                roc_data[task].append((X, y, name))

    weights = draw_roc(roc_data, args.fig_save)

    if args.plot_weights:
        plot_weights(weights['vernal'], args.plot_weights)


    # print(accuracy)
    if args.accuracy_output:
        with open(args.accuracy_output, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            motif_sets = list(accuracy.keys())
            writer.writerow(['task'] + motif_sets)
            for task in tasks:
                row = [accuracy[name][task] for name in motif_sets]
                # dummy_row = [accuracy[name][task][1] for name in motif_sets]
                writer.writerow([task] + row)
                # writer.writerow([task + '_dummy'] + dummy_row)


if __name__ == '__main__':
    main()
