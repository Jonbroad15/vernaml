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
                        load_from_cache = True):
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

def get_motifs_json(json_dict):
    """
    Create mapping of nodes to motifs from json dict

    :param json_file: path to json file containing motifs
    :return node_to_motifs: dictionary mapping nodes to motifs
    :return motif_set: set of motifs found
    """

    node_to_motifs = defaultdict(set)
    motif_set = set()

    for motif, instances in json_dict.items():
        for instance in instances:
            for node_dict in instance:
                node = node_dict['node']
                node[1] = tuple(node[1])
                node = tuple(node)

                node_to_motifs[node].add(motif)
                motif_set.add(motif)

    return node_to_motifs, motif_set

def build_onehot_nodes(node_labels,interface_dir,
                                 meta_graph_path=None,
                                 json_motifs=None,
                                 load_from_cache=True,
                                 maximal_only=True):
    """
    Extracts one_hots for each node in the set interface_dir

    :param meta_graph_path: path to meta graph
    :param interface_dir: directory of graphs
    :param maximal_only: if True, only keeps largest motif if superset.

    :return X: one hot array of number of nodes in all graphs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    if meta_graph_path:
        node_to_motifs, motif_set = get_motifs_metagraph(meta_graph_path,
                                                        load_from_cache=load_from_cache)
    elif json_motifs:
        node_to_motifs, motif_set = get_motifs_json(json_motifs)
    else:
        raise Exception("Build_onehot_nodes needs a metagraph or json_motifs")

    # print(node_to_motifs)
    # print(motif_set)

    print('Building onehot')
    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(node_labels), len(hot_map)))
    target_labels = []
    # print('node_labels keys:')
    # for key in node_labels.keys():
        # if '1csl.nx' in key:
            # print(key)
    # print('node_to_motifs keys:')
    # for key in node_to_motifs.keys():
        # if '1csl.nx' in key:
            # print(key)

    # print(list(node_labels.keys())[:10])
    # print(list(node_to_motifs.keys())[:10])
    for i, (node, label) in enumerate(node_labels.items()):
        target_labels.append(str(label))
        for motif in node_to_motifs[node]:
            X[i][hot_map[motif]] = 1

        # target_labels.append(label)

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    # print(target_encode)
    y = [target_encode[label] for label in target_labels]

    return X, y

def build_onehot_graphs(meta_graph_path,
                 graph_annotations,
                 interface_dir,
                 maximal_only=True,
                 task='ALL'
                 ):
    """
    Extract onehots for each PDB in the meta_graph

    :param meta_graph_path: path to meta graph
    :param graph_annotations: JSON file containing  { graph : label }
    :param maximal_only: if True, only keeps largest motif if superset.

    :return X: one hot array of number of PDBs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    # map graph files to dict of motif occurrences
    # TODO: change interface graphs to be in the same containing folder 
    #       and use new nomenclature
    graph_to_motifs = defaultdict(Counter)

    with open(graph_annotations, 'r') as labels:
        graph_annot_dict = json.load(labels)

    maga_graph = pickle.load(open(meta_graph_path, 'rb'))

    # print('\nmaga_graph attributes: ', dir(maga_graph))
    meta_nodes = sorted(maga_graph.maga_graph.nodes(data=True),
                        key=lambda x: len(x[0]),
                        reverse=True)
    motif_set = set()
    for motif, d in tqdm(meta_nodes):
        # TODO: performance of this function can be optimized
        # - First turn the frozensets into one big set
        # - then for each node in the graph loop through motifs and check if it is in there
        for i, instance in tqdm(enumerate(d['node_set'])):
            for node_id in instance:
                # print('node_id:', node_id)
                # print('reversed_node_map: ', maga_graph.reversed_node_map[node_id])
                # Node_id are integer ids that map back to nx nodes
                node = maga_graph.reversed_node_map[node_id]
                pbid = (node[0])[:4]
                graphs = os.listdir(interface_dir)
                for graph_file in graphs:
                    if pbid not in graph_file: continue
                    path = os.path.join(interface_dir, graph_file)
                    g = nx.read_gpickle(path)
                    if node in g.nodes:

                    # add motif to the graph
                        if maximal_only:
                            # make sure larger motif not already counted
                            for larger in graph_to_motifs[graph_file].keys():
                                if motif.issubset(larger):
                                    break
                            else:
                                graph_to_motifs[graph_file].update([motif])
                                motif_set.add(motif)
                        else:
                            graph_to_motifs[graph_file].update([motif])
                            motif_set.add(motif)

    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(graph_to_motifs), len(hot_map)))
    graphs = []
    for i, (graph, motif_counts) in enumerate(graph_to_motifs.items()):
        graphs.append(graph)
        for motif, count in motif_counts.items():
            X[i][hot_map[motif]] = count

    # encode prediction targets
    target_labels = []
    for graph in graphs:
        label = graph_annot_dict[task][graph]
        target_labels.append(label)

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    y = [target_encode[label] for label in target_labels]

    return X, y

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

def draw_roc(data, save_fig, task):
    """
    Draw ROC curve for binary classification task
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    plt.clf()
    first = True

    task_to_name = {'rna': 'RNA',
                    'ion': 'Ion',
                    'ligand': 'Small Molecule',
                    'protein': 'Protein'}

    for X, y, label in data:
        print(f"Data shape {X.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=.2,
                                                            random_state=0
                                                            )

        # fit a model
        # rf = RandomForestClassifier(max_features = 5,n_estimators = 500)
        classifier = SGDClassifier()
        model = classifier.fit(X_train, y_train)

        # get prediction probabilities
        r_probs = [0 for _ in range(len(y_test))]
        model_probs = model.decision_function(X_test)
        # model_probs = model_probs[:, 1]

        # Compute AUROC and draw ROC
        r_auc = roc_auc_score(y_test, r_probs)
        model_auc = roc_auc_score(y_test, model_probs)

        # print scores:
        print(f'{label} AUROC = %.3f' % (model_auc))

        # Calculate ROC curve
        r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
        # print(r_probs)
        model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

        # Plot the curve
        if first: plt.plot(r_fpr, r_tpr, linestyle='--',
                    label='Random (AUROC = %0.3f)' % r_auc)
        plt.plot(model_fpr, model_tpr,
                label=f'{label} (AUROC = %0.3f)' % model_auc)


        # Title
        plt.title(f'RNA-{task_to_name[task]}')
        # Axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        first = False
    # Show legend
    plt.legend()
    # Show plot
    plt.savefig(save_fig)


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
                                            'interfaces_pickle4'))
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
    parser.add_argument('-f', '--fig_dir',
                        help='directory to save ROC figures',
                        default = os.path.join(script_dir, '..', 'images', 'rocs'))
    args = parser.parse_args()

    tasks = args.tasks.split()
    accuracy = defaultdict(dict)
    roc_data = []

    for task in tasks:
        print('Computing accuracy for: ', task)
        print('Doing node_level predictions')
        labels = {}
        labels = get_binary_node_labels(os.path.join(args.graph_dir, task), task)

        if args.metagraph:
            name = 'vernal'
            print('Motif Set:', name)
            X, y = build_onehot_nodes(labels, args.graph_dir,
                                        load_from_cache=args.c,
                                        meta_graph_path=args.metagraph)
            # accuracy['vernal'][task] = kfold(X, y)
            roc_data.append((X, y, name))

        if args.json_motifs:
            with open(args.json_motifs, 'r') as f:
                data = json.load(f)
            for name, motif_set in data.items():
                print('Motif Set: ', name)
                X, y = build_onehot_nodes(labels, args.graph_dir,
                                            load_from_cache=args.c,
                                            json_motifs=motif_set)
               # accuracy[name][task] = kfold(X, y)
                roc_data.append((X, y, name))

        draw_roc(roc_data, os.path.join(args.fig_dir, task), task)

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
