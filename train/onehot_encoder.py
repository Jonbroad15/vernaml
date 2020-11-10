import argparse
import json
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

def write_onehot_data(graph_dir, output):
    """
    Write onehot_data.json. Contains labels for each graph
    """

    return

def build_onehot(meta_graph_path,
                 pdb_annotations,
                 maximal_only=True,
                 task='rfam'
                 ):
    """
    Extract onehots for each PDB in the meta_graph

    :param meta_graph_path: path to meta graph
    :param pdb_annotations: JSON file containing  { graph : label }
    :param maximal_only: if True, only keeps largest motif if superset.

    :return X: one hot array of number of PDBs by number of motifs.
    """

    from sklearn.preprocessing import OneHotEncoder

    # map pdbs to dict of motif occurrences
    pdb_to_motifs = defaultdict(Counter)

    with open(pdb_annotations, 'r') as labels:
        pdb_annot_dict = json.load(labels)

    maga_graph = pickle.load(open(meta_graph_path, 'rb'))

    meta_nodes = sorted(maga_graph.nodes(data=True),
                        key=lambda x: len(x[0]),
                        reverse=True)
    motif_set = set()

    for motif, d in meta_nodes:
        # motif_id = "-".join(map(str, list(motif)))
        for i, instance in enumerate(d['node_set']):
            node = list(instance).pop()
            pdbid = node[0].split("_")[0]

            # skip if we don't have annotation for this pdb
            if pdbid not in pdb_annot_dict[task]:
                print("Missing")
                continue

            if maximal_only:
                # make sure larger motif not already counted
                for larger in pdb_to_motifs[pdbid].keys():
                    if motif.issubset(larger):
                        break
                else:
                    pdb_to_motifs[pdbid].update([motif])
                    motif_set.add(motif)
            else:
                pdb_to_motifs[pdbid].update([motif])
                motif_set.add(motif)

    # get one hot
    hot_map = {motif: i for i, motif in enumerate(sorted(motif_set))}
    X = np.zeros((len(pdb_to_motifs), len(hot_map)))
    pdbs = []
    for i, (pdb, motif_counts) in enumerate(pdb_to_motifs.items()):
        pdbs.append(pdb)
        for motif, count in motif_counts.items():
            X[i][hot_map[motif]] = count

    # encode prediction targets
    target_labels = []
    for pdb in pdbs:
        label = pdb_annot_dict[task][pdb]
        target_labels.append(label)

    target_encode = {label: i for i, label in
                     enumerate(sorted(list(set(target_labels))))}

    y = [target_encode[label] for label in target_labels]

    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-G', '--graph_dir',
                        help='directory containing graphs to predict on')
    parser.add_argument('-onehot-data-output',
                        help='JSON output for onehot data',
                        default=os.path.join(script_dir, '..', 'data', 'onehot_data.json'))
    args = parser.parse_args()

    write_onehot_data(args.graph_dir, args.onehot-data-output)



if __name__ == '__main__':
    main()
