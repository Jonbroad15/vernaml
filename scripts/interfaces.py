"""
Module for detecting RNA-protein, RNA-RNA and RNA-smallMolecule interface residues.
Interface residues decided on Euclidean distance cutoff.
"""

import argparse
import os

import numpy as np
from Bio.PDB import *


scriptdir = os.path.dirname(__file__)


def get_interfaces(cif_path, cutoff=10):
    """Obtain only RNA interfacing residues within a single structure of polymers. Uses
   KDTree data structure for vector search, by the biopython NeighborSearch module.

    Args:
        `cif_path (str)`: Path to structure to analyze (MMCif format)
        `cutoff (float, int)`: Number of Angstroms to use as cutoff distance
        for interface calculation.
    Returns:
        `rna_interface_residues`: List of tuples of the pbid, position, chain of RNA-RNA interface residues
        `protein_interface_residues`: List of tuples of the pbid, position, chain of RNA-protein interface residues
        `other_interface_residues`: List of tuples of the pbid, position, chain, interacting molecule of RNA-other interface residues. (small molecules and DNA)
        `Structure`: BioPython Structure object
    """
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path[-8:-4]
    print(f'Loading structure {structure_id}...')
    try:
        structure = parser.get_structure('X', cif_path)
    except FileNotFoundError:
        print(f'Error file {cif_path} not found')
        return None
        # structure = load_structure(pdb_path)

    print(f'Finding RNA interfaces for structure: {structure_id}')
    #3-D KD tree
    atom_list = Selection.unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)
    rna_bases = ['A', 'U', 'C', 'G']
    close_residues = neighbors.search_all(cutoff, level='R')
    rna_interface_residues = []
    protein_interface_residues = []
    other_interface_residues = []
    for r in close_residues:
        res_1 = r[0]
        res_2 = r[1]

        # skip residues pairs that do not contain rna
        if res_1.get_resname() not in rna_bases and res_2.get_resname() not in rna_bases:
            continue
        #if interaction between different chains add to list
        if res_1.get_parent() != res_2.get_parent():
            c1 = res_1.get_parent().id.strip()
            c2 = res_2.get_parent().id.strip()

            # interaction type and append to corresponding dataset
            # RNA-Protein 
            if is_aa(res_1):
                protein_interface_residues.append((structure_id, res_2.id[1], c2))
            elif is_aa(res_2):
                protein_interface_residues.append((structure_id, res_1.id[1], c1))
            # RNA-RNA
            elif res_1.get_resname() in rna_bases and res_2.get_resname() in rna_bases:
                rna_interface_residues.append((structure_id, res_1.id[1], c1))
                rna_interface_residues.append((structure_id, res_2.id[1], c2))
            # RNA-other
            elif res_1.get_resname() in rna_bases:
                other_interface_residues.append((structure_id, res_1.id[1], c1, res_1.id[0]))
            elif res_2.get_resname() in rna_bases:
                other_interface_residues.append((structure_id, res_2.id[1], c2, res_1.id[0]))


    return list(set(rna_interface_residues)), list(set(protein_interface_residues)), list(set(other_interface_residues)), structure

if __name__ == "__main__":

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input directory of CIF files', default='../data/rna_structure_representative_set/')
    parser.add_argument('-o', help='Output directory', default='../data/')
    parser.add_argument( '-cutoff', help='Cutoff (in angstroms) of the distance between interacting residues', default=10)
    args = parser.parse_args()
    output_dir = os.path.join(scriptdir, args.o)

    # initialize empty residue lists
    rna_interface_residues = []
    protein_interface_residues = []
    other_interface_residues = []

    # find interfaces
    for cif_file in os.listdir(args.i):
        rna, protein, others, _ = get_interfaces(f"../data/rna_structure_representative_set/{cif_file}", cutoff = int(args.cutoff))
        rna_interface_residues = rna_interface_residues + rna
        protein_interface_residues = protein_interface_residues + protein
        other_interface_residues = other_interface_residues + others

    # Output files
    protein_out = os.path.join(output_dir, 'RNAProtein_interactions.csv')
    rna_out =  os.path.join(output_dir, 'RNARNA_interactions.csv')
    other_out =  os.path.join(output_dir, 'otherRNA_interactions.csv')

    with open(rna_out, 'w') as f:
        # Header:
        f.write('pbid,postion,chain\n')
        # Data
        for line in rna_interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]}\n')
    with open(protein_out, 'w') as f:
        # Header:
        f.write('pbid,postion,chain\n')
        # Data
        for line in protein_interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]}\n')
    with open(other_out, 'w') as f:
        # Header:
        f.write('pbid,postion,chain,binding_molecule\n')
        # Data
        for line in other_interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]},{line[3]}\n')




