"""
Module for detecting RNA-protein, RNA-RNA and RNA-smallMolecule interface residues.
Interface residues decided on Euclidean distance cutoff.
"""

import argparse
import os

import numpy as np
from Bio.PDB import *


scriptdir = os.path.dirname(os.path.realpath(__file__))


def is_dna(res):
    """
    returns true if the input residue is a DNA molecule
    """
    if res.id[0] != ' ':
        return False
    if is_aa(res):
        return False
    if 'D' in res.get_resname():
        return True
    else:
        return False


def get_interfaces(cif_path, ligands, cutoff=10, skipWater=True):
    """Obtain RNA interface residues within a single structure of polymers. Uses
   KDTree data structure for vector search, by the biopython NeighborSearch module.

    Args:
        `cif_path (str)`: Path to structure to analyze (MMCif format)
        `cutoff (float, int)`: Number of Angstroms to use as cutoff distance
        for interface calculation.
    Returns:
        `interface_residues`: List of tuples of the pbid, position, chain of RNA-RNA, interaction type
        `Structure`: BioPython Structure object
    """
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path[-8:-4]
    print(f'Loading structure {structure_id}...')
    try:
        structure = parser.get_structure('X', cif_path)
    except FileNotFoundError:
        print(f'ERROR: file {cif_path} not found')
        return None

    print(f'Finding RNA interfaces for structure: {structure_id}')
    #3-D KD tree
    atom_list = Selection.unfold_entities(structure, 'A')
    neighbors = NeighborSearch(atom_list)
    close_residues = neighbors.search_all(cutoff, level='R')
    interface_residues = []
    for r in close_residues:
        res_1 = r[0]
        res_2 = r[1]

       # skip interactions with water
        if skipWater:
            if res_1.id[0] == 'W' or res_2.id[0] == 'W': continue

        # skip protein-protein pairs
        if is_aa(res_1) and is_aa(res_2):
            continue

        # skip interactions with DNA
        if is_dna(res_1) or is_dna(res_2):
            continue

        #if interaction between different chains add to list
        if res_1.get_parent() != res_2.get_parent():

            # get chain names and res names
            c1 = res_1.get_parent().id.strip()
            c2 = res_2.get_parent().id.strip()
            r1 = res_1.get_resname().strip()
            r2 = res_2.get_resname().strip()


            # Determine interaction type and append to corresponding dataset
            # RNA-Protein 
            if is_aa(res_1):
                interface_residues.append((structure_id, res_2.id[1], c2, 'protein'))
            elif is_aa(res_2):
                interface_residues.append((structure_id, res_1.id[1], c1, 'protein'))
            # RNA-RNA or RNA-DNA
            elif res_1.id[0] == ' ' and res_2.id[0] == ' ':
                interface_residues.append((structure_id, res_1.id[1], c1, 'rna'))
                interface_residues.append((structure_id, res_2.id[1], c2, 'rna'))
            # RNA-smallMolecule
            elif  r1 in ligands:
                interface_residues.append((structure_id, res_1.id[1], c1, 'ligand'))
            elif  r2 in ligands:
                interface_residues.append((structure_id, res_2.id[1], c2, 'ligand'))
            # RNA-Ion
            elif 'H' in res_1.id[0] and ' ' in res_2.id[0]:
                interface_residues.append((structure_id, res_2.id[1], c2, 'ion'))
            elif 'H' in res_2.id[0] and ' ' in res_1.id[0]:
                interface_residues.append((structure_id, res_1.id[1], c2, 'ion'))
            elif  'H' in res_2.id[0] and 'H' in res_1.id[0]:
                continue
            else:
                print('warning unmatched residue pair \t res_1.id:', res_1.id, 'res_2.id:', res_2.id)

    # remove duplicates and sort by seqid
    interface_residues = list(set(interface_residues))
    interface_residues_sorted = sorted(interface_residues, key=lambda tup: tup[2])
    interface_residues_sorted = sorted(interface_residues, key=lambda tup: tup[1])

    return interface_residues_sorted, structure



def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Input directory of CIF files for RNA structures',
                        default= os.path.join(scriptdir,'..','data','rna_representative_set/'))
    parser.add_argument('-o', '--output',
                        help='Output csv file of interacting residue list',
                        default = os.path.join(scriptdir, '..', 'data',
                            'interface_residue list.csv'))
    parser.add_argument( '-c', '--cutoff',
                        help='Cutoff (in angstroms) of the distance \
                                between interacting residues',
                                default=10)
    parser.add_argument('-l', '--ligands',
                        help='list of ligands',
                        default= os.path.join(scriptdir, '..', 'data', 'ligand_list.txt'))
    args = parser.parse_args()
    input_dir = args.input
    output_file = args.output
    cutoff = int(args.cutoff)
    ligands_file = args.ligands

    # Get ligands list
    ligands = []
    with open(ligands_file, 'r') as f:
        for line in f.readlines():
            ligands.append(line.strip())

    # initialize empty residue lists
    interface_residues = []
    files_not_found = []

    # find interfaces
    for cif_file in os.listdir(input_dir):
        path = os.path.join(input_dir, cif_file)
        try:
            residues, _ = get_interfaces(path, ligands = ligands, cutoff = cutoff)
        except TypeError:
            files_not_found.append(path)
        interface_residues = interface_residues + residues

    with open(output_file, 'w') as f:
        # Header:
        f.write('pbid,position,chain,type\n')
        # Data
        for line in interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]},{line[3]}\n')

    print("DONE", '\n\n', 'NOTE: \t The following files were not found:\n')
    for line in files_not_found:
        print(line)

if __name__ == "__main__":
    main()



