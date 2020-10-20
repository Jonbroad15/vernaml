"""
Module for detecting RNA-protein, RNA-RNA and RNA-smallMolecule interface residues.
Interface residues decided on Euclidean distance cutoff.
"""

import sys
import argparse
import os
import numpy as np
from Bio.PDB import *


scriptdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(scriptdir, '..'))

from prepare_data.interfaces import *

def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Input directory of CIF files for RNA structures',
                        default= os.path.join(scriptdir,'..','data','structures'))
    parser.add_argument('-o', '--output',
                        help='Output csv file of interacting residue list',
                        default = os.path.join(scriptdir, '..', 'data',
                            'interface_residue_list.csv'))
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
        break

    with open(output_file, 'w') as f:
        # Header:
        f.write('pbid,position,chain,type\n')
        # Data
        for line in interface_residues:
            f.write(f'{line[0]},{line[1]},{line[2]},{line[3]}\n')

    print("DONE")

    if len(files_not_found) > 0:
        print('\n\n', 'NOTE: \t The following files were not found:\n')
        for line in files_not_found:
            print(line)

if __name__ == "__main__":
    main()



