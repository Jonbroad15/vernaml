import unittest
from Bio.PDB import *
from ..is_dna import *
import os

class is_dnaTestCase(unittest.TestCase):

    def test_is_dna(self):

        parser = MMCIFParser(QUIET=True)

        dna_files = os.listdir('../data/test_structures/dna')
        residues = []
        for cif in dna_files:
            structure = parser.get_structure(cif, f'../data/test_structures/dna/{cif}')
            residues = residues + list(structure.get_residues())

        # remove ions/water/ligands
        dna_residues = []
        for res in residues:
            if res.id[0] == ' ':
                dna_residues.append(res)

        # test function
        filter_residues = []
        for res in dna_residues:
            if is_dna(res):
                filter_residues.append(res)

        self.assertCountEqual(dna_residues, filter_residues)

    def test_is_dna_proteins(self):

        parser = MMCIFParser(QUIET=True)

        dna_files = os.listdir('../data/test_structures/protein')
        residues = []
        for cif in dna_files:
            structure = parser.get_structure(cif, f'../data/test_structures/protein/{cif}')
            residues = residues + list(structure.get_residues())

        # remove ions/water/ligands
        dna_residues = []
        for res in residues:
            if res.id[0] == ' ':
                dna_residues.append(res)

        # test function
        filter_residues = []
        for res in dna_residues:
            if is_dna(res):
                filter_residues.append(res)

        self.assertEqual(len(filter_residues), 0)

    def test_is_dna_rna(self):

        parser = MMCIFParser(QUIET=True)

        dna_files = os.listdir('../data/test_structures/rna')
        residues = []
        for cif in dna_files:
            structure = parser.get_structure(cif, f'../data/test_structures/rna/{cif}')
            for res in list(structure.get_residues()):
                residues.append((cif, res))

        # remove ions/water/ligands
        dna_residues = []
        for res in residues:
            if res[1].id[0] == ' ':
                dna_residues.append(res)

        # test function
        filter_residues = []
        for res in dna_residues:
            if is_dna(res[1]):
                filter_residues.append(res)

        for res in filter_residues:
            print('structure: ', res[0], 'chain: ', res[1].get_parent().id, 'position: ', res[1].id, 'name: ', res[1].get_resname())
        self.assertEqual(len(filter_residues), 0)


