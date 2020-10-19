import unittest
from Bio.PDB import *
from ..interfaces import *

scriptdir = os.path.dirname(os.path.realpath(__file__))
rnadir = os.path.join(scriptdir, '..', '..', '..',
                        'data', 'rna_representative_set')

ligands_file = os.path.join(scriptdir, '..', '..', '..',
                                'data', 'ligand_list.txt')
ligands = []
with open(ligands_file, 'r') as f:
    for line in f.readlines():
        ligands.append(line.strip())

class InterfacesTestCase(unittest.TestCase):


    def test_interface_none(self):

        path = os.path.join(rnadir, '1av6.cif')
        rna,_ = get_interfaces(path, ligands)

        self.assertEqual(len(rna), 0)

    def test_interface_none2(self):

        path ='../data/rna_structure_representative_set/1a9n.cif'
        rna,_,_,_ = get_interfaces(path)

        self.assertEqual(len(rna), 0)

    def test_interface_other(self):

        # Should find a calcium ION here
        path ='../data/rna_structure_representative_set/1dqf.cif'
        _,_,other,_ = get_interfaces(path, cutoff=8)
        binding_molecules = []
        for line in other:
            binding_molecules.append(line[3])
        self.assertIn(('H_CA', 201, ' '), binding_molecules)

    def test_RNA_interface(self):

        path ='../data/rna_structure_representative_set/1dqf.cif'
        rna,_,_,_ = get_interfaces(path)
        self.assertEqual(len(rna), 18)

    def test_interface_protein(self):

        path ='../data/rna_structure_representative_set/1a9n.cif'
        rna,protein,_,_ = get_interfaces(path)

        self.assertIn(('1a9n', 10, 'Q'), protein)
