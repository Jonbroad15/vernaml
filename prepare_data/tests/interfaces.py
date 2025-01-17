import unittest
from Bio.PDB import *
from ..interfaces import *
import os

scriptdir = os.path.dirname(os.path.realpath(__file__))
rnadir = os.path.join(scriptdir, '..', '..',
                        'data', 'structures')
test_structures_dir = os.path.join(scriptdir, '..', '..',
                        'data', 'test_structures')
data_dir = os.path.join(scriptdir, '..', '..',
                        'data')
ligands_file = os.path.join(scriptdir, '..', '..',
                                'data', 'ligand_list.txt')
ligands = []
with open(ligands_file, 'r') as f:
    for line in f.readlines():
        ligands.append(line.strip())

parser = MMCIFParser(QUIET=True)

class InterfacesTestCase(unittest.TestCase):

    def test_get_repr_set(self):
        bgsu_file = os.path.join(data_dir, 'nrlist_3.145_4.0A.csv')

        repr_set = get_repr_set(bgsu_file)
        print(repr_set)

    def test_get_interfaces_none(self):

        path = os.path.join(rnadir, '1av6.cif')
        interfaces,_ = get_interfaces(path, ligands, cutoff=10)

        self.assertEqual(len(interfaces), 6)

    def test_get_interfaces_SAM(self):

        path = os.path.join(test_structures_dir, '5fk3.cif')
        interfaces,_ = get_interfaces(path, ligands, cutoff=50)

        known_interfaces = find_ligand_annotations(path, ligands)
        self.assertIn(known_interfaces, interfaces)

#    def test_get_interfaces_other(self):
#
#        # Should find a calcium ION here
#        path ='../data/rna_structure_representative_set/1dqf.cif'
#        _,_,other,_ = get_interfaces(path, cutoff=8)
#        binding_molecules = []
#        for line in other:
#            binding_molecules.append(line[3])
#        self.assertIn(('H_CA', 201, ' '), binding_molecules)
#
#    def test_RNA_interface(self):
#
#        path ='../data/rna_structure_representative_set/1dqf.cif'
#        rna,_,_,_ = get_interfaces(path)
#        self.assertEqual(len(rna), 18)
#
#    def test_interface_protein(self):
#
#        path ='../data/rna_structure_representative_set/1a9n.cif'
#        rna,protein,_  = get_interfaces(path)
#
#        self.assertIn(('1a9n', 10, 'Q'), protein)
