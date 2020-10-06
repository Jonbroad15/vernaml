from Bio.PDB import *
import os

scriptdir = os.path.dirname(__file__)


def get_interfaces(cif_path, cutoff=5):
    """Obtain interfacing residues within a single structure of polymers. Uses
   KDTree data structure for vector search. If structure not found in complex
    databse, it is automatically downloaded from RCSB.

    Args:
        `pdb_path (str)`: Path to PDB structure to analyze
        `cutoff (float, int)`: Number of Angstroms to use as cutoff distance
        for interface calculation.
    Returns:
        `list`: containing all Residue objects belonging to interface. As pairs
        of residues (res_1, res_2)
        `Structure`: BioPython Structure object
    """
    parser = MMCIFParser(QUIET=True)
    structure_id = cif_path[-8:-4]
    print(f'Loading structure {structure_id}...')
    try:
        structure = parser.get_structure('X', cif_path)
    except FileNotFoundError:
        print(f'Error file {cif_path} not found')
        return [], [], [], None
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
                other_interface_residues.append((structure_id, res_1.id[1], c1, res_2.id))
            elif res_2.get_resname() in rna_bases:
                other_interface_residues.append((structure_id, res_2.id[1], c2, res_1.id))

    return list(set(rna_interface_residues)), list(set(protein_interface_residues)), list(set(other_interface_residues)), structure


