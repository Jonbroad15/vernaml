from Bio.PDB import *

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
