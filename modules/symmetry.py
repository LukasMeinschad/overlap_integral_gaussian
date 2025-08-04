from pymatgen.core import Molecule as PymatgenMolecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from molecule import Molecule
import numpy as np

c2v_ops = {
    "E": np.eye(3),
    "C2(z)": np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]]), # 180 degree rotation around z-axis
    "σ(xz)": np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]]), # Reflection in the xz-plane
    "σ(yz)": np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]), # Reflection in the yz-plane
}

def detect_point_group(molecule):
    """ 
    Uses pymatgen to detect the point group symmetry of a molecule.
    """
    atoms = molecule.atoms 
    coords = [coords for coords in atoms.values()]
    atoms = [atom for atom in atoms.keys()]
    pymatgen_molecule = PymatgenMolecule(atoms, coords)
    pga = PointGroupAnalyzer(pymatgen_molecule)
    point_group = pga.get_pointgroup()
    return point_group,


def find_mirror_planes(pg_symbol):
    """ 
    Finds the mirror planes in the symmetry operations.
    """

    mirror_planes = []
    if pg_symbol == "C2v":
       for op_matrix in c2v_ops.values():
           print(np.linalg.det(op_matrix)) 
