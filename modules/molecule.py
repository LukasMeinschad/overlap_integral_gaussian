""" 
Implementation of a general molecule class 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



class Molecule:
    """ 
    Class to represent a molecule with its atoms and coordinates
    """

    def __init__(self, atoms):
        """
        Initialize the molecule with a dictionary of atoms and their coordinates
        :param atoms: Dictionary where keys are atomic symbols and values are tuples of coordinates
        """
        self.atoms = atoms

    def __repr__(self):
        return f"Molecule(atoms={self.atoms})"

    num_atoms = property(lambda self: len(self.atoms))

    atom_colors = {
        "C": "black",
        "H": "gray",
        "O": "red",
        "N": "blue",
        "S": "yellow",
        "F": "green",
    }
    atom_sizes = {
        "C": 100,
        "H": 50,
        "O": 80,
        "N": 90,
        "S": 110,
        "F": 70,
    }

    def plot_molecule(self):
        """ 
        Make a 3D plot of the molecule, this is mainly to check if the import is coorect from molpro
        """

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection="3d")

        for atom, coords in self.atoms.items():
            x, y, z = coords
            color = self.atom_colors.get(atom[0], "gray")
            size = self.atom_sizes.get(atom[0], 50)
            ax.scatter(x, y, z, color=color, s=size, label=atom)
        all_coords = []
        for coords in self.atoms.values():
            all_coords.append(coords)
        
        # Draw lines between atoms if they are withing a threshold distance
        threshold = 1.8 # Angstrom
        for i in range(len(all_coords)):
            for j in range(i + 1, len(all_coords)):
                dist = np.linalg.norm(np.array(all_coords[i]) - np.array(all_coords[j]))
                if dist < threshold:
                    ax.plot([all_coords[i][0], all_coords[j][0]], 
                            [all_coords[i][1], all_coords[j][1]], 
                            [all_coords[i][2], all_coords[j][2]], color="gray", alpha=0.5)
        ax.set_xlabel("X (Angstrom)")
        ax.set_ylabel("Y (Angstrom)")
        ax.set_zlabel("Z (Angstrom)")
        ax.set_title("Molecule Structure")
        ax.legend()
        plt.show()


    