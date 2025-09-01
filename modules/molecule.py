""" 
Implementation of a general molecule class 
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.colors import Normalize



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

    def plot_molecule_simple_gaussian(self, vdw_radii):
        """ 
        Make a 3D plot of the gaussian density and futher make xy,xz and yz projections of this density
        """

        # First of all extract the parameters of the gaussian functions
        normalization_constant = 1/(3*np.sqrt(2*np.pi))

        # Save the gaussian parameters in a dictionary for each atom
        gaussian_params = {}

        for atom, (x,y,z) in self.atoms.items():
            # TODO add Exception error here
            if atom in vdw_radii:
                radius = vdw_radii[atom] / 100 # Convert Angstrom
            else:
                radius = 1.0 

            # the VDW radius is c in the gaussian function we further need b which is the x,y,z coordinates
            b = np.array([x,y,z])

            gaussian_params[atom] = {
                "a": normalization_constant,
                "b": b,
                "c": radius
            }
        def gaussian_function(x,a,b,c):
            return a * np.exp(-np.sum((x-b)**2)  / (2*c**2))
        
        def total_gaussian_density(point):
            # Total density is given by the multiplication of the individual gaussian functions
            density = 1.0
            for atom, params in gaussian_params.items():
                a = params["a"]
                b = params["b"]
                c = params["c"]
                density += gaussian_function(point, a, b, c)
            return density
        
        # Create a grid of points in 3D space
        grid_size = 80
        x = np.linspace(-8, 8, grid_size)
        y = np.linspace(-8, 8, grid_size)
        z = np.linspace(-8, 8, grid_size)
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        density_values = np.array([total_gaussian_density(point) for point in points])
        density_values = density_values.reshape(X.shape)

        
        # Normalize Density for Colormap
        norm = Normalize(vmin=np.min(density_values), vmax=np.max(density_values))
        normalized_density = norm(density_values.ravel())



        base_cmap = plt.cm.inferno
        cmap_colors = base_cmap(np.linspace(0, 1, 256))
        alpha_mask = np.linspace(0, 1, 256)
        alpha_mask[:90] = 0  # Make low values transparent
        cmap_colors[:, -1] = alpha_mask
        transparent_cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)

        rgba_colors = transparent_cmap(normalized_density)

        # Prepare 3D grid Coordinattes
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        Z_flat = Z.ravel()

        # Filter out fully transparent points
        visible = rgba_colors[:, 3] > 0

        X_visible = X_flat[visible]
        Y_visible = Y_flat[visible]
        Z_visible = Z_flat[visible]
        colors_visible = rgba_colors[visible]


        # Three suplots from different angles
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        ax1.scatter(X_visible, Y_visible, Z_visible, c=colors_visible, marker='o', s=1)
        ax1.set_title('3D Gaussian Density View 1')
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')
        ax1.view_init(elev=30, azim=30)
        ax2.scatter(X_visible, Y_visible, Z_visible, c=colors_visible, marker='o', s=1)
        ax2.set_title('3D Gaussian Density View 2')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')
        ax2.view_init(elev=30, azim=120)
        ax3.scatter(X_visible, Y_visible, Z_visible, c=colors_visible, marker='o', s=1)
        ax3.set_title('3D Gaussian Density View 3')
        ax3.set_xlabel('X axis')
        ax3.set_ylabel('Y axis')
        ax3.set_zlabel('Z axis')
        ax3.view_init(elev=30, azim=210)
        plt.tight_layout()
        plt.savefig('molecule_gaussian_density_3D_views.png', dpi=300)
        plt.close()

        

        # Make projections
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # XY projection
        axs[0].imshow(np.sum(density_values, axis=2), extent=(-3, 3, -3, 3), origin='lower', cmap=transparent_cmap)
        axs[0].set_title('XY Projection')
        axs[0].set_xlabel('X axis')
        axs[0].set_ylabel('Y axis')
        
        # XZ projection
        axs[1].imshow(np.sum(density_values, axis=1), extent=(-3, 3, -3, 3), origin='lower', cmap=transparent_cmap)
        axs[1].set_title('XZ Projection')
        axs[1].set_xlabel('X axis')
        axs[1].set_ylabel('Z axis')

        # YZ projection
        axs[2].imshow(np.sum(density_values, axis=0), extent=(-3, 3, -3, 3), origin='lower', cmap=transparent_cmap)
        axs[2].set_title('YZ Projection')
        axs[2].set_xlabel('Y axis')
        axs[2].set_ylabel('Z axis') 
        plt.savefig('molecule_gaussian_density_projections.png', dpi=300)
        plt.close()