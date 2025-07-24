"""
Modeling density of molecule using gaussian functions
"""

from molecule import Molecule
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gaussian_3d(a=1,b=(1,1,1),c=1,x=(0,0,0)):
    """ 
    Computes a 3D gaussian function

    Parameters
    ----------
    a (float) = bell curve slope
    b (x,y,z) = coordinate as a tuple
    c (float) = variance of the 3d gaussian will the VDW in our case
    x (x,y,z) = 3D coordinates
    """
    bx,by,bz = b
    xx,xy,xz = x
    exponent = -((xx - bx)**2 + (xy - by)**2 + (xz-bz)**2)/(2*c**2)
    return a* np.exp(exponent)

def normalization_parameter():
    """ 
    Computes the normalization parameter a1 for a given Gaussian Function

    This functions are normalized using the vdw_radii and the corresponding sphere with V_sphere = 4 * pi * R³_Vdw
    the normalization parameter is here given by 1/(3*np.sqrt(2pi))
    """


    return 1/(3*np.sqrt(2*np.pi))

def generate_grid(molecule,padding=5,resolution=50):
    """
    Function that generates a grid of points around the molecule
    """ 

    all_coords = np.array([coord for coord in molecule.atoms.values()])
    x_min, y_min, z_min = np.min(all_coords, axis=0) - padding
    x_max, y_max, z_max = np.max(all_coords, axis=0) + padding
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X,Y,Z = np.meshgrid(x, y, z, indexing='ij') 
    return X,Y,Z

def compute_density(X,Y,Z, molecule, vdw_radii):
    """
    Computes the density of the molecule using the gaussian function
    """
    density = np.zeros(X.shape)
    for atom, coords in molecule.atoms.items():
        x, y, z = coords
        vdw_radius = vdw_radii[atom]
        density += gaussian_3d(a=normalization_parameter(), b=(x,y,z), c=vdw_radius, x=(X,Y,Z))
    
   
    return density 

def plot_isosurface(X,Y,Z, density, molecule):
    """
    3D Visualization using Matplotlib with approximate isosurface
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"Density Rage: {density.min()} - {density.max()}")

    if np.max(density) > 1e-6:
        try:
            # Auto calculate a reasonable level
            level = 0.5 * np.percentile(density[density > 0],50)
            print(f"Using level: {level}")
            verts, faces, _, _ = measure.marching_cubes(density, level=level,
                                spacing=(
                                    (X[-1,0,0] - X[0,0,0]) / (X.shape[0] - 1),
                                    (Y[0,-1,0] - Y[0,0,0]) / (Y.shape[1] - 1),
                                    (Z[0,0,-1] - Z[0,0,0]) / (Z.shape[2] - 1)
                                ))
            
            # transform pertices to real coordinates
            verts = np.array([
                X[0,0,0] + verts[:, 0] * (X[-1,0,0] - X[0,0,0]) / (X.shape[0] - 1),
                Y[0,0,0] + verts[:, 1] * (Y[0,-1,0] - Y[0,0,0]) / (Y.shape[1] - 1),
                Z[0,0,0] + verts[:, 2] * (Z[0,0,-1] - Z[0,0,0]) / (Z.shape[2] - 1)
            ]).T
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, 
                            cmap='viridis', edgecolor='none', alpha=0.5)
        except Exception as e:
            print(f"Error in plotting isosurface: {e}")
            print("Falling back to scatter plot.")
            ax.scatter(X, Y, Z, c=density.flatten(), cmap='viridis', alpha=0.5)
    plt.show()