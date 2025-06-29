""" 
Implementation to compute a overlap integral between Gaussian functions which are placed
on a molecule.

Goal is to compute changes in the overlap integral if the Gaussian-spheres are displaced by the 
vibrational normal modes
"""

"""
Parser for Arguments
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from mendeleev.fetch import fetch_table
from string import digits
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from itertools import combinations

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input",
    type=Path,
    required=True,
    help="Path to the molpro output file containing the atomic coordinates.",
)


def parse_atoms(molpro_out):
    """ 
    Parses the atoms with coordinates and symbol from a molpro output file
    """
    with open(molpro_out, 'r') as f:
        lines = f.readlines()
        # Search for Atomic Coordinates in line
        switch = False

        atomic_coordinates = []

        for i, line in enumerate(lines):
            if "Atomic Coordinates" in line:
                switch = True
                continue
            if switch:
                if "Gradient norm" in line:
                    break
                atomic_coordinates.append(line.strip())

        # Remove first entry the header
        atomic_coordinates = list(filter(None, atomic_coordinates))
        # Remove header
        atomic_coordinates = atomic_coordinates[1:]
        
        molecules = {} # Dictionary to hold molecule data
        for atom in atomic_coordinates:
            parts = atom.split()
            # First part is numbering, second part is symbol, fourth fifth and sixth are coordinates
            number = int(parts[0])
            symbol = parts[1]
            atomic_symbol = str(symbol) + str(number)
            x = float(parts[3])
            y = float(parts[4])
            z = float(parts[5])
            coordinates = (x, y, z)
            molecules[atomic_symbol] = coordinates
    return molecules
        
def parse_normal_modes(molpro_out, num_atoms):
    """ 
    Helper Function to Parse the Normal Modes

    1. Determine the non zero normal modes
    2. Extract the Cartesian coordinates of normal modes and sort according to atomic symbols
    """

    with open(molpro_out, 'r') as f:
        lines = f.readlines()
        
        eigenvalues = []
        eigenvalue_switch = False
        normal_modes = []
        normal_modes_switch = False

        for i, line in enumerate(lines):
            if "Mass Weighted 2nd Derivative Matrix Eigenvalues" in line:
                eigenvalue_switch = True
                continue
            if eigenvalue_switch:
                if "Mass Weighted 2nd Derivative Matrix Eigenvectors" in line:
                    eigenvalue_switch = False
                    # Turn on the normal modes switch
                    normal_modes_switch = True
                    continue
                eigenvalues.append(line.strip())
            
            if normal_modes_switch:
                if "Low Vibration" in line:
                    normal_modes_switch = False
                    continue
                normal_modes.append(line.strip())


        # Process Eigenvalues
        eigenvalues = list(filter(None, eigenvalues))
        eigenvalues = eigenvalues[0].split()

        # Remove first two entries and convert to float
        eigenvalues = [float(v) for v in eigenvalues[2:]]    
        # Count the number of non-zero eigenvalues
        non_zero_eigenvalues = [v for v in eigenvalues if v != 0]
                                

        cleaned_lines = []
        for line in normal_modes:
            parts = list(filter(None, line.split())) 
            # First part is row number rest are eigenvectors
            if parts:
                cleaned_lines.append(parts[1:]) 

        # Remove the first entry
        cleaned_lines = cleaned_lines[1:]
        # Remove empty lines
        cleaned_lines = list(filter(None, cleaned_lines))
        
        

        num_modes = len(cleaned_lines[0])
        modes = {}
        print(num_modes)

        for mode_idx in range(num_modes):
            mode_dict = {}
            for atom_idx in range(num_atoms):

                # Each atom has three rows
                row_start = atom_idx * 3
                if row_start + 2 < len(cleaned_lines):
                    x = float(cleaned_lines[row_start][mode_idx])
                    y = float(cleaned_lines[row_start + 1][mode_idx])
                    z = float(cleaned_lines[row_start + 2][mode_idx])
                    mode_dict[atom_idx + 1] = (x, y, z)
            
            modes[f"Mode {mode_idx + 1}"] = mode_dict
        print(modes)
        #TODO FIX this this
 
def fetch_elements():
    """ 
    Fetches the elements table using Mendeleev
    """ 
    ptable = fetch_table("elements")
    return ptable

def remove_string_digits(string):
    remove_digits = str.maketrans("","",digits)
    return string.translate(remove_digits)

def retrieve_vdw_radii(molecule, elements_table):
    """ 
    Retrieves the VDW radii using mendeleev
    """

    cols = [
        "atomic_number",
        "symbol",
        "vdw_radius"
    ]
    elements_table = elements_table[cols]

    vdw_radii = dict.fromkeys(molecule.keys())


    for key in molecule.keys():
        remove_digits = str.maketrans("","",digits)
        key_without_digit = key.translate(remove_digits)
        vdw_radius_key = elements_table[elements_table["symbol"]==key_without_digit]["vdw_radius"].values[0]
        vdw_radii[key] = vdw_radius_key

    return vdw_radii

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


def plot_vdw_gaussian_density(molecule, vdw_radii, resolution=100, isovalue=0.2):
    """
    Plot 3D vdW density using Gaussian functions.
    
    Args:
        molecule: Dict of {atom_name: (x,y,z)} in Å
        vdw_radii: Dict of {element: vdW_radius_in_pm}
        resolution: Grid resolution
        isovalue: Density level to display (0-1)
    """
    # Convert to Å and prepare grid
    coords = np.array(list(molecule.values()))
    elements = [name for name in molecule.keys()]
    padding = 3.0  # Å
    
    # Create 3D grid
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    z = np.linspace(z_min, z_max, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Compute density field
    density = np.zeros_like(X)
    for (name, pos), element in zip(molecule.items(), elements):
        c = vdw_radii[element] / 100  # pm → Å
        r_sq = (X-pos[0])**2 + (Y-pos[1])**2 + (Z-pos[2])**2
        density += np.exp(-r_sq / (2 * c**2))
    
    # Normalize density
    density = (density - density.min()) / (density.max() - density.min())
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot isosurface
    verts, faces, _, _ = measure.marching_cubes(
        density, level=isovalue, 
        spacing=(
            (x_max-x_min)/resolution,
            (y_max-y_min)/resolution,
            (z_max-z_min)/resolution
        )
    )
    
    # Transform vertices to real coordinates
    verts += [x_min, y_min, z_min]
    
    # Color by height (Z coordinate)
    mesh = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap=cm.viridis, alpha=0.5, edgecolor='none'
    )
    
    # Plot atoms as spheres
    for (name, pos), element in zip(molecule.items(), elements):
        radius = vdw_radii[element] / 100
        ax.scatter(*pos, s=300*radius, 
                  label=f"{name} ({element}, {radius:.2f}Å)",
                  depthshade=False)
    
    # Add colorbar
    fig.colorbar(mesh, ax=ax, shrink=0.5, label='Normalized Density')
    
    # Formatting
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'vdW Gaussian Density (Isovalue = {isovalue})')
    ax.legend()
    plt.tight_layout()
    plt.show()


def gaussian_overlap(a1,b1,c1,a2,b2,c2):
    """
    Implementation of the overlap formula between the two gaussians
    """

    c_sq = (c1**2 * c2**2) /(c1**2 + c2**2)
    exponent = -np.sum((np.array(b1) - np.array(b2))**2 / (2*(c1**2 + c2**2)))
    prefactor = a1*a2*(2*np.pi)**(3/2) *c_sq**(3/2)
    return prefactor*np.exp(exponent)

def compute_pairwise_vdw_overlaps(molecule, vdw_radii):

    results = []
    atoms = list(molecule.items())
    for (symbol1, pos1), (symbol2,pos2) in combinations(atoms,2):

        c1 = vdw_radii[symbol1] / 100 # Convert to Angstrom
        c2 = vdw_radii[symbol2] / 100

        distance = np.linalg.norm(np.array(pos1)- np.array(pos2))

        overlap = gaussian_overlap(
            a1=1, b1=pos1, c1=c1,
            a2=1, b2=pos2, c2=c2
            )
        # Maximum overlap if both atoms conicide
        max_overlap = (2*np.pi)**(3/2) * ((c1**2 * c2**2)/(c1**2 + c2**2) )**(3/2)
        overlap_ratio = overlap / max_overlap

        results.append({
            "atom1": symbol1,
            "atom2": symbol2,
            "distance": distance,
            "gaussian_overlap": overlap,
            "overlap_ratio" : overlap_ratio,
            "c1" : c1,
            "c2": c2,
        })
    return results



def main():
    args = parser.parse_args()
    molpro_out = args.input

    if not molpro_out.exists():
        raise FileNotFoundError(f"File {molpro_out} does not exist.")

    molecule = parse_atoms(molpro_out)
    num_atoms = len(molecule)
    #parse_normal_modes(molpro_out, num_atoms)
    
    # Fetch Elements
    elements_table = fetch_elements()
    

    # Extract the vdw radii
    vdw_radii = retrieve_vdw_radii(molecule,elements_table) 
    pairwise_overlaps_dict = compute_pairwise_vdw_overlaps(molecule,vdw_radii)
    print(pairwise_overlaps_dict)
    plot_vdw_gaussian_density(molecule,vdw_radii,resolution=500,isovalue=0.5)




if __name__ == "__main__":
    main()
