

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
from itertools import combinations

import sys

module_dir = Path(__file__).parent / "modules"
sys.path.insert(0, str(module_dir))

import arguments
import molpro_parser
from molecule import Molecule
import gaussian
import symmetry


 
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

    vdw_radii = dict.fromkeys(molecule.atoms.keys())


    for key in molecule.atoms.keys():
        remove_digits = str.maketrans("","",digits)
        key_without_digit = key.translate(remove_digits)
        vdw_radius_key = elements_table[elements_table["symbol"]==key_without_digit]["vdw_radius"].values[0]
        vdw_radii[key] = vdw_radius_key

    return vdw_radii

def retrieve_polarizability(molecule, elements_table):
    """ 
    Retrieves the polarizability using mendeleev
    """

    cols = [
        "atomic_number",
        "symbol",
        "dipole_polarizability"
    ]
    elements_table = elements_table[cols] 

    polarizabilities = dict.fromkeys(molecule.atoms.keys())

    for key in molecule.atoms.keys():
        remove_digits = str.maketrans("","",digits)
        key_without_digit = key.translate(remove_digits)
        polarizability_key = elements_table[elements_table["symbol"]==key_without_digit]["dipole_polarizability"].values[0]
        
        # Unit is given in bohr^3
        # Convert to Angstrom for radius usage

        polarizability_key = np.sqrt(polarizability_key) * 0.52917721092  * 100
        polarizabilities[key] = polarizability_key

    
    
    return polarizabilities

def visualize_normal_mode(molecule,normal_mode,mode_number=1,scale_factor = 0.5):
    """ 
    Helper Function to visualize a normal mode with arrows showing the displacement
    """

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    mode = normal_mode[mode_number]
    displacements = mode['displacements']

    # Create dynamic color mapping based on atom types
    atom_colors = {
        'O': 'red',
        'H': 'gray',
        'C': 'black',
        'N': 'blue',
        'S': 'yellow',
        'P': 'orange',
        'F': 'green',
        'Cl': 'lime',
        'Br': 'darkred',
        'I': 'purple'
    }
    
    # Default color for unknown elements
    default_color = 'silver'
    
    # Plot atoms with dynamic coloring
    for atom_key, pos in molecule.items():
        element = atom_key.rstrip('0123456789')  # Extract element name
        color = atom_colors.get(element, default_color)
        label = f"{element}{atom_key[len(element):]}" if any(c.isdigit() for c in atom_key) else element
        
        ax.scatter(*pos, c=color, s=200, label=label)
    
    # Plot displacement vectors with proper atom mapping
    for atom_key, pos in molecule.items():
        element = atom_key.rstrip('0123456789')
        atom_num = int(atom_key[len(element):]) if any(c.isdigit() for c in atom_key) else 1
        
        # Find matching displacement (handles cases like 'O' vs 'O1')
        disp = None
        if element in displacements:
            disp = displacements[element]
        elif atom_key in displacements:
            disp = displacements[atom_key]
        else:
            continue  # Skip if no displacement found
        
        # Convert displacement dict to vector
        dx = disp.get('x', 0) * scale_factor
        dy = disp.get('y', 0) * scale_factor
        dz = disp.get('z', 0) * scale_factor
        
        # Plot arrow
        ax.quiver(pos[0], pos[1], pos[2], 
                 dx, dy, dz, 
                 color='blue', arrow_length_ratio=0.1, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Normal Mode {mode_number} ({mode["symmetry"]})\n'
                f'Wavenumber: {mode["wavenumber"]} cm⁻¹')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.show()




def check_normalization(normal_mode):
    """ 
    Checks the normalization of a normal mode
    """
    displacements = normal_mode["displacements"]
    disp_array = np.array([(d.get("x",0), d.get("y",0),d.get("z",0)) for d in displacements.values()])

    
def check_orthogonality(normal_modes):
    """ 
    Function to check the orthogonality of the normal modes 
    """

    # Each mode --> 1 Displacement vector
    mode_vectors = []
    for mode in normal_modes.values():
        disp = mode["displacements"]
        mode_vector = []
        for atom, displacement in disp.items():
            # Create a vector for the displacement
            vector = np.array([
                displacement.get("x", 0),
                displacement.get("y", 0),
                displacement.get("z", 0)
            ])
            mode_vector.extend(vector)
        mode_vectors.append(np.array(mode_vector))

    # Check orthogonality
    num_modes = len(mode_vectors)
    orthogonal = True
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            dot_product = np.dot(mode_vectors[i], mode_vectors[j])
            if not np.isclose(dot_product, 0, atol=1e-6):
                #print(f"Modes {i+1} and {j+1} are not orthogonal (dot product: {dot_product})")
                orthogonal = False
    if orthogonal:
        print("All modes are orthogonal.")
    else:
        print("Normal Modes are not orthogonal --> Non-Mass-Weighted Normal Modes")
 

def calculateS0(a1,a2,c1,c2):
    """ 
    Function that calculates the prefactor for the overlap integral
    """
    return a1 * a2 * (2 *np.pi * c1**2 * c2**2 / (c1**2 + c2**2))**(3/2)

def calculate_initial_overlap(b1,b2,a1,a2,c1,c2):
    """ 
    Function that calculates the initial overlap S between the two Gaussians
    """    
    r = np.linalg.norm(np.array(b1) - np.array(b2))
    exponent = -r**2 / (2 * (c1**2 + c2**2))
    S0 = calculateS0(a1,a2,c1,c2)
    return S0 * np.exp(exponent)

def calculate_delta_S(S,r,u1,u2,c1,c2):
    """ 
    Function that calculates the change in overlap between two Gaussians where u1,u2 are the displacements at the gaussians
    """
    relative_disp = np.array(u1) - np.array(u2)
    dot_product = np.dot(r, relative_disp)
    return -S * (dot_product / (c1**2 + c2**2))




def compute_mode_norm(displacments):
    """ 
    Computes the euclidian norm of the entire normal mode
    """
    total = 0.0
    for atom, disp in displacments.items():
        total += disp["x"]**2 + disp["y"]**2 + disp["z"]**2
    return np.sqrt(total)



def calculate_volume_change_v2(molecule, vdw_radii, normal_mode):
    """ 
    New function to calculate the volume change using the new approach
    """
    volume_changes = {}   

    displacements = normal_mode['displacements']
    a = normalization_parameter()  # Normalization parameter for Gaussian

    atoms = list(displacements.keys())

    # Calculate the norm of the normal mode
    mode_norm = compute_mode_norm(displacements)
    

    total_delta_S = 0.0
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)): # Avoid Duplicates
            atom1, atom2 = atoms[i], atoms[j]

            # Get equilibrium positions and displacements
            b1 = molecule[atom1]
            b2 = molecule[atom2]
            
            # Get displacements for the current mode
            u1 = np.array([
                displacements[atom1].get('x', 0),
                displacements[atom1].get('y', 0),
                displacements[atom1].get('z', 0)
            ])
            u2 = np.array([
                displacements[atom2].get('x', 0),
                displacements[atom2].get('y', 0),
                displacements[atom2].get('z', 0)
            ])
            # Calculate the initial overlap S0
            c1 = vdw_radii[atom1] / 100  # Convert to Angstrom
            c2 = vdw_radii[atom2] / 100  # Convert to Angstrom
            
            # Calculate the initial overlap S0
            S0 = calculate_initial_overlap(b1, b2, a, a, c1, c2)
            
            # Calculate the Change in overlap delta_S
            r = np.array(b1) - np.array(b2)  # Distance vector between the two atoms
            delta_S = calculate_delta_S(S0, r, u1, u2, c1, c2)

            # Store the results
            volume_changes[f"{atom1}-{atom2}"] = delta_S
            # Take the absolute value of the change
            delta_S_abs = np.abs(delta_S)
            total_delta_S += delta_S_abs

    print(normal_mode, total_delta_S)
    
    return total_delta_S, volume_changes


def plot_molecule_vdw_spheres(molecule, vdw_radii):
    """ 
    Plots the molecule in 3D with VDW spheres
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for atom, coords in molecule.atoms.items():
        x, y, z = coords
        radius = vdw_radii[atom] / 100  # Convert to Angstrom
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
        y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
        z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z

        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.3)

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    plt.title('Molecule with VDW Spheres')
    plt.show()

def main():
    args =  arguments.get_args()
    molpro_out = args.input

    if not molpro_out.exists():
        raise FileNotFoundError(f"File {molpro_out} does not exist.")

    molecule = Molecule(molpro_parser.parse_atoms(molpro_out))
    
    if args.plot == "plot_molecule":
        molecule.plot_molecule()

    # Parse normal modes

    normal_modes = molpro_parser.parse_normal_modes(molpro_out)


    # Make a symmetry detection and analysis
    pg_symbol = symmetry.detect_point_group(molecule)
    pg_symol = "C2v" # testing
    mirror_planes = symmetry.find_mirror_planes(pg_symbol)

    print("============== Point Group Detection ==============")
    print(f"Point Group: {pg_symbol}") 




    # Fetch Elements
    elements_table = fetch_elements()

    # Extract the vdw radii
    vdw_radii = retrieve_vdw_radii(molecule,elements_table) 
    polarizabilities = retrieve_polarizability(molecule, elements_table)


    # Generate a grid of points for density computation
    if args.plot == "plot_density":
        X,Y,Z = gaussian.generate_grid(molecule, padding=5, resolution=100) 
        density = gaussian.compute_density(X,Y,Z, molecule, vdw_radii)
        gaussian.plot_isosurface(X,Y,Z, density, molecule, vdw_radii)

    if args.plot == "plot_molecules_vdw_spheres":
        plot_molecule_vdw_spheres(molecule, vdw_radii)
    
    # Compute pairwise change in overlap
    pairwise_overlap_change = gaussian.compute_pairwise_change_in_overlap(molecule,vdw_radii,normal_modes)
    gaussian.plot_pairwise_overlap_changes_barplot(pairwise_overlap_change, molecule)
    gaussian.plot_pairwise_overlap_changes_heatmap(pairwise_overlap_change, molecule)
    gaussian.plot_total_overlap_change(pairwise_overlap_change, molecule)
   
    
    # Plot and calculate pairwise gradients
    pairwise_gradient_change = gaussian.compute_pairwise_gradients(molecule,vdw_radii,normal_modes)
    gaussian.plot_mode_gradients(pairwise_gradient_change,molecule)

       

    # ======= Testing Section =======

    if args.tests == "1d_gaussian_mult":
        gaussian.plot_1d_gaussian_multiplication()

    if args.tests == "gaussian_overlap":
        gaussian.test_gaussian_overlap_identical()
        gaussian.test_gaussian_overlap_numerical()
        gaussian.test_gaussian_overlap_1d()

    if args.tests == "gradient_test":
        gaussian.gradient_test_1d_gaussian_overlap()
        gaussian.gradient_test_2d_gradient_field()
        gaussian.gradient_test_displacement_of_both_atoms()


    detailed_results_change = []
    results_change = []


    # Check Orthogonality
    check_orthogonality(normal_modes)

    # Calculate the volume change for each normal mode

    changes_results = {}
    delta_S_list = []
    for mode in normal_modes.keys():
        total_delta_S, changes = calculate_volume_change_v2(molecule, vdw_radii, normal_modes[mode])
        changes_results[mode] = {
            "total_change": total_delta_S,
            "changes": changes
        }
        delta_S_list.append(total_delta_S)

    print(delta_S_list)

    # Plot delta_S_list
    plt.figure(figsize=(10, 6))
    plt.bar(changes_results.keys(), delta_S_list, color='skyblue')
    plt.xlabel('Normal Mode')
    plt.ylabel('Volume Change Delta S')
    plt.title('Volume Change per Normal Mode')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("volume_change_per_normal_mode_v2.png", dpi=300)
    #plt.show()






if __name__ == "__main__":
    main()
