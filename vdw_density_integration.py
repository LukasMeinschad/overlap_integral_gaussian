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

def parse_normal_modes(molpro_out):
    """ 
    Function to parse the normal modes from a molpro output file. The Normal Modes are given in blocks of 5
    """
    with open(molpro_out, "r") as f:
        lines = f.readlines()

        # Find all normal mode sections
        mode_blocks = []
        current_block = []
        in_block = False
        for line in lines:
            if "Normal Modes" in line and "low/zero frequencies" not in line:
                in_block = True
                current_block = [line]
                continue
            elif "Normal Modes of low/zero frequencies" in line and in_block:
                in_block = False
                if current_block:
                    mode_blocks.append(current_block)
                continue
            elif in_block:
                current_block.append(line)

        # Combine all blocks into a continious block
        full_block = []
        for block in mode_blocks:
            full_block.extend(block)
        
        normal_modes = {}

        # First find all mode headers (may be in multiple lines)
        mode_headers = []
        mode_lines = []
        

        for line in full_block:
            if line.startswith(" " * 12) and not any(x in line for x in ["Wavenumbers", "Intensities"]):

               # Clean up the line by replacing multiple spaces
               cleaned_line = " ".join(line.strip().split())
               mode_lines.append(cleaned_line)

        combined_mode_line = " ".join(mode_lines)
        
        # Parse mode numbers and symmetry
        mode_info = []
        parts = combined_mode_line.split()
        i = 0
        while i < len(parts):
            if parts[i].isdigit():
                mode_num = int(parts[i])
                symmetry = parts[i+1] if i+1 < len(parts) else ""
                mode_info.append((mode_num,symmetry))
                i +=2
            else:
                i += 1
        num_modes = len(mode_info)

        # parse wavenumbers and intensities
        wavenumbers = []
        intensities_km = []
        intensities_rel = []
        for line in full_block:
            if "Wavenumbers" in line:
                parts = line.split()
                new_wavenumbers = list(map(float, parts[2:2+num_modes]))
                wavenumbers.extend(new_wavenumbers)
            elif "Intensities [km/mol]" in line:
                parts = line.split()
                new_intensities = list(map(float,parts[2:2+num_modes]))
                intensities_km.extend(new_intensities)
            elif "Intensities [relative]" in line:
                parts = line.split()
                new_intensities = list(map(float,parts[2:2+num_modes]))
                intensities_rel.extend(new_intensities)

        # Some error printing
        if len(wavenumbers) != num_modes:
            raise ValueError(f"Expected {num_modes} wavenumbers, got {len(wavenumbers)}")
        if len(intensities_km) != num_modes:
            raise ValueError(f"Expected {num_modes} km/mol intensities, got {len(intensities_km)}")
        if len(intensities_rel) != num_modes:
            raise ValueError(f"Expected {num_modes} [relative] intensities, got {len(intensities_rel)}")
        
        for i, (mode_num, sym) in enumerate(mode_info):
            normal_modes[mode_num] = {
                "symmetry": sym,
                "wavenumber": wavenumbers[i] if i < len(wavenumbers) else 0.0,
                "intensity_km_mol": intensities_km[i] if i < len(intensities_km) else 0.0,
                "intensity_relative": intensities_rel[i] if i < len(intensities_rel) else 0.0,
                "displacements": {}
            }
        
        current_block_modes = [] # Track modes in current block
        current_block_size = 5 # Molpro Block size

        for line in full_block:
            if not line.strip() or any(x in line for x in ["Normal Modes", "Wavenumbers", "Intensities"]):
                continue
            
            # Check if its a mode header line
            if line.startswith(" " * 12) and not any(x in line for x in ["Wavenumbers", "Intensities"]):
                # This is new block of modes
                cleaned_line = " ".join(line.strip().split())
                parts = cleaned_line.split()
                current_block_modes = []
                i = 0
                while i < len(parts):
                    if parts[i].isdigit:
                        mode_num = int(parts[i])
                        current_block_modes.append(mode_num)
                        i += 2 # Skip symmetry label
                    else:
                        i +=1
                continue

            # Now process the displacement lines
            parts = line.split()
            if len(parts) < 2: # Skip lines without data
                continue

            label = parts[0]
            values = list(map(float,parts[1:1 + len(current_block_modes)])) # only take values of current block
            

            # Parse the atom info
            element = "".join([c for c in label if c.isalpha() and c not in ["X","Y","Z"]])
            atom_num = "".join([c for c in label if c.isdigit()])
            direction = label[len(element):-len(atom_num)].lower()
            atom_name = f"{element}{atom_num}" if atom_num else element

            # Add displacment for modes in current block
            for i, mode_num in enumerate(current_block_modes):
                if mode_num not in normal_modes:
                    continue # Skip if mode wasn't properly registered
                if atom_name not in normal_modes[mode_num]["displacements"]:
                    normal_modes[mode_num]["displacements"][atom_name] = {}
                normal_modes[mode_num]["displacements"][atom_name][direction] = values[i]
        return normal_modes
        



 
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


def gaussian_overlap(a1,b1,c1,a2,b2,c2):
    """
    Implementation of the overlap formula between the two gaussians
    """
    c_sq = (c1**2 * c2**2) /(c1**2 + c2**2)
    exponent = -np.sum((np.array(b1) - np.array(b2))**2 / (2*(c1**2 + c2**2)))
    prefactor = a1*a2*(2*np.pi)**(3/2) *c_sq**(3/2)
    return prefactor*np.exp(exponent)

def normalization_parameter():
    """ 
    Computes the normalization parameter a1 for a given Gaussian Function

    This functions are normalized using the vdw_radii and the corresponding sphere with V_sphere = 4 * pi * R³_Vdw
    the normalization parameter is here given by 1/(3*np.sqrt(2pi))
    """


    return 1/(3*np.sqrt(2*np.pi))

def compute_pairwise_vdw_overlaps(molecule, vdw_radii):

    results = []
    atoms = list(molecule.items())
    for (symbol1, pos1), (symbol2,pos2) in combinations(atoms,2):

        c1 = vdw_radii[symbol1] / 100 # Convert to Angstrom
        c2 = vdw_radii[symbol2] / 100


        # We have to discuss this
        a = normalization_parameter() 
         

        

        distance = np.linalg.norm(np.array(pos1)- np.array(pos2))

        overlap = gaussian_overlap(
            a1=a, b1=pos1, c1=c1,
            a2=a, b2=pos2, c2=c2
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

def check_normalization(normal_mode):
    """ 
    Checks the normalization of a normal mode
    """
    displacements = normal_mode["displacements"]
    disp_array = np.array([(d.get("x",0), d.get("y",0),d.get("z",0)) for d in displacements.values()])


def calculate_volume_change(molecule, vdw_radii, normal_mode, lam_dis=0.1):
    """ 
    Calculates the Volume change under displacment of the molecule using a normal mode in its cartesian representation
    
    If we consider that b_i is the center of a Gaussian and we already implemented the overlap integral

    we can calculate the gradient along the d:= b_i - b_j direction

    If b_1 --> b_1 + lam*q_1 is our first displacement and b_2 --> b_2 + lam*q_2 is our second displacement
    we know that d = b_1 -b_2 = d + lambda* delta q

    Now the change is given by -S / (c_1**2 + c_2**2) * (d * lambda * delta q)

    which gives as a fractional change in respect to the original overlap integral

    delta S / S = - lambda / (c_1 ** 2 + c_2 ** 2) * d * delta q
    """

    results = {}
    
    #check_normalization(normal_mode)
    masses_lookup = {
        "H": 1.0078,
        "O": 15.999,
        "C": 12.011
    }
    # Loop pairwise over the atoms
    for (atom1, b_1), (atom2, b_2) in combinations(molecule.items(), 2):
        # Get the displacements for the current mode
        disp1 = normal_mode['displacements'].get(atom1, {})
        disp2 = normal_mode['displacements'].get(atom2, {})
        element1 = "".join([c for c in atom1 if not c.isdigit()])
        element2 = "".join([c for c in atom2 if not c.isdigit()])
        mass1 = masses_lookup.get(element1, 1.0)
        mass2 = masses_lookup.get(element2,1.0)
        


        # Calculate delta q
        delta_q = np.array([
            disp1.get('x', 0) - disp2.get('x', 0),
            disp1.get('y', 0) - disp2.get('y', 0),
            disp1.get('z', 0) - disp2.get('z', 0)
        ])
        # Massweight
       #ä delta_q = np.array([
       #ä     disp1.get('x', 0)*np.sqrt(mass1) - disp2.get('x', 0)*np.sqrt(mass2),
       #ä     disp1.get('y', 0)*np.sqrt(mass1) - disp2.get('y', 0)*np.sqrt(mass2),
       #ä     disp1.get('z', 0)*np.sqrt(mass1) - disp2.get('z', 0)*np.sqrt(mass2)
       #ä ])

        # Calculate the distance vector d
        d = np.array(b_1) - np.array(b_2)
        # The magnitued c1^2 and c2^2 are given by the vdw radii
        c1 = vdw_radii[atom1] / 100  # Convert to Angstrom
        c2 = vdw_radii[atom2] / 100  # Convert to Angstrom

        c1_sq = c1 ** 2
        c2_sq = c2 ** 2

        # Fractional change in overlap integral
        delta_S_over_S = - lam_dis / (c1_sq + c2_sq) * np.dot(d, delta_q)

        # Calculate absolute value and write to results
        delta_S_over_S_abs = abs(delta_S_over_S)
        results[(atom1, atom2)] = {
            "delta_S_over_S": delta_S_over_S,
            "delta_S_over_S_abs": delta_S_over_S_abs,
            "c1": c1,
            "c2": c2,
            "d": d,
            "delta_q": delta_q
        }
    
    # Calculate the total change as a sum of absolute changes
    total_change = sum(res['delta_S_over_S_abs'] for res in results.values())
    return total_change, results
        

def barplot_change(results_change):
    """ 
    Visualizes the changes in volume in a barplot
    """    

    # Convert results to DataFrame
    df = pd.DataFrame(results_change)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(df['mode'], df['total_change'], color='skyblue')
    
    # Add labels and title
    plt.xlabel('Normal Mode')
    plt.ylabel('Total Change in Volume (Fractional)')
    plt.title('Volume Change per Normal Mode')
    plt.xticks(rotation=45)
    
    # Show plot
    plt.tight_layout()
    plt.savefig("volume_change_per_normal_mode.png", dpi=300)
    plt.show()


def main():
    args = parser.parse_args()
    molpro_out = args.input

    if not molpro_out.exists():
        raise FileNotFoundError(f"File {molpro_out} does not exist.")

    molecule = parse_atoms(molpro_out)
    num_atoms = len(molecule)

    # Parse normal modes

    normal_modes = parse_normal_modes(molpro_out)

    
    # Fetch Elements
    elements_table = fetch_elements()
    

    # Extract the vdw radii
    vdw_radii = retrieve_vdw_radii(molecule,elements_table) 
    pairwise_overlaps_dict = compute_pairwise_vdw_overlaps(molecule,vdw_radii)

    detailed_results_change = []
    results_change = []
    for mode in normal_modes.keys():

        total_change_mode, detailed_results = calculate_volume_change(molecule, vdw_radii, normal_modes[mode], lam_dis=0.1)
        detailed_results_change.append({
            "mode": mode,
            "total_change": total_change_mode,
            "detailed_results": detailed_results
        }) 
        results_change.append({
            "mode": mode,
            "total_change": total_change_mode
        })
    

    
    
    
    
    # Some visualization functions
    #plot_vdw_gaussian_density(molecule,vdw_radii,resolution=50,isovalue=0.5)
    #visualize_normal_mode(molecule,normal_modes,3)
    barplot_change(results_change)




if __name__ == "__main__":
    main()
