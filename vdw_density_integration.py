

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
    # Check if the output file ends with .log
    if args.input.suffix == ".log":
        print("Log file as input detected.")
        
        def parse_log_file(log_file):
            """ 
            Parse log file with multiple normal modes sections
            """
            data = {}
            current_section = None
            current_mode_data = None
            
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines and separators
                    if not line or line.startswith("==="):
                        continue
                    
                    # New section starts with "Normal Modes for"
                    if line.startswith("Normal Modes for"):
                        section_name = line.split('for ')[1].strip()
                        current_section = section_name
                        
                        # Initialize data for this section
                        if section_name not in data:
                            data[section_name] = {
                                'filename': section_name,
                                'modes': []
                            }
                        continue
                    
                    # Skip header lines
                    if line.startswith("Mode") and "Wavenumber" in line:
                        continue
                    
                    # Parse mode data line
                    parts = line.split()
                    if len(parts) >= 3 and current_section:
                        try:
                            mode_num = int(parts[0])
                            wavenumber = float(parts[1])
                            symmetry = parts[2]

                            # Rest is the displacement vector
                            displacement_str = ' '.join(parts[3:]) 
                            displacements = parse_displacement_vector(displacement_str)
                            
                            mode_data = {
                                'mode_number': mode_num,
                                'wavenumber': wavenumber,
                                'symmetry': symmetry,
                                'displacements': displacements
                            }
                            
                            # Add to current section
                            data[current_section]['modes'].append(mode_data)
                            
                        except ValueError:
                            continue
            
            return data

        def parse_displacement_vector(displacement_str):
            """ 
            Parses the displacement vector string into a dictionary
            """
            displacements = {}
        
            # Split by atom entries
            entries = displacement_str.split(')')
        
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue
        
                # Find colon that separates atom label from coordinates
                if ":" in entry:
                    atom_part, coords_part = entry.split(':', 1)
                    atom_label = atom_part.strip()
                     
                    if '(' in coords_part: 
                        coords_str = coords_part.split('(', 1)[1].strip()
                        coords = [float(coord.strip()) for coord in coords_str.split(',')]
                        if len(coords) == 3:
                            displacements[atom_label] = {
                                'x': coords[0],
                                'y': coords[1],
                                'z': coords[2]
                            }
            return displacements
        
        log_data = parse_log_file(args.input)
        # Loop through each file and plot the length of the normal modes
        print("Calculating Norm of Normal Modes from Log File:")
        print("="*50)
        norms = []
        for calculation_type, data in log_data.items():
            print(f"Processing calculation: {calculation_type} from file {data['filename']}")
            modes = data['modes']
            for mode in modes:
                mode_number = mode['mode_number'] 
                norm = compute_mode_norm(mode['displacements'])
                norms.append((calculation_type, mode_number, norm))
                print(f"Mode {mode_number}: Norm = {norm:.6f}")
        
        # Plot the norms in a bar plot 
        df_norms = pd.DataFrame(norms, columns=['Calculation', 'Mode', 'Norm'])
        plt.figure(figsize=(12, 6))
        for calculation, group in df_norms.groupby('Calculation'):
            plt.bar(group['Mode'] + (0.1 * list(log_data.keys()).index(calculation)), group['Norm'], width=0.1, label=calculation)
        plt.xlabel('Normal Mode Number')
        plt.ylabel('Norm')
        plt.title('Norm of Normal Modes from Log File')
        plt.legend()
        plt.savefig('normal_modes_norms_log_file.png', dpi=300)


        def calculate_vector_difference_norm(data, file1,file2,atom,mode_number):
            """ 
            Helper function to calculate the norm of the difference between two vectors
            """

            # Im an idiot an lists in python are zero indexed 
            vec1 = data[file1]["modes"][mode_number]['displacements'][atom]
            vec2 = data[file2]["modes"][mode_number]['displacements'][atom]

            def vector_from_dict(vec_dict):
                """ 
                Converts a vector from the dictionary to a numpy array
                """
                return np.array([vec_dict['x'], vec_dict['y'], vec_dict['z']])
            
            vec1_array = vector_from_dict(vec1)
            vec2_array = vector_from_dict(vec2)

            difference = vec1_array - vec2_array
            norm = np.linalg.norm(difference)
            return difference, norm

        # Extract all the keys from the data
        filenames = list(log_data.keys())
        
        # Make all possible combinations
        vector_differences = {}
        
        # Determine how many modes we have
        num_of_modes = len(log_data[filenames[0]]["modes"])            

        # Determine the list of atoms
        list_of_atoms = list(log_data[filenames[0]]["modes"][0]['displacements'].keys())
        

        norm_results = {}        

        for filename1,filename2 in combinations(filenames,2):
            for mode_number in range(1, num_of_modes + 1):
                for atom in list_of_atoms:
                    difference,norm = calculate_vector_difference_norm(log_data,filename1,filename2,atom,mode_number-1)
                    norm_results[(filename1,filename2,mode_number,atom)] = norm

        # Convert to DataFrame for easier plotting
        df_differences = pd.DataFrame(
            [(k[0], k[1], k[2], k[3], v) for k, v in norm_results.items()],
            columns=['File1', 'File2', 'Mode', 'Atom', 'Norm']
        )
        
        # Plot differences group by Atom make suplot for each atom
        atoms = df_differences['Atom'].unique()
        num_atoms = len(atoms)
        fig, axs = plt.subplots(num_atoms, 1, figsize=(12, 6 * num_atoms), sharex=True)
        if num_atoms == 1:
            axs = [axs]  # Ensure axs is iterable
        for ax, atom in zip(axs, atoms): 
            atom_data = df_differences[df_differences['Atom'] == atom]
            for (file1, file2), group in atom_data.groupby(['File1', 'File2']):
                ax.plot(group['Mode'], group['Norm'], marker='o', label=f'{file1} vs {file2}')
            ax.set_title(f'Norm of Vector Differences for Atom {atom}')
            ax.set_ylabel('Norm of Difference')
            ax.legend()
        axs[-1].set_xlabel('Normal Mode Number')
        plt.tight_layout()
        plt.savefig('normal_modes_vector_differences_log_file.png', dpi=300)
        
        return

    else:
        molpro_out = args.input

    if not molpro_out.exists():
        raise FileNotFoundError(f"File {molpro_out} does not exist.")

    molecule = Molecule(molpro_parser.parse_atoms(molpro_out))
    
    if args.plot == "plot_molecule":
        molecule.plot_molecule()

    # Parse normal modes

    normal_modes = molpro_parser.parse_normal_modes(molpro_out)


    #TODO rewrite this piece of junk
    if args.vector:
        # Check if the output file already exists
        normal_mode_log = "normal_modes_vectors.log"
        if Path(normal_mode_log).exists():
            print(f"Log file {normal_mode_log} already exists, adding new data to the file.")
            # Ask user if they want to clear this file
            user_input = input("Do you want to clear the log file? (y/n): ")
            if user_input.lower() == "y":
                Path(normal_mode_log).unlink()
                print(f"Log file {normal_mode_log} cleared.")
                with open(normal_mode_log, "w") as f:
                    f.write(f"Normal Modes for {molpro_out.name}\n")
                    f.write(f"{'Mode':<6}{'Wavenumber (cm^-1)':<20}{'Symmetry':<10}{'Displacement Vector (x,y,z) in Angstrom':<50}\n")
                    f.write("="*100 + "\n")
                    for mode_number, mode in normal_modes.items():
                        displacements = mode['displacements']
                        f.write(f"{mode_number:<6}{mode['wavenumber']:<20}{mode['symmetry']:<10}")
                        for atom, disp in displacements.items():
                            f.write(f"{atom}: ({disp.get('x',0):.4f}, {disp.get('y',0):.4f}, {disp.get('z',0):.4f})  ")
                        f.write("\n")
            else:
                print(f"Appending to existing log file {normal_mode_log}.")
                with open(normal_mode_log, "a") as f:
                    f.write(f"\nNormal Modes for {molpro_out.name}\n")
                    f.write(f"{'Mode':<6}{'Wavenumber (cm^-1)':<20}{'Symmetry':<10}{'Displacement Vector (x,y,z) in Angstrom':<50}\n")
                    f.write("="*100 + "\n")
                    for mode_number, mode in normal_modes.items():
                        displacements = mode['displacements']
                        f.write(f"{mode_number:<6}{mode['wavenumber']:<20}{mode['symmetry']:<10}")
                        for atom, disp in displacements.items():
                            f.write(f"{atom}: ({disp.get('x',0):.4f}, {disp.get('y',0):.4f}, {disp.get('z',0):.4f})  ")
                        f.write("\n")
        else:
            # If it does not exist, create a new file
            with open(normal_mode_log, "w") as f:
                f.write(f"Normal Modes for {molpro_out.name}\n")
                f.write(f"{'Mode':<6}{'Wavenumber (cm^-1)':<20}{'Symmetry':<10}{'Displacement Vector (x,y,z) in Angstrom':<50}\n")
                f.write("="*100 + "\n")
                for mode_number, mode in normal_modes.items():
                    displacements = mode['displacements']
                    f.write(f"{mode_number:<6}{mode['wavenumber']:<20}{mode['symmetry']:<10}")
                    for atom, disp in displacements.items():
                        f.write(f"{atom}: ({disp.get('x',0):.4f}, {disp.get('y',0):.4f}, {disp.get('z',0):.4f})  ")
                    f.write("\n")

        
         


    




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
     
    # Some further tests for the gradient field 
    #gaussian.gaussian_density_test_2d()
    #gaussian.gaussian_density_test_2d_three_atoms()
    #gaussian.gradient_field_gaussian_density_test_2d()
    #gaussian.gradient_field_gaussian_density_test_2d_three_atoms()
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

    





if __name__ == "__main__":
    main()
