"""
Modeling density of molecule using gaussian functions
"""

from molecule import Molecule
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from scipy.integrate import nquad
from scipy.integrate import quad
from itertools import combinations
import math

def gaussian_1d(a=1, b=0, c=1, x=0):
    """ 
    Computes a 1D gaussian function
    """
    exponent = - ((x - b) ** 2) / (2 * c ** 2)
    return a * np.exp(exponent)

def gaussian_2d(a=1, b=(0, 0), c=1, x=(0, 0)):
    """ 
    Computes a 2D gaussina function
    """
    bx, by = b
    xx, xy = x 
    exponent = -((xx - bx) ** 2 + (xy - by) ** 2) / (2 * c ** 2)
    return a * np.exp(exponent)

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

def multiply_gaussians(a1, b1,c1,a2,b2,c2):
    """ 
    Performs the multiplication of two 3D gaussian functions
    """

    c_sq_new = (c1**2 * c2**2) / (c1**2 + c2**2)
    c_new = np.sqrt(c_sq_new)
    
    # Compute new center
    b1_arr = np.array(b1)
    b2_arr = np.array(b2)
    b_new = (b1_arr * c2**2 + b2_arr * c1**2) / (c1**2 + c2**2)

    # compute new amplitude
    exponent = -np.sum((b1_arr - b2_arr)**2) / (2*(c1**2 + c2**2))
    a_new = a1 * a2 * np.exp(exponent)

    return a_new, tuple(b_new), c_new

def multiply_gaussians_1d(a1, b1, c1, a2, b2, c2):
    """ 
    Performs the multiplication of two 1D gaussian functions
    """
    c_sq_new = (c1**2 * c2**2) / (c1**2 + c2**2)
    c_new = np.sqrt(c_sq_new)
    
    # Compute new center
    b_new = (b1 * c2**2 + b2 * c1**2) / (c1**2 + c2**2)

    # compute new amplitude
    exponent = -((b1 - b2)**2) / (2*(c1**2 + c2**2))
    a_new = a1 * a2 * np.exp(exponent)

    return a_new, b_new, c_new

def gaussian_overlap_1d(a1, b1, c1, a2, b2, c2):
    """ 
    Computes the overlap of two 1D gaussian functions
    """
    c_sq = (c1**2 * c2**2) / (c1**2 + c2**2)
    exponent = -((b1 - b2)**2) / (2*(c1**2 + c2**2))
    prefactor = a1 * a2 * np.sqrt(2 * np.pi) * c_sq**0.5
    return prefactor * np.exp(exponent)

def gaussian_overlap_2d(a1, b1, c1, a2, b2, c2):
    """ 
    Computes the overlap of two 2D gaussian functions
    """
    c_sq = (c1**2 * c2**2) / (c1**2 + c2**2)
    exponent = -((b1[0] - b2[0])**2 + (b1[1] - b2[1])**2) / (2*(c1**2 + c2**2))
    prefactor = a1 * a2 * (2 * np.pi) * c_sq
    return prefactor * np.exp(exponent)


def gaussian_overlap(a1,b1,c1,a2,b2,c2):
    """
    Implementation of the overlap formula between the two gaussians
    """
    c_sq = (c1**2 * c2**2) /(c1**2 + c2**2)
    exponent = -np.sum((np.array(b1) - np.array(b2))**2 / (2*(c1**2 + c2**2)))
    prefactor = a1*a2*(2*np.pi)**(3/2) *c_sq**(3/2)
    return prefactor*np.exp(exponent)

def compute_gradient(a1,b1,c1,a2,b2,c2,u1,u2):
    """ 
    Computes the gradient after a displacment by the vector u
    """ 
    S = gaussian_overlap(a1,b1,c1,a2,b2,c2)
    if u1 is not None:
        b1 = np.array(b1) + np.array(u1)
    if u2 is not None:
        b2 = np.array(b2) + np.array(u2)
    
    # Compute the displacement vector
    r = np.array(b1) - np.array(b2)
    grad_u1 = (S / (c1**2 + c2**2)) * r # Gradient for atom 1
    grad_u2 = -grad_u1 # Gradient for atom 2 because of symmetry
    return grad_u1, grad_u2

def compute_pairwise_gradients(molecule, vdw_radii, normal_modes):
    """ 
    Function that computes the pairwise gradients for all atoms under normal mode displacements,
    sums them up per mode
    """
    results = {}
    atom_names = list(molecule.atoms.keys())
    n_atoms = len(atom_names)

    for mode_id, mode_data in normal_modes.items():
        # Initialize the results
        mode_results = {
            "wavenumber": mode_data["wavenumber"],
            "atom_gradients": np.zeros((n_atoms,3)), #store net gradients per atom
            "pairwise_gradients": {}, # Stores indiviudal pair gradients
            "per_atom_gradient": np.zeros((n_atoms,3)), # Store per atom gradients
            "total_gradient_magnitude": 0.0,
            "displacements": np.zeros((n_atoms,3)) # Store displacments for plotting
        }
        # Collect all displacments for this mode
        displacements = np.zeros((n_atoms,3))
        for i, atom in enumerate(atom_names):
            disp = mode_data["displacements"][atom]
            displacements[i] = np.array([disp["x"], disp["y"], disp["z"]])
        mode_results["displacements"] = displacements

        # Compute gradients for all unique atom pairs
        for i,j in combinations(range(n_atoms),2):
            atom1, atom2 = atom_names[i], atom_names[j]

            # Get coordinates and vdw
            b1 = np.array(molecule.atoms[atom1])
            b2 = np.array(molecule.atoms[atom2])
            c1,c2 = vdw_radii[atom1] / 100, vdw_radii[atom2] / 100 # Convert to Angstrom
            # Get displacments
            u1 = displacements[i]
            u2 = displacements[j]

            # Compute the gradients
            grad_u1, grad_u2 = compute_gradient(
                a1=normalization_parameter(),b1=b1,c1=c1,
                a2=normalization_parameter(),b2=b2,c2=c2,
                u1=u1, u2=u2
            )

            mode_results["atom_gradients"][i] += grad_u1
            mode_results["atom_gradients"][j] += grad_u2
            mode_results["pairwise_gradients"][f"{atom1}-{atom2}"] = (grad_u1, grad_u2)

        # Calculate total gradient change
        grad_dot_disp = np.sum(mode_results["atom_gradients"] * displacements, axis=1)
        mode_results["total_gradient_magnitude"] = np.sum(np.abs(grad_dot_disp))
        results[mode_id] = mode_results
    return results



def compute_pairwise_change_in_overlap(molecule,vdw_radii,normal_modes):
    """
    Computes the pairwise change in overlap for all atoms in the molecule after displacement by a normal mode vector
    """
    pairwise_changes = {}

    atom_names = list(molecule.atoms.keys())
    n_atoms = molecule.num_atoms

    # Process each normal mode
    for mode_id, mode_data in normal_modes.items():
        mode_changes = {}
        total_pairs = 0
       

        # Calculate for all unique atom pairs
        for (i,j) in combinations(range(n_atoms), 2):
            atom1 = atom_names[i]
            atoms2 = atom_names[j]

            # Get original coordinates
            b1 = np.array(molecule.atoms[atom1])
            b2 = np.array(molecule.atoms[atoms2])

            # Get displacments of normal mode
            u1 = np.array([mode_data['displacements'][atom1]["x"],
                            mode_data['displacements'][atom1]["y"],
                            mode_data['displacements'][atom1]["z"]])
            u2 = np.array([mode_data['displacements'][atoms2]["x"],
                            mode_data['displacements'][atoms2]["y"],
                            mode_data['displacements'][atoms2]["z"]])

            # Get the VDW radii
            c1 = vdw_radii[atom1] / 100 # Convert to Angstrom
            c2 = vdw_radii[atoms2] / 100 # Convert to Angstrom

            # Calculate the original and displaced overlaps
            S0 = gaussian_overlap(a1=normalization_parameter(), b1=b1, c1=c1, a2=normalization_parameter(), b2=b2, c2=c2)
            S_new = gaussian_overlap(a1=normalization_parameter(), b1=b1 + u1, c1=c1, a2=normalization_parameter(), b2=b2 + u2, c2=c2)
            delta_S = S_new - S0

            pair_key = f"{atom1}-{atoms2}"
            mode_changes[pair_key] = delta_S
            total_pairs += 1
        
        pairwise_changes[mode_id] = {
            "wavenumber": mode_data['wavenumber'],
            "changes": mode_changes,
            "total_pairs": total_pairs
        }
    return pairwise_changes




# ======= Visualization of Pairwise Overlap Change =======

def plot_pairwise_overlap_changes_barplot(pairwise_changes, molecule):
    """ 
    Plots the pairwise overlap changes for each normal mode
    """
    atom_names = list(molecule.atoms.keys())
    n_modes = len(pairwise_changes)

    # Create a figure with subplots for each normal mode

    if n_modes >= 4:
        ncols = 4
        n_rows = math.ceil(n_modes / ncols)
    else:
        ncols = n_modes
        n_rows = 1

    fig,axes = plt.subplots(n_rows, ncols, figsize=(10*ncols, 5 * n_rows))

    # Flatten axes array for easy iteration
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (mode_id, mode_data) in enumerate(pairwise_changes.items()):
        ax = axes[idx]
        
        # Extract data for this mode
        pairs = list(mode_data['changes'].keys())
        changes = np.abs(list(mode_data['changes'].values())) # Take abs value
        wavenumber = mode_data['wavenumber']

        # Sort pairs by magnitude of changes
        sorted_indices = np.argsort(changes)[::-1]
        sorted_pairs = [pairs[i] for i in sorted_indices]
        sorted_changes = changes[sorted_indices]

        # Create bar plot
        bars = ax.bar(sorted_pairs, sorted_changes, color='skyblue')
        ax.set_title(f"Mode {mode_id} ($\lambda$: {wavenumber:.2f} cm⁻¹)")
        ax.set_xlabel("Atom Pairs")
        ax.set_ylabel("Change in Overlap, |ΔS|")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Rotate 
        ax.tick_params(axis='x', rotation=45) 
        # Highlight max change
        max_change = np.max(changes) if len(changes) > 0 else 1
        for i, change in enumerate(sorted_changes):
            if change > 0.1* max_change:
                bars[i].set_color('salmon')
        
    # Hide any unused subplots
    for ax in axes[n_modes:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("pairwise_overlap_changes.png",bbox_inches='tight')
    plt.close()

def plot_pairwise_overlap_changes_heatmap(pairwise_changes, molecule):
    """ 
    Plots a heatmap of the pairwise overlap changes for each normal mode
    """
    atom_names = list(molecule.atoms.keys())
    n_modes = len(pairwise_changes)

    # Create a figure with subplots for each normal mode
    if n_modes >= 4:
        ncols = 4
        n_rows = math.ceil(n_modes / ncols)
    else:
        ncols = n_modes
        n_rows = 1

    fig, axes = plt.subplots(n_rows, ncols, figsize=(10*ncols, 5 * n_rows))

    # Flatten axes array for easy iteration
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, (mode_id, mode_data) in enumerate(pairwise_changes.items()):
        ax = axes[idx]
        
        # Extract data for this mode
        pairs = list(mode_data['changes'].keys())
        changes = np.abs(list(mode_data['changes'].values())) # Take abs value
        wavenumber = mode_data['wavenumber']

        # Create a matrix for the heatmap
        heatmap_matrix = np.zeros((len(atom_names), len(atom_names)))

        for pair in pairs:
            i, j = pair.split('-')
            i_idx = atom_names.index(i)
            j_idx = atom_names.index(j)
            heatmap_matrix[i_idx, j_idx] = changes[pairs.index(pair)]
            heatmap_matrix[j_idx, i_idx] = changes[pairs.index(pair)]  # Symmetric matrix

        # Mask Lower Triangle
        mask = np.triu(np.ones_like(heatmap_matrix, dtype=bool), k=1)
        heatmap_matrix[mask] = 0  # Set lower triangle to zero

        
        # Plot heatmap
        cax = ax.matshow(heatmap_matrix, cmap='viridis', norm=Normalize(vmin=0, vmax=np.max(changes)))
        ax.set_xticks(range(len(atom_names)))
        ax.set_yticks(range(len(atom_names)))
        ax.set_xticklabels(atom_names, rotation=45)
        ax.set_yticklabels(atom_names)
        
        ax.set_title(f"Mode {mode_id} ($\lambda$: {wavenumber:.2f} cm⁻¹)")
        fig.colorbar(cax, ax=ax, label='Change in Overlap |ΔS|')

    # Hide any unused subplots
    for ax in axes[n_modes:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("pairwise_overlap_changes_heatmap.png", bbox_inches='tight')
    plt.close()

def plot_total_overlap_change(pairwise_changes, molecule):
    """ 
    Function that plots the total change in overlap for each mode after displacement with
    a normal mode vector

    We plot both the total absolute change in overlap |delta S| and just the sum of changes
    """
    modes = list(pairwise_changes.keys())
    total_changes_abs = [np.sum(np.abs(list(pairwise_changes[mode]['changes'].values()))) for mode in modes]
    total_changes = [np.sum(list(pairwise_changes[mode]['changes'].values())) for mode in modes]
    frequencies = [pairwise_changes[mode]["wavenumber"] for mode in modes]
    # Make two subplots one row
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].bar(modes, total_changes_abs, color='skyblue')
    axes[0].set_title("Total Absolute Change in Overlap |ΔS| per Normal Mode")
    axes[0].set_xlabel("Normal Mode ID")
    axes[0].set_ylabel("Total Absolute Change in Overlap |ΔS|")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].tick_params(axis='x', rotation=45) 
    axes[0].set_xticks(modes)
    axes[0].set_xticklabels(modes)
    axes[1].bar(modes, total_changes, color='lightgreen')
    axes[1].set_title("Total Change in Overlap ΔS per Normal Mode")
    axes[1].set_xlabel("Normal Mode ID")
    axes[1].set_ylabel("Total Change in Overlap ΔS")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_xticks(modes)
    axes[1].set_xticklabels(modes)
    plt.tight_layout()
    plt.savefig("total_overlap_change.png", bbox_inches='tight')

    print("======= Total Overlap Change =======")
    for mode in modes:
        print(f"Mode {mode} (Wavenumber: {pairwise_changes[mode]['wavenumber']:.2f} cm⁻¹):")
        print(f"  Total Absolute Change in Overlap |ΔS|: {total_changes_abs[modes.index(mode)]:.4f}")
        print(f"  Total Change in Overlap ΔS: {total_changes[modes.index(mode)]:.4f}")


# ===== Visualization of Pairwiese Gradients =====

def plot_mode_gradients(pairwise_gradients, molecule):
    """ 
    Plots the total magnitude of the gradients for each normal mode
    """
    modes = list(pairwise_gradients.keys())
    frequencies = [pairwise_gradients[mode]["wavenumber"] for mode in modes]
    changes = [pairwise_gradients[mode]["total_gradient_magnitude"] for mode in modes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(modes, changes, color='skyblue')
    ax.set_title("Total Gradient Magnitude per Normal Mode")
    ax.set_xlabel("Normal Mode ID")
    ax.set_ylabel("Total Gradient Magnitude")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("mode_gradients.png", bbox_inches='tight')
    plt.close()


# ======= Tests for Gaussian Overlap =======

def test_gaussian_overlap_identical():
    """ 
    Test the overlap of two identical gaussians
    """
    a1,b1,c1 = 1.0, (0.0, 0.0, 0.0), 1.0
    a2,b2,c2 = 1.0, (0.0, 0.0, 0.0), 1.0

    overlap = gaussian_overlap(a1, b1, c1, a2, b2, c2)


    # Integrate one of the gaussians over the entire space
    def integrand(x, y, z):
        return gaussian_3d(a1, b1, c1, (x, y, z)) * gaussian_3d(a2, b2, c2, (x, y, z))
    
    x_lim = [-5, 5]
    numerical_integral, _ = nquad(integrand, [x_lim, x_lim, x_lim])

    print("======= Test of Identical Gaussian =======")
    print(f"Overlap: {overlap}")
    print(f"Numerical Integral: {numerical_integral}")
    
    assert np.isclose(overlap, numerical_integral), "Overlap does not match numerical integral"

def test_gaussian_overlap_numerical():
    a1,b1,c1 = 1.0, (0.0,0.0,0.0), 1.0
    a2,b2,c2 = 1.0, (0.5, 0.5, 0.5), 1.0

    overlap_analytical = gaussian_overlap(a1, b1, c1, a2, b2, c2)

    def integrand(x, y, z):
        return gaussian_3d(a1, b1, c1, (x, y, z)) * gaussian_3d(a2, b2, c2, (x, y, z))

    x_lim = [-5, 5]
    numerical_integral, _ = nquad(integrand, [x_lim, x_lim, x_lim])

    print("======= Test of Numerical Gaussian Overlap =======")
    print(f"Analytical Overlap: {overlap_analytical}")
    print(f"Numerical Integral: {numerical_integral}")

    assert np.isclose(overlap_analytical, numerical_integral), "Overlap does not match numerical integral" 

def test_gaussian_overlap_1d():
    a1,b1,c1 = 1.0, -1.0, 1.0 
    a2,b2,c2 = 1.0, 1.0, 1.0

    overlap_analytical = gaussian_overlap_1d(a1, b1, c1, a2, b2, c2)

    def integrand(x):
        return gaussian_1d(a1, b1, c1, x) * gaussian_1d(a2, b2, c2, x)
    
    overlap_numerical, _ = quad(integrand, -np.inf, np.inf)

    x = np.linspace(-5, 5, 500)
    g1 = gaussian_1d(a1, b1, c1, x)
    g2 = gaussian_1d(a2, b2, c2, x)
    product = g1 * g2

    plt.figure(figsize=(10, 6))
    plt.plot(x, g1, label=f'Gaussian 1: a={a1}, b={b1}, c={c1}', color='blue')
    plt.plot(x, g2, label=f'Gaussian 2: a={a2}, b={b2}, c={c2}', color='orange')
    plt.plot(x, product, label='Product of Gaussians', linestyle='--', color='red')
    plt.fill_between(x,product, alpha=0.3, color='gray', label='Area under Product')
    plt.title(f"1D Gaussian Overlap \n" f"Analytical Overlap: {overlap_analytical:.4f}, Numerical Integral: {overlap_numerical:.4f}")
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(-5, 5)
    plt.grid()
    plt.show()


# ======= 1D Visualization ===== 

def plot_1d_gaussian_multiplication():
    a1,b1,c1 = 1.0,-1.0, 1.0
    a2,b2,c2 = 1.0,1.0, 1.0

    a_new, b_new, c_new = multiply_gaussians_1d(a1, b1, c1, a2, b2, c2)

    x = np.linspace(-5, 5, 500)

    g1 = gaussian_1d(a1, b1, c1, x)
    g2 = gaussian_1d(a2, b2, c2, x)
    g_new = gaussian_1d(a_new, b_new, c_new, x)
    product = g1 * g2
    
    # Plot 
    plt.figure(figsize=(10, 6))
    plt.plot(x, g1, label=f'Gaussian 1: a={a1}, b={b1}, c={c1}', color='blue')
    plt.plot(x, g2, label=f'Gaussian 2: a={a2}, b={b2}, c={c2}', color='orange')
    plt.plot(x, g_new, label=f'Result with GPT', color='green')
    plt.plot(x, product, label='Product of Gaussians', linestyle='--', color='red')
    plt.title('1D Gaussian Multiplication')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    # Make legends box outside the plot
    plt.legend(loc='upper left')
    plt.xlim(-5, 5)
    plt.grid()
    plt.show()


# ====== Gradient Test with H_1-H_2 Displacement ======

def gradient_test_1d_gaussian_overlap():
    """ 
    Testing function to visualize the gradient of the 1D gaussian overlap with respect to displacment u2
    """
    def overlap_gradient_1d(a1,b1,c1,a2,b2,c2,u2):
        """ 
        Computes the gradient of the 1D gaussian overlap with respect to u2
        """
        S = gaussian_overlap_1d(a1, b1, c1, a2, b2 + u2, c2)
        return S * (b1 - (b2 + u2)) / (c1**2 + c2**2)
    
    a1 = a2 = normalization_parameter()
    b1 = 0.0 # First gaussian fixed at 0
    c1 = c2 = 1.2 # VDW radii of a Hydrogen atom
    b2 = 1.0 # initial position of the second gaussian

    u2_values = np.linspace(-2, 5, 100)  # Displacement values for the second gaussian

    overlaps = []
    gradients = []
    for u2 in u2_values:
        overlaps.append(gaussian_overlap_1d(a1, b1, c1, a2, b2 + u2, c2))
        gradients.append(overlap_gradient_1d(a1, b1, c1, a2, b2, c2, u2))
    
    # Plot the results
    plt.figure(figsize=(12, 6))

    # Overlap plot
    plt.subplot(1, 2, 1)
    plt.plot(u2_values, overlaps, label='Overlap', color='blue')
    plt.title('1D Gaussian Overlap vs Displacement')
    plt.xlabel('Displacement u2')
    plt.ylabel('Overlap')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid()
    plt.legend()

    # Gradient plot
    plt.subplot(1, 2, 2)
    plt.plot(u2_values, gradients, label='Gradient', color='orange')
    plt.title('Gradient of Overlap vs Displacement')
    plt.xlabel('Displacement u2')
    plt.ylabel('Gradient')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

def gradient_test_2d_gradient_field():
    """ 
    Test function visualize the gradient field for x-y displacement
    """
    def overlap_gradient_2d(a1, b1, c1, a2, b2, c2, u2):
        """ 
        Computes the gradient of the 2D gaussian overlap with respect to u2
        """
        b2_displaced = (b2[0] + u2[0], b2[1] + u2[1])
        S = gaussian_overlap_2d(a1, b1, c1, a2, b2_displaced, c2)
        grad_x = S * (b1[0] - b2_displaced[0]) / (c1**2 + c2**2)
        grad_y = S * (b1[1] - b2_displaced[1]) / (c1**2 + c2**2)
        return np.array([grad_x, grad_y])
    
    a1 = a2 = normalization_parameter()
    b1 = np.array([0.0,0.0]) # First gaussian fixed at (0,0)
    c1 = c2 = 1.2 # VDW radii of a Hydrogen
    b2 = np.array([1.0, 0.0]) # initial position of the second gaussian

    # Generate grid of X and Y Displacements
    x = np.linspace(-5, 5, 40)
    y = np.linspace(-5, 5, 40)

    X, Y = np.meshgrid(x, y)

    U = np.zeros(X.shape)
    V = np.zeros(Y.shape)
    for i in range(len(x)):
        for j in range(len(y)):
            u2 = np.array([x[i], y[j]])  # Displacement vector for the second gaussian
            grad = overlap_gradient_2d(a1, b1, c1, a2, b2, c2, u2)
            U[j,i] = grad[0]  # Note the order of indices for U and V
            V[j,i] = grad[1]


    # Plot the gradient field
    plt.figure(figsize=(10, 8))
    plt.quiver(X,Y,U,V, color="red", scale=1.5, width = 0.003, headwidth=3)

    # Mark vdw radii as circles
    circle1 = plt.Circle(b1, c1, color='blue', fill=False, linestyle='--', label='VDW Radius 1')
    circle2 = plt.Circle(b2, c2, color='orange', fill=False, linestyle='--', label='VDW Radius 2')
    plt.gca().add_artist(circle1)
    plt.gca().add_artist(circle2)

    # Mark original positions 
    plt.scatter(b1[0], b1[1], color='blue', label='Gaussian 1 Center', s=100, edgecolor='black')
    plt.scatter(b2[0], b2[1], color='orange', label='Gaussian 2 Center', s=100, edgecolor='black')

    # visualize both the vdw radii as circles

    plt.title('Gradient Field of 2D Gaussian Overlap')
    plt.xlabel('Displacement in X')
    plt.ylabel('Displacement in Y')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()


def gaussian_density_test_2d():
    """
    Function that multiplies two 2D gaussians and visualizes the density
    """
    a1 = a2 = normalization_parameter()
    b1 = np.array([-2, 0.0])  # Center
    c1 = c2 = 1.2  # VDW radii of a Hydrogen atom
    b2 = np.array([2, 0.0])  # Second Gaussian center

    # Generate grid of X and Y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    # Compute the density
    density = gaussian_2d(a1, b1, c1, (X, Y)) * gaussian_2d(a2, b2, c2, (X, Y))

    # Three subplots the Density of First gaussian, density of second gaussian and the product
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Density of first Gaussian
    axes[0].contourf(X, Y, gaussian_2d(a1, b1, c1, (X, Y)), levels=20, cmap='viridis') 
    axes[0].set_title('Density of Gaussian 1')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
    # Add horizontal and vertical lines at 0
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    # Density of second Gaussian
    axes[1].contourf(X, Y, gaussian_2d(a2, b2, c2, (X, Y)), levels=20, cmap='viridis')
    axes[1].set_title('Density of Gaussian 2')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True)
    # Add horizontal and vertical lines at 0
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    # Density of product
    axes[2].contourf(X, Y, density, levels=20, cmap='viridis')
    axes[2].set_title('Density of Product of Gaussians')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True)
    # Add horizontal and vertical lines at 0
    axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[2].axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig("gaussian_density_2d_2_atoms.png", bbox_inches='tight')

def gradient_field_gaussian_density_test_2d():
    """ 
    Function that visualizes the gradient field of the 2D gaussian density for two atoms
    """
    a1 = a2 = normalization_parameter()
    b1 = np.array([-2, 0.0])  # Center of first Gaussian
    b2 = np.array([2, 0.0])   # Center of second Gaussian
    c1 = c2 = 1.2  # VDW radii of a Hydrogen atom

    # Generate grid of X and Y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    # Compute the density for each Gaussian

    def total_density(X, Y):
        """ 
        Computes the total density for two Gaussians
        """
        density1 = gaussian_2d(a1, b1, c1, (X, Y))
        density2 = gaussian_2d(a2, b2, c2, (X, Y))
        return density1 + density2
    
    def density_gradient(X,Y):
        """ 
        Computes the gradient of the total density
        """
        density = total_density(X, Y)
        grad_x = np.gradient(density, axis=1)
        grad_y = np.gradient(density, axis=0)
        return grad_x, grad_y
    
    # One Subplot with the density and one with the gradient field
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Density plot
    density = total_density(X, Y)
    axes[0].contourf(X, Y, density, levels=20, cmap='viridis')
    axes[0].set_title('Total Density of Two Gaussians')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    # Gradient field plot
    grad_x, grad_y = density_gradient(X, Y)
    axes[1].quiver(X, Y, grad_x, grad_y, color='red', scale=0.4, width=0.004, headwidth=1)
    axes[1].set_title('Gradient Field of Total Density')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    # Mark VDW radii as circles
    circle1 = plt.Circle(b1, c1, color='blue', fill=False, linestyle='--', label='VDW Radius 1')
    circle2 = plt.Circle(b2, c2, color='orange', fill=False, linestyle='--', label='VDW Radius 2')
    axes[1].add_artist(circle1)
    axes[1].add_artist(circle2)
    # Mark original positions
    axes[1].scatter(b1[0], b1[1], color='blue', label='Gaussian 1 Center', s=100, edgecolor='black')
    axes[1].scatter(b2[0], b2[1], color='orange', label='Gaussian 2 Center', s=100, edgecolor='black')
    # Add legend
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("gaussian_density_gradient_field_2d.png", bbox_inches='tight') 

def gradient_field_gaussian_density_test_2d_three_atoms():
    """ 
    Plots the gradient field similar to the previous function but for three atoms
    """
    a1 = a2 = a3 = normalization_parameter()
    b1 = np.array([-2, 0.0])  # Center of first Gaussian
    b2 = np.array([0, 0.0])   # Center of second Gaussian
    b3 = np.array([2, 0])   # Center of third Gaussian
    c1 = c2 = c3 = 1.2  # VDW radii of a Hydrogen atom

    # Generate grid of X and Y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    def total_density(X, Y):
        """ 
        Computes the total density for three Gaussians
        """
        density1 = gaussian_2d(a1, b1, c1, (X, Y))
        density2 = gaussian_2d(a2, b2, c2, (X, Y))
        density3 = gaussian_2d(a3, b3, c3, (X, Y))
        return density1 + density2 + density3
    
    def density_gradient(X,Y):
        """ 
        Computes the gradient of the total density
        """
        density = total_density(X, Y)
        grad_x = np.gradient(density, axis=1)
        grad_y = np.gradient(density, axis=0)
        return grad_x, grad_y
    
    # One Subplot with the density and one with the gradient field
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # Density plot
    density = total_density(X, Y)
    axes[0].contourf(X, Y, density, levels=20, cmap='viridis')
    axes[0].set_title('Total Density of Three Gaussians')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    # Gradient field plot
    grad_x, grad_y = density_gradient(X, Y)
    axes[1].quiver(X, Y, grad_x, grad_y, color='red', scale=0.4, width=0.004, headwidth=1)
    axes[1].set_title('Gradient Field of Total Density')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)    
    # Mark VDW radii as circles
    circle1 = plt.Circle(b1, c1, color='blue', fill=False, linestyle='--', label='VDW Radius 1')
    circle2 = plt.Circle(b2, c2, color='orange', fill=False, linestyle='--', label='VDW Radius 2')
    circle3 = plt.Circle(b3, c3, color='green', fill=False, linestyle='--', label='VDW Radius 3')
    axes[1].add_artist(circle1)
    axes[1].add_artist(circle2)
    axes[1].add_artist(circle3)
    # Mark original positions
    axes[1].scatter(b1[0], b1[1], color='blue', label='Gaussian 1 Center', s=100, edgecolor='black')
    axes[1].scatter(b2[0], b2[1], color='orange', label='Gaussian 2 Center', s=100, edgecolor='black')
    axes[1].scatter(b3[0], b3[1], color='green', label='Gaussian 3 Center', s=100, edgecolor='black')
    # Add legend
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("gaussian_density_gradient_field_2d_three_atoms.png", bbox_inches='tight')

def gaussian_density_test_2d_three_atoms():
    """
    Plots the Gaussian Density of three atoms in 2D
    """
    a1 = a2 = a3 = normalization_parameter()
    b1 = np.array([-2, 0.0])  # Center of first Gaussian
    b2 = np.array([0, 0.0])   # Center of second Gaussian
    b3 = np.array([2, 0])   # Center of third Gaussian
    c1 = c2 = c3 = 1.2  # VDW radii of a Hydrogen atom

    # Generate grid of X and Y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Compute the density for each Gaussian
    density1 = gaussian_2d(a1, b1, c1, (X, Y))
    density2 = gaussian_2d(a2, b2, c2, (X, Y))
    density3 = gaussian_2d(a3, b3, c3, (X, Y))

    # Total density is the sum of individual densities
    total_density = density1 + density2 + density3

    # Plotting the densities
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].contourf(X, Y, density1, levels=20, cmap='viridis')
    axes[0].set_title('Density of Gaussian 1')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True)
    
    axes[1].contourf(X, Y, density2, levels=20, cmap='viridis')
    axes[1].set_title('Density of Gaussian 2')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True)

    axes[2].contourf(X, Y, density3, levels=20, cmap='viridis')
    axes[2].set_title('Density of Gaussian 3')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True)

    axes[3].contourf(X, Y, total_density, levels=20, cmap='viridis')
    axes[3].set_title('Total Density of Three Gaussians')
    axes[3].set_xlabel('X')
    axes[3].set_ylabel('Y')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("gaussian_density_2d_three_atoms.png", bbox_inches='tight')



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

    # Normalize the density
    density /= np.max(density)  # Normalize to the maximum value
    
   
    return density 

def plot_isosurface(X,Y,Z, density, molecule, vdw_radii):
   """ 
   Function to visualize the vdw density using Pyvista, further the function plots 2D slices
   """

   
   grid = pv.ImageData()
   # Set grid dimensions number of points in each dimension
   grid.dimensions = np.array(density.shape)   
   dx = X[1,0,0] - X[0,0,0]
   dy = Y[0,1,0] - Y[0,0,0]
   dz = Z[0,0,1] - Z[0,0,0]
   grid.spacing = (dx, dy, dz)
   grid.origin = (X[0,0,0], Y[0,0,0], Z[0,0,0])
   grid.point_data['density'] = density.flatten(order='F')  # Flatten the density array for PyVista

   min_val = density.min()
   max_val = density.max() 

   atom_colors = molecule.atom_colors
   atom_radii = {
        "H": 0.25, "C": 0.35, "N": 0.4, "O": 0.4,
        "F": 0.35, "Cl": 0.4, "Br": 0.45, "I": 0.5
    }
   iso_val = np.percentile(density, 95)  # Adjust this value to control the isosurface threshold 
   contour = grid.contour([iso_val])
   

   p = pv.Plotter()
   p.add_mesh(contour, color='lightblue', opacity=0.5)
   p.add_volume(grid, scalars='density', cmap='viridis', opacity=0.03, shade=True, show_scalar_bar=True,
                scalar_bar_args={
                    "title": 'Density',
                    "vertical": True,
                    "title_font_size": 20,
                    "label_font_size":16,
                    "n_labels":5,
                    "color":"black"
                }
                )

   for atom, coords in molecule.atoms.items():
       color = atom_colors.get(atom[0], "gray")
       radius = atom_radii.get(atom, 0.3)
       sphere = pv.Sphere(center=coords, radius=radius)
       p.add_mesh(sphere, color=color, show_edges=True, name=atom)
       
   p.add_axes()
   p.show()

   # Now we plot 2D slices of the density

   fig, axes = plt.subplots(1, 3  , figsize=(15, 5))

   # Calculate slice indices, middle of each dimension
   x_slice = X.shape[0] // 2
   y_slice = Y.shape[1] // 2
   z_slice = Z.shape[2] // 2

   # XY slice
   im1 = axes[0].contourf(X[:, :, z_slice], Y[:, :, z_slice], density[:, :, z_slice], levels=20, cmap='viridis')
   axes[0].set_title('XY Slice')
   axes[0].set_xlabel('X')
   axes[0].set_ylabel('Y')
   fig.colorbar(im1, ax=axes[0], label='Density')

   # XZ slice
   im2 = axes[1].contourf(X[:, y_slice, :], Z[:, y_slice, :], density[:, y_slice, :], levels=20, cmap='viridis')
   axes[1].set_title('XZ Slice')
   axes[1].set_xlabel('X')
   axes[1].set_ylabel('Z')
   fig.colorbar(im2, ax=axes[1], label='Density')

   # YZ Slice
   im3 = axes[2].contourf(Y[x_slice, :, :], Z[x_slice, :, :], density[x_slice, :, :], levels=20, cmap='viridis')
   axes[2].set_title('YZ Slice')
   axes[2].set_xlabel('Y')
   axes[2].set_ylabel('Z')
   fig.colorbar(im3, ax=axes[2], label='Density')

   plt.tight_layout()
   plt.show()


def hermite_gauss(x,b_1,c_1):
    """
    Implement a first exited state aka Hermite Function H1 Multiplied with Gaussian
    """
    return (x-b_1) * np.exp(-(x-b_1)**2 / (2*c_1**2))

def hermite_gauss_2d(x,b_1,c_i):
    """ 
    X is now 2d vector and b_1 = (b1_x, b1_y)
    """
    return (x[0]-b_1[0]) *(x[1]-b_1[1]) * np.exp(-((x[0]-b_1[0])**2 + (x[1]-b_1[1])**2) / (2*c_i**2))

def test_hermite_gauss_2d():
    b_1 = (0.0, 0.0)
    c_1 = 1.0
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.array([X, Y])
    Z = hermite_gauss_2d(pos, b_1, c_1)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('2D Hermite-Gauss Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    plt.grid()
    plt.show()

def test_hermite_gauss():

    b_1 = 0.0
    c_1 = 1.0
    x = np.linspace(-5, 5, 500)
    y = hermite_gauss(x, b_1, c_1)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Hermite-Gauss: b={b_1}, c={c_1}', color='blue')
    plt.title('1D Hermite-Gauss Function')
    plt.xlabel('x')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.xlim(-5, 5)
    plt.grid()
    plt.show()

