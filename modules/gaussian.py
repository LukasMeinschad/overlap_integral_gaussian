"""
Modeling density of molecule using gaussian functions
"""

from molecule import Molecule
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from matplotlib.colors import Normalize
from scipy.integrate import nquad
from scipy.integrate import quad

def gaussian_1d(a=1, b=0, c=1, x=0):
    """ 
    Computes a 1D gaussian function
    """
    exponent = - ((x - b) ** 2) / (2 * c ** 2)
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


def gaussian_overlap(a1,b1,c1,a2,b2,c2):
    """
    Implementation of the overlap formula between the two gaussians
    """
    c_sq = (c1**2 * c2**2) /(c1**2 + c2**2)
    exponent = -np.sum((np.array(b1) - np.array(b2))**2 / (2*(c1**2 + c2**2)))
    prefactor = a1*a2*(2*np.pi)**(3/2) *c_sq**(3/2)
    return prefactor*np.exp(exponent)



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
    
    a1 = a2 = 1.0
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