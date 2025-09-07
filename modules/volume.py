import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from molecule import Molecule


def union_volume_mc_molecule(molecule,radii,n_samples=1_000_000):
    """ 
    Computes the total volume of a arbitrary number of spheres using
    Monte-Carlo Integration
    """
    print(f"Computing the union volume of {molecule.num_atoms} spheres via Monte Carlo Integration")
    print(f"Molecule: {molecule}" )
    print(f"Radii: {radii}" )
    atoms = list(molecule.atoms.values())
    centers = np.array(atoms) 
    radii = np.array([radii[atom] for atom in molecule.atoms.keys()]) / 100 # Convert Angstrom to nm
    
    # Find the bounding box
    mins = centers.min(axis=0) - radii.max()
    maxs = centers.max(axis=0) + radii.max()
    vol_box = np.prod(maxs-mins)
    print(f"Bounding box volume: {vol_box:.3f} nm^3")

    # Sample points
    points = np.random.uniform(mins,maxs, (n_samples, 3))

    # Now we implement a check if the point is inside any of the spheres
    inside_any = np.zeros(n_samples, dtype=bool)
    for c,r in zip(centers, radii):
        inside_any |= np.linalg.norm(points - c, axis=1) <= r
    
    volume_estimate = vol_box * inside_any.mean()
    print(f"Estimated union volume: {volume_estimate:.3f} Angstrom^3")
    return volume_estimate

def check_volume_convergence(molecule,radii,max_samples=5_000_000): 
    """ 
    Checks the convergence of the Integral in respect to the number of sample points used    
    """ 
    Volumes = []
    for n in [10,100,1_000,10_000,100_000,1_000_000,5_000_000]:
        if n > max_samples:
            break
        vol = union_volume_mc_molecule(molecule,radii,n_samples=n)
        Volumes.append(vol)
    plt.plot([10,100,1_000,10_000,100_000,1_000_000,5_000_000][:len(Volumes)], Volumes, marker='o')
    plt.xscale('log')
    plt.xlabel('Number of Sample Points')
    plt.ylabel('Estimated Union Volume (Angstrom^3)')
    plt.title('Convergence of Union Volume Estimate')
    plt.grid(True, which="both", ls="--")
    plt.show()




def union_volume_mc(r1,r2,d,n_samples=1_000_000):
    """ 
    Computes the total volume of two spheres via Monte Carlo Integration
    """
    c1 = np.array([0,0,0])
    c2 = np.array([d,0,0])
    R = max(r1,r2)
    # Generate A grid box
    box_min = -R
    box_max = d+R
    vol_box = (box_max-box_min)**3

    # Generate random points inside of the box
    points = np.random.uniform(box_min,box_max,(n_samples, 3))
    in_s1 = np.linalg.norm(points - c1, axis=1) <= r1
    in_s2 = np.linalg.norm(points - c2, axis=1) <= r2
    
    union = np.sum(in_s1 | in_s2)

    plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    ax.set_xlim([box_min, box_max])
    ax.set_ylim([box_min, box_max])
    ax.set_zlim([box_min, box_max])
    ax.scatter(points[in_s1,0], points[in_s1,1], points[in_s1,2], color='blue', alpha=0.1, label='Inside Sphere 1')
    ax.scatter(points[in_s2,0], points[in_s2,1], points[in_s2,2], color='red', alpha=0.1, label='Inside Sphere 2')
    ax.scatter(points[~(in_s1 | in_s2),0], points[~(in_s1 | in_s2),1], points[~(in_s1 | in_s2),2], color='gray', alpha=0.01, label='Outside Both Spheres')
    ax.legend()
    plt.title('Monte Carlo Integration of Sphere Union Volume')
    plt.show()

    return vol_box * union / n_samples

def comparison_mc_and_analytical():
    """ 
    Comparison of Monte-Carlo Integration and Analytical Solution in evaluating
    the integral of a 3D sphere
    """
    c = np.array([0,0,0])
    r = 1
    # Generate a box
    box_min = -r
    box_max = r
    vol_box = (box_max-box_min)**3

    n_random_points = [10,100,1000,10_000,100_000,1_000_000]
    mc_results = []
    for n in n_random_points:
        points = np.random.uniform(box_min,box_max,(n, 3))
        inside_sphere = np.linalg.norm(points - c, axis=1) <= r
        n_inside = np.sum(inside_sphere)
        mc_volume = vol_box * n_inside / n
        mc_results.append(mc_volume)
    
    analytical_volume = (4/3) * np.pi * r**3
    errors = [abs(mc - analytical_volume) for mc in mc_results]
    plt.figure(figsize=(10,6))
    plt.loglog(n_random_points, errors, marker='o')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Number of Random Points')
    plt.ylabel('Absolute Error')
    plt.title('Convergence of Monte Carlo Integration for Sphere Volume')
    plt.show()
