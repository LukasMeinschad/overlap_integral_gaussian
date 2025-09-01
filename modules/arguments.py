from argparse import ArgumentParser
from pathlib import Path


parser = ArgumentParser()


def get_args():
    """ 
    Parse Command Line Arguments
    """

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the molpro output file containing the atomic coordinates.",
    )

    parser.add_argument(
        "-p",
        "--plot",
        type=str,
        choices = ["plot_molecule","plot_density",'plot_molecules_vdw_spheres'],
        help="""
             Specify the plotting options 
             
             + If plot_molecule is specified, a 3D plot of the molecule will be generated
             + If plot_density is specified, a 3D plot of the density will be generated
             """


    )
    parser.add_argument(
        "-t",
        "--tests",
        type=str,
        choices=["1d_gaussian_mult", "gaussian_overlap", "gradient_test"],
        help="""
            Includes some testing functions just to check if the code is working as expected

            + 1d_gaussian_mult: Tests the multiplication of two 1D gaussian functions
            + gaussian_overlap: Performs various test calculations of the gaussian overlap
            + gradient_test: Various test for the gradient of the gaussian overlap
            """
    )
    parser.add_argument(
        "-v",
        "--vector",
        type=bool,
        default=False,
        help=""" 
            If set to True, additionaly the vector length of the normal modes and the position of the atoms in 3D space will be written to a log file

            + Default is False
            """
    )


    return parser.parse_args()