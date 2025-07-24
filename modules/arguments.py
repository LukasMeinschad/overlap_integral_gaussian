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
        choices = ["plot_molecule","plot_density"],
        help="""
             Specify the plotting options 
             
             + If plot_molecule is specified, a 3D plot of the molecule will be generated
             + If plot_density is specified, a 3D plot of the density will be generated
             """


    )
    return parser.parse_args()