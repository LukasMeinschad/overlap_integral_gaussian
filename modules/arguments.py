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
    return parser.parse_args()