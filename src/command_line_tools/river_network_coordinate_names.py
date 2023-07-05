"""
Print table of river network name to coorindate name
"""
import typer

from hat.cli import prettyprint, title
from hat.config import load_package_config


def river_network_coordinate_names():
    """Station locations need to be adjusted for each 'river network'.
    This is currently done using a latlon pair per river network"""

    coord_name = load_package_config("river_network_coordinate_names.json")

    # table of station coordinates
    print()
    prettyprint(f"{'River Network to Coordinate Names'.center(60)}", color="yellow")
    title(f"{'River Network':20}{'X Coordinate':20}{'Y Coordinate':20}", color="cyan")
    for key in coord_name:
        print(f"{key:20}{coord_name[key][0]:20}{coord_name[key][1]:20}")
    print()


def main():
    typer.run(river_network_coordinate_names)


if __name__ == "__main__":
    main()
