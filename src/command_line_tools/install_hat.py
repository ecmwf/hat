"""
Install 'hat' function in bash to

1. load HPC modules
2. activate python environment
3. pull latest source code
4. upgrade hat pip installation

"""

from hat.cli import prettyprint, title
import os
import sys

import typer

# hat modules
here = os.path.dirname(__file__)
src = os.path.dirname(here)
sys.path.append(src)


def text_block_for_bash_profile(header, footer):
    """This text block is for bash_profile.
    It creates hat function and environment variables"""

    # filepaths
    hat_code_dir = os.path.dirname(os.path.dirname(here))
    startup_script = f"{hat_code_dir}/src/bash/start-hat.sh"

    # body of text
    body = f"""
# Contents in this block were created using {os.path.abspath(__file__)}
=
# hat environment variables
export HAT_CODE_DIR={hat_code_dir}
export HAT_STARTUP_SCRIPT={startup_script}

# hat function
function hat() {{
    source $HAT_STARTUP_SCRIPT
}}
"""

    return "\n".join([header, body, footer])


def update_bash_profile(force=False):
    """find and update the bash profile
    (creates one if one doesn't already exist)"""

    header = "# 🌊 Hydrological Analysis Tool 🌊"
    footer = "# ~~~ hat config end ~~~"
    fpath = os.path.expanduser("~/.bash_profile")

    # User permission required to update bash profile
    if not force:
        prettyprint(" User input required ⬇ ",
                    color="black",
                    background="cyan")
        permission = input("Update .bash_profile (y/n)? ")
        if not permission.lower() in ["y", "yes"]:
            print("Please type 'y' to update .bash_profile")
            return

    # Create a .bash_profile if one does not exist
    if not os.path.exists(fpath):
        with open(fpath, "w") as f:
            f.write("\n".join([header, "temporary body", footer]))

    # Read in the existing bash_profile
    with open(fpath, "r") as f:
        content = f.read()

    # Find the start and end of the old block
    start_index = content.find(header)
    end_index = content.find(footer)

    # If the block is found, remove it
    if start_index != -1 and end_index != -1:
        content = content[:start_index] + content[end_index + len(footer):]

    # Add the text block to the end of the content
    content += text_block_for_bash_profile(header, footer)

    # Write the updated content back to bash_profile
    with open(fpath, "w") as f:
        f.write(content)

    return True


def install(force: bool = False):
    """Installs hat (bash function) which runs start-hat.sh
    - Detect HPC
    - Load conda modules
    - Activate conda environment
    - Pull latest code from main branch
    - Installs hat as pip package from source
      (i.e. for command line tools and setup.py)
    """
    title("Hydrological Analysis Tool (HAT) Setup")

    success = update_bash_profile(force)

    if success:
        prettyprint("HAT setup complete", last_line_empty=True)
        prettyprint(
            "Open a new terminal for changes to effect ➡️  ",
            color="black",
            background="yellow",
            last_line_empty=True,
        )


def main():
    typer.run(install)


if __name__ == "__main__":
    main()
