### Installation

Clone source code repository

    git clone git@github.com:ecmwf-projects/hat.git

Create conda python environment

    cd hat

    # If on HPC..
    # module load conda

    conda env create -f environment.yml
    conda activate hat

Install HAT software

    pip install .

Open a new terminal to start