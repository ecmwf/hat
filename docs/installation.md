### Installation

Clone source code repository

    git clone https://github.com/ecmwf/hat.git
    cd hat

Create conda python environment

    # If on HPC..
    # module load conda
    conda create -n hat python=3.10
    conda activate hat

Installation of required dependencies

    pip install .
