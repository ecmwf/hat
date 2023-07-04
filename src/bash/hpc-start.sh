# Starts a HAT environment on the HPC. It is necessary that hat is installed first
# 
# README
# https://github.com/ecmwf-projects/hat/tree/dev


#!/bin/bash

# # HPC required
# if [[ "$OSTYPE" != "linux-gnu"* ]] || [[ -z "$ECPLATFORM" ]]; then
#   echo 'HPC not detected'
#   echo 'returning..'
#   return
# fi

# HPC detected
if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ -n "$ECPLATFORM" ]]; then
  echo 'HPC detected'
  echo 'Loading conda module..'
  module reset 2>/dev/null
  module load conda 2>/dev/null
fi

# Start HAT environment
echo ''
echo '🌊 STARTING HAT 🌊'
echo ''
echo 'HAT (Hydrological Analysis Tools)'

echo 'Loading conda environment...'
conda activate hat

echo 'Moving to command_line_tools directory..'
cd $HAT_CODE_DIR/src/command_line_tools

echo 'Loading dev branch..'
git checkout dev --quiet

# echo 'Pulling latest code from github..'
# git pull <- user needs ssh access to hat code repo for this to work.. 

# WARNING this uses DEPRECATED setup.py (!!!)
echo 'Installing as pip package..' 
pip install --upgrade --quiet --use-pep517 --disable-pip-version-check $HAT_CODE_DIR

# Update path to find HAT functions first
export PATH="$CONDA_PREFIX/bin:$PATH"

echo ''
echo '✨ HAT activated ✨'
echo''
