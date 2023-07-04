#!/bin/bash

# Starts a HAT environment. It is necessary to install hat first
# 
# README
# https://github.com/ecmwf-projects/hat/tree/dev

# Start HAT environment
echo ''
echo '🌊 STARTING HAT 🌊'
echo ''
echo 'HAT (Hydrological Analysis Tools)'

# HPC detected
if [[ "$OSTYPE" == "linux-gnu"* ]] && [[ -n "$ECPLATFORM" ]]; then
  echo 'HPC detected'
  echo 'Loading conda module..'
  module reset 2>/dev/null
  module load conda 2>/dev/null
fi

echo 'Loading conda environment...'
conda activate hat

echo 'Moving to code repo...'
echo $HAT_CODE_DIR
cd $HAT_CODE_DIR

echo 'Loading main branch..'
git checkout main --quiet

echo 'Pulling latest code from github..'
git pull #<- user needs ssh access to hat code repo for this to work.. 

# WARNING this uses DEPRECATED setup.py (!!!)
echo 'Installing as pip package..' 
pip install --upgrade --quiet --use-pep517 --disable-pip-version-check $HAT_CODE_DIR

echo 'Returning to previous directory'
cd -

# When in HAT environment search for HAT functions first
export PATH="$CONDA_PREFIX/bin:$PATH"

echo ''
echo '✨ HAT activated ✨'
echo ''
