#!/bin/bash
cd ./flownet_networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
python setup.py install

cd ..