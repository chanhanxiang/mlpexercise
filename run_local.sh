#!/usr/bin/env sh

#Note 1: Uncomment rm -rf to remove local directory
#Note 2: cd to direct local path to make local virtualenv and also to retrieve requirements.txt - change the cd as required. If cd path is long, enclose in commas
#rm -rf localaiap

cd 'C:\Users\Han\Documents\AIAP\AIAP 11\Assessment 2\aiap11-chan-han-xiang-496z\aiap11-chan-han-xiang-496z'
mkdir localaiap

pip install jupyterlab

pip install --upgrade virtualenv

python -m virtualenv localaiap

cd 'C:\Users\Han\Documents\AIAP\AIAP 11\Assessment 2\aiap11-chan-han-xiang-496z\aiap11-chan-han-xiang-496z'
# Install libraries in requirements.txt
pip install -r requirements.txt

source localaiap/Scripts/activate

pip install ipykernel

python -m ipykernel install --name=localaiap

jupyter-notebook eda.ipynb

