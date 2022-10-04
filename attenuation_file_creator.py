#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:31:59 2022

@author: ivan

https://www.nist.gov/pml/xcom-photon-cross-sections-database
Coherent scattering, incoherent scattering, photoelectric absorption, nuclear pair production
Delimiter |
Ctrl + L and Ctrl + C
"""

import os
import pandas as pd

"""
base_path = r"/home/ivan/Documents/Monte-Carlo-Gamma-Ray-Spectroscopy-Simulator"
file = r"test_file.csv"
result_filename = r"test_file_2.csv"
"""

base_path = input("Path to folder: ")
file = input("File to convert (example.csv): ")
result_filename = input("Name of converted file (example.csv): ")

path = os.path.join(base_path, file)
result_path = os.path.join(base_path, result_filename)

column_names = ["energy", 
                "coh scatt", 
                "incoh scatt", 
                "photo abs",
                "pair prod"]

data = pd.read_csv(path, delimiter="|", header=1, names=column_names, index_col=False)
data.to_csv(result_path)

finished = input("File succesfully converted! Press enter to exit.")