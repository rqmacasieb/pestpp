# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:13:04 2024

@author: mac732
"""

import pandas as pd
import benchmark_functions as bf
import csv

func = bf.Michalewicz(n_dimensions=50)
point = pd.read_csv("dv.dat").values.reshape(-1).tolist()

val = [[func(point)], [0], [0]]

with open('hf_output.dat', 'w+', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(val)