# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:13:04 2024

@author: mac732
"""

import pandas as pd
import benchmark_functions as bf
import csv

func = bf.Michalewicz(n_dimensions=5)
point = pd.read_csv("dv.dat").values.reshape(-1).tolist()

val = [["x"], [func(point)], [0], [0]]

with open('output.dat', 'w+', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(val)