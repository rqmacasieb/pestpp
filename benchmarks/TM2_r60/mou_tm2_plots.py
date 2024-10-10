import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

run_data = pd.read_csv('mou_tm2.pareto.archive.summary.csv')
#pareto = run_data.loc[(run_data['generation']==5)]
np_arr = run_data.values

x1=np_arr[:,2]
y1=np_arr[:,3]

fig_1=plt.figure(figsize=(9,6))
axes_1 = fig_1.add_axes([0.08,0.1,0.9,0.88])
axes_1.set_xlabel('Total Pumping (maximize)')
axes_1.set_ylabel('Sum of Squared Drawdown (minimize)')
axes_1.scatter(x1,y1, marker='o', c = 'g', edgecolors = 'none'  , alpha=0.6)

fig_path = './plots/'

if not os.path.isdir(fig_path):
    os.mkdir(fig_path)

fig_1.savefig(f"{fig_path}/TM2_fig1.png")