import os
import numpy as np
import pandas as pd
def zdt1(x):
    g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
    return (x[0], g * (1 - np.sqrt(x[0] / g))),[]

def helper(func):
    pdf = pd.read_csv("dv.dat",delim_whitespace=True,index_col=0, header=None, names=["parnme","parval1"])
    #obj1,obj2 = func(pdf.values)
    objs,constrs = func(pdf.values)
    
    if os.path.exists("additive_par.dat"):
        obj1,obj2 = objs[0],objs[1]
        cdf = pd.read_csv("additive_par.dat", delim_whitespace=True, index_col=0,header=None, names=["parnme","parval1"])
        obj1[0] += cdf.parval1.values[0]
        obj2[0] += cdf.parval1.values[1]
        for i in range(len(constrs)):
            constrs[i] += cdf.parval1.values[i+2]

    with open("obj.dat",'w') as f:
        for i,obj in enumerate(objs):
            f.write("obj_{0} {1}\n".format(i+1,float(obj)))
        #f.write("obj_2 {0}\n".format(float(obj2)))
        for i,constr in enumerate(constrs):
            f.write("constr_{0} {1}\n".format(i+1,float(constr)))
    return objs,constrs

if __name__ == '__main__':
    helper(zdt1)
    np.random.seed(1111)
    with open("obj_link.dat",'w') as f:
        f.write("obj_1 {0}\n".format(np.random.random()))
        f.write("obj_2 {0}\n".format(np.random.random()))


