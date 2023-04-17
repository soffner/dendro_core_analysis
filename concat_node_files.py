import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
from natsort import natsorted 

tag = 'M2e4_C_M_J_RT_W_R30_v1.1_2e7'
out_nodefile = 'nodes_edges_'+tag+'_all.csv'

files = 'nodes_edges_'+tag+'*.csv'
fns = glob.glob(files)
fns = natsorted(fns) #fns.sort() # Need natsort for snapshot > 1000, otherwise do time sort

allnodes = []
for fn in fns:
    print("reading ", fn)
    try:
        data = pd.read_csv(fn, converters = {0: str, 1: str}, header=None)
        allnodes.append(data)
    except pd.errors.EmptyDataError:
        print(fn, " is empty")


nodes_edges_df = pd.DataFrame(pd.concat(allnodes))
nodes_edges_df.to_csv(out_nodefile, index=False, header=False)
print("Saved nodes to ", out_nodefile)
