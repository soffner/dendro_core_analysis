import numpy as np
import pandas as pd
from get_properties_v2 import *
from leaf_history_functions import *
import gc
import glob
import matplotlib
matplotlib.use('Agg')

dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'
outdir = './'

res_limit = 1e-3  # Minimum resolution

fns = glob.glob(dir+'snapshot_*[0,1,2,3,4,5,6,7,8,9].hdf5')
fns.sort()

leaf_history, tree_history, merge_history, nodes_edges = create_leaf_history(fns,run,res_limit=res_limit)

# Save all the histories
leaf_history_df = pd.DataFrame(leaf_history)
leaf_history_df.to_csv('leaf_history_'+run+'_tmp.csv', index=False, header=False)

tree_history_df = pd.DataFrame(tree_history)
tree_history_df.to_csv('tree_history_'+run+'_tmp.csv', index=False, header=False)

merge_history_df = pd.DataFrame(merge_history)
merge_history_df.to_csv('merge_history_'+run+'_tmp.csv', index=False, header=False)

nodes_edges_df = pd.DataFrame(nodes_edges)
nodes_edges_df.to_csv('nodes_edges_'+run+'_tmp.csv', index=False, header=False)
