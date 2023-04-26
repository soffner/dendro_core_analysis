import numpy as np
import pandas as pd
from get_properties_v2 import *
from leaf_history_functions import *
import gc
import glob
import matplotlib
matplotlib.use('Agg')

# Location of STARFORGE outputs
dir  = '/scratch/05917/tg852163/GMC_sim/Runs/Physics_ladder/M2e4_C_M_J_RT_W_R30_v1.1_2e7/output/' 

# Run name
run = 'M2e4_C_M_J_RT_W_R30_v1.1_2e7'
outdir = './'

res_limit = 1e-3  # Minimum resolution

fns = glob.glob(dir+'snapshot_*[0,1,2,3,4,5,6,7,8,9].hdf5')
fns.sort()

create_leaf_history_fast(fns,run,res_limit=res_limit,search_radius=1.0)

print("Completed history!")
