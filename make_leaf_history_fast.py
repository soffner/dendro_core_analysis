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

fns = glob.glob(dir+'snapshot_*0.hdf5') #[0,1,2,3,4,5,6,7,8,9].hdf5')
fns.sort()

create_leaf_history_fast(fns,run,res_limit=res_limit)

print("Completed history!")
