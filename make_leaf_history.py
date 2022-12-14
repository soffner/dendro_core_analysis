import numpy as np
import pandas as pd
from get_properties_v2 import *
from leaf_history_functions import *
import gc
import glob
import matplotlib
matplotlib.use('Agg')

#file = 'M2e3_mid'
#snap = '400' # Temporary file number

dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'
outdir = './'

fns = glob.glob(dir+'snapshot_*[0,2,4,6,8].hdf5')
fns.sort()

leaf_history = create_leaf_history(fns,run)
print(leaf_history)
leaf_history_df = pd.DataFrame(leaf_history)
leaf_history_df.to_csv('leaf_history_'+run+'_v3.csv', index=False, header=False)

plot_leaves(fns, run)

