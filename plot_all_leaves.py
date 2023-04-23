import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
from astrodendro import Dendrogram
from leaf_history_functions import *
from natsort import natsorted

run = 'M2e4_C_M_J_RT_W_R30_v1.1_2e7'

# Read in Data From Folder of csv files
vers = 'v1'
tag = run+'_all_prop_'+vers+'.csv'
allprop_file = run+'_all_prop_'+vers+'.csv'
dendro_file_names = run+'*res1e-3.fits'

dir  = '/scratch/05917/tg852163/GMC_sim/Runs/Physics_ladder/M2e4_C_M_J_RT_W_R30_v1.1_2e7/output/'

converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
data = pd.read_csv(tag)

# Get Property List
converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
fns = glob.glob(allprop_file)
fns.sort()
frames = []
for fn in fns:
    data = pd.read_csv(fn, 
                       converters = { 'ID': str, 
                           'Radii': converter_nparray,
                                     'Density [cm^-3]': converter_nparray,
                                    'Dispersion [cm/s]': converter_nparray,
                                    'V Bulk [cm/s]': converter_nparray,
                                    'Center Position [pc]':converter_nparray,
                                    'Shape [pc]':converter_nparray,
                                     'Mean B [G]':converter_nparray
                                    })
    frames.append(data)
    
props = pd.concat(frames)

IDstr = props['ID'].values # ID in padded string form

IDnum = []  #Get IDs in numeric form
for i in IDstr:
    IDnum.append(int(i))


# Make a plot of all the cores
dendro_files = glob.glob(dendro_file_names)
dendro_files = natsorted(dendro_files)

for dendro_file in dendro_files[256:]:
    snapshot=dendro_file[-27:-24]
    print("Reading ", snapshot, dendro_file)
    dendro = Dendrogram.load_from(dendro_file)
    gizmofile = dir+"snapshot_"+snapshot+".hdf5"
    print("Reading ", gizmofile)
    x,den = load_pos_den(gizmofile)
    starpos,starmass = load_star_pos_mass(gizmofile)

    # Find the leaf in the dendro file    
    #all_leaf_ids = [leaf.idx for leaf in dendro.leaves]
    #print(all_leaf_ids) 
    #loc = np.where(np.array(all_leaf_ids) == leafid)[0]
    #print("leaf id, loc", leafid, loc)
    #leaf = (dendro.leaves)[loc[0]]
   
    fig,ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], alpha=0.1,s=1, color='black') 
    if len(starpos) > 0:
        ax.scatter(starpos[:,0],starpos[:,1], alpha=1.0, s=starmass*100, color='red', marker="*")
    for leaf in dendro.leaves:
        mask = leaf.get_mask()
        ax.scatter(x[mask,0], x[mask,1], alpha=0.5,s=1)
                
        #for j,star in enumerate(starmass):
        #    ax.annotate(str(star), (starpos[j,0],starpos[j,1]+0.01), fontsize=20) #, xycoords=data)
        centeridx = leaf.get_peak()[0]
        center=x[centeridx]
    
        ax.scatter(center[0],center[1], s=10, color="green", marker="+")

    plt.xlim([148,154])
    plt.ylim([143,149])
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    fig.savefig('Snapshot_'+snapshot+'_leaves.png', dpi=100)    
    
                                      
