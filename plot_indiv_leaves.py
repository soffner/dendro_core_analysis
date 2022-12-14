import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
from astrodendro import Dendrogram
from leaf_history_functions import *

tag = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_data_orphansink.csv'
dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'

converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
data = pd.read_csv(tag)

# Read in Josh's list of sinks and their cluster id
IDsink = data['ID'].values
orphansink = data['OrphanSink'].values #Will be TRUE when sink is 'out of place'
clusid = data['Uclus'].values # Assigned cluster values

# Get Property List
converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
files = '_M2e3_cores_v3/M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_snapshot_*0_prop_v3.csv'
fns = glob.glob(files)
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
                                    'Eigenvals [pc]':converter_nparray,
                                     'Mean B [G]':converter_nparray
                                    })

    frames.append(data)
    
props = pd.concat(frames)

IDstr = props['ID'].values # ID in padded string form

IDnum = []  #Get IDs in numeric form
for i in IDstr:
    IDnum.append(int(i))

orphans = np.where(np.array(orphansink) == True)[0]

# Check that indicies match (they do)
#for i in orphans:
#    print(IDsink[i], IDnum[i], IDstr[i])

#print("matches = ",np.array(IDnum)[orphans], np.array(IDsink)[orphans])

# Make a plot of the core
lastsnap = []
notleaves = []
for i in orphans:
    id = IDsink[i]
    ind = np.where(IDnum == id)[0]
    print("id=", id, " IDstr =", IDstr[ind])
    snapshot = (IDstr[ind])[0][2:5]
    snap = int(snapshot)
    leafid = int((IDstr[ind])[0][5:])
    
    if snap != lastsnap:
        # We don't have this data yet, so load it
        dendro_file = run+'_snapshot_'+snapshot+'_min_val1e3.fits'
        print("Reading ", dendro_file)
        dendro = Dendrogram.load_from(dendro_file)
        gizmofile = dir+"snapshot_"+snapshot+".hdf5"
        print("Reading ", gizmofile)
        x,den = load_pos_den(gizmofile)
        starpos,starmass = load_star_pos_mass(gizmofile)

    # Find the leaf in the dendro file    
    all_leaf_ids = [leaf.idx for leaf in dendro.leaves]
    print(all_leaf_ids) 
    loc = np.where(np.array(all_leaf_ids) == leafid)[0]
    print("leaf id, loc", leafid, loc)
    leaf = (dendro.leaves)[loc[0]]
    if not leaf.is_leaf:
        notleaves.append(IDstr[ind][0])
        #print(" %s is not a leaf" %IDstr[ind][0])

    fig,ax = plt.subplots()
    ax.scatter(x[:,0], x[:,1], alpha=0.2,s=20, color='black')  
    mask = leaf.get_mask()
    ax.scatter(x[mask,0], x[mask,1], alpha=0.5,s=20, color='blue')
    ax.scatter(starpos[:,0],starpos[:,1], alpha=1.0, s=starmass*120, color='red', marker="*")
                
    for j,star in enumerate(starmass):
        ax.annotate(str(star), (starpos[j,0],starpos[j,1]+0.01), fontsize=20) #, xycoords=data)

    #maxx = np.max(x[mask,0])
    #maxy = np.max(x[mask,1])
    #minx = np.min(x[mask,0])
    #miny = np.min(x[mask,1])
    #meanx = np.mean(x[mask,0])
    #meany = np.mean(x[mask,1])
    centeridx = leaf.get_peak()[0]
    center=x[centeridx]
    #print("Center =", centeridx, center)
    ax.scatter(center[0],center[1], s=20, color="green", marker="+")
    plt.xlim([center[0]-0.2,center[0]+0.2])
    plt.ylim([center[1]-0.2,center[1]+0.2])
    fig = plt.gcf()
    fig.set_size_inches(12,12)
    fig.savefig('Snapshot_'+snapshot+'_leaf_'+IDstr[ind][0]+'_'+str(clusid[i])+'.png', dpi=100)    
    
    lastsnap = snap
    
print("Not leaves = ", notleaves)    
