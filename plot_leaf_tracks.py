import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
from astrodendro import Dendrogram
from leaf_history_functions import *
from csv import reader
from matplotlib.pyplot import cm
from sklearn.utils import shuffle

#-------------------------------------------------------
# Code to read in leaf histories and plot images of leaf 
# evolution in each track.
#------------------------------------------------------

# STARFORGE snapshot files
dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'

# Run name
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'

# Version of analysis files
vers = '_v3'

# Leaf history file
hist_file = 'leaf_history_'+run+vers+'.csv'

# Core property files
files = '_M2e3_cores_v4_proto/M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_snapshot_*_prop_v4.csv'

# Specific leaves to look at
all_tracks = True # If True, plot all tracks
track_list = [3, 30, 82, 7,8] # Else plot these

# Image box radius
boxrad = 0.2 

# Time increment between snapshots
dt = 0.026    

with open(hist_file, newline='') as file:
    data = reader(file)
    leaf_history = list(data)

if all_tracks:
    ntracks = len(leaf_history)
    track_list = list(np.linspace(0, ntracks-1, ntracks))

# Read in core properties
converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
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
                                    'Eigenval [pc]':converter_nparray,
                                     'Mean B [G]':converter_nparray
                                    })
    frames.append(data)
    
profiles = pd.concat(frames)

IDstr = profiles['ID'].values # ID in padded string form
density = profiles['Density [cm^-3]'].values
veldisp = profiles['Dispersion [cm/s]'].values
# Bulk properties
lradius = np.array(profiles['Reff [pc]'].values)
rcoh = profiles['CoherentRadius [pc]'].values # Calculated from profile
rhopow = profiles['DensityIndex'].values # Calculated from profile
lvbulk = profiles['V Bulk [cm/s]'].values
ldisp = profiles['LeafDisp [cm/s]'].values
lmass = profiles['LeafMass [msun]'].values
lcenter = profiles['Center Position [pc]'].values
lidx = profiles['Center index'].values
lmax = profiles['Max Den [cm^-3]'].values
leigenvals = profiles['Eigenvals [pc]'].values
leigenvecs = profiles['Eigenvecs'].values
lhalf = profiles['Half Mass R[pc]'].values
lb = profiles['Mean B [G]'].values
lke = profiles['LeafKe'].values #Still in code units
lgrave = profiles['LeafGrav'].values #Still in code units
lmage = profiles['Mag. Energy'].values #Still in code units
lsink = profiles['Num. Sinks'].values
lsinkmass = profiles['Sink Masses [Msun]'].values
lsound = profiles['Sound speed [cm/s]'].values # in m/s

IDnum = []  #Get IDs in numeric form
for i in IDstr:
    IDnum.append(int(i))

for track_num in track_list:
    track_num = int(track_num) 
    track_ids = leaf_history[track_num]

    time = 0
    extra = 0 # How many extra snaps to plot after leaf end
    maxden = [] # To store this leaf tracks properties
    mass = []
    disp = []
    half = []

    for leafid in track_ids:
        print("leafid =", leafid)
        if leafid:
            snapshot = leafid[2:5]
            snap = int(snapshot)
            leaf_idx = int(leafid[5:])
            ind = np.where(IDstr == leafid)[0]
            print("Snapshot =", snap, " leaf_idx=", leaf_idx, " IDstr =", IDstr[ind])
            print("Leaf properties =", ldisp[ind], rcoh[ind], lmass[ind])
            maxden.append(lmax[ind][0])
            mass.append(lmass[ind][0])
            disp.append(ldisp[ind][0])
            half.append(lhalf[ind][0])

            # Get leaf information from dendro file and Gizmo snapshot
            dendro_file = run+'_snapshot_'+snapshot+'_min_val1e3.fits'
            print("Reading ", dendro_file)
            dendro = Dendrogram.load_from(dendro_file)
            gizmofile = dir+"snapshot_"+snapshot+".hdf5"
            print("Reading ", gizmofile)
            x,den = load_pos_den(gizmofile)
            starpos,starmass = load_star_pos_mass(gizmofile)

            # Find the leaf in the dendro file    
            all_leaf_ids = [leaf.idx for leaf in dendro.leaves]
            #print(all_leaf_ids) 
            loc = np.where(np.array(all_leaf_ids) == leaf_idx)[0]
            leaf = (dendro.leaves)[loc[0]]
            print("Check: leaf id, leaf.idx, loc", leaf_idx, leaf.idx, loc)  
  
            # Make plot
            fig,ax = plt.subplots()
            ax.scatter(x[:,0], x[:,1], alpha=0.1,s=45, color='black',edgecolors='none')
            color = iter(cm.viridis(shuffle(np.linspace(0, 1, len(dendro.leaves)))))
            for otherleaf in dendro.leaves:
                mask = otherleaf.get_mask()
                c= next(color)
                ax.scatter(x[mask,0], x[mask,1], alpha=0.4,s=45, color=c,edgecolors='none')
   
            mask = leaf.get_mask()
            ax.scatter(x[mask,0], x[mask,1], alpha=0.5,s=50, color='blue')
   
            if len(starmass)>0:
                ax.scatter(starpos[:,0],starpos[:,1], alpha=1.0, s=starmass*120, color='red', marker="*",zorder=20)
                #for j,star in enumerate(starmass):
                #    ax.annotate(str(star), (starpos[j,0],starpos[j,1]+0.01), fontsize=20) #, xycoords=data)

            centeridx = leaf.get_peak()[0]
            center=x[centeridx]
            ax.scatter(center[0],center[1], s=50, color="green", marker="+",zorder=15)
            plt.xlim([center[0]-boxrad, center[0]+boxrad])
            plt.ylim([center[1]-boxrad, center[1]+boxrad])
            ax.set_xlabel('X [pc]', fontsize=15)
            ax.set_ylabel('Y [pc]', fontsize=15)
            #ax.annotate("%2.3f Myr"%time, (center[0]+0.12,center[1]+0.17), fontsize=20)#, prop=dict(weight='bold')) #, xycoords=data)
            t = ax.text(center[0]+0.12,center[1]+0.17, "%2.3f Myr"%time, fontsize=20)
            t.set_bbox(dict(facecolor="white", edgecolor="white"))
            fig = plt.gcf()
            fig.set_size_inches(12,12)
            fig.savefig('Snapshot_'+snapshot+'_leaf_'+leafid+'_'+str(track_num)+vers+'.png', dpi=100)    
            time = time + dt

            # Read and plot the next snapshot in the sequence
        else:
            snap = snap+2
            snapshot = "%03i" %snap
            extra = extra+1

            dendro_file = run+'_snapshot_'+snapshot+'_min_val1e3.fits'
            print("Reading ", dendro_file)
            dendro = Dendrogram.load_from(dendro_file)
            gizmofile = dir+"snapshot_"+snapshot+".hdf5"
            print("Reading ", gizmofile)
            x,den = load_pos_den(gizmofile)
            starpos,starmass = load_star_pos_mass(gizmofile)
            

            # Make core plot
            fig,ax = plt.subplots()
            ax.scatter(x[:,0], x[:,1], alpha=0.1,s=45, color='black', edgecolors='none')
            color = iter(cm.viridis(shuffle(np.linspace(0, 1, len(dendro.leaves)))))
            for otherleaf in dendro.leaves:
                mask = otherleaf.get_mask()
                c = next(color)
                ax.scatter(x[mask,0], x[mask,1], alpha=0.4,s=45, color=c, edgecolors='none')
 
            if len(starmass)>0:
                ax.scatter(starpos[:,0],starpos[:,1], alpha=1.0, s=starmass*120, color='red', marker="*")
                #for j,star in enumerate(starmass):
            #    ax.annotate(str(star), (starpos[j,0],starpos[j,1]+0.01), fontsize=20) #, xycoords=data)
            # Use center from last snapshot
            ax.scatter(center[0],center[1], s=40, color="green", marker="+")
            plt.xlim([center[0]-boxrad, center[0]+boxrad])
            plt.ylim([center[1]-boxrad, center[1]+boxrad])
            ax.set_xlabel('X [pc]', fontsize=15)
            ax.set_ylabel('Y [pc]', fontsize=15)

            t = ax.text(center[0]+0.12,center[1]+0.17, "%2.3f Myr"%time, fontsize=20)
            t.set_bbox(dict(facecolor="white", edgecolor="white"))

            fig = plt.gcf()
            fig.set_size_inches(12,12)
            fig.savefig('Snapshot_'+snapshot+'_leaf_'+leafid+'_'+str(track_num)+vers+'.png', dpi=100)
            plt.close()

            if extra > 3:

                # Make plots
                times = np.linspace(0, time, len(maxden))
                fig,ax = plt.subplots()
                ax.plot(times, maxden, marker="o")
                ax.set_xlabel("t [Myr]")
                ax.set_ylabel("$n_{max}$ [cm$^{-3}$]")
                fig.savefig("Max_den_track"+str(track_num)+vers+".png")

                fig,ax = plt.subplots()
                ax.plot(times, half, marker="o")
                ax.set_xlabel("t [Myr]")
                ax.set_ylabel("$R_{half}$ [pc]")
                fig.savefig("Rhalf_track"+str(track_num)+vers+".png")

                fig,ax = plt.subplots()
                ax.plot(times, np.array(disp)/1e5, marker="o")
                ax.set_xlabel("t [Myr]")
                ax.set_ylabel("$\sigma$ [km/s]")
                fig.savefig("Vdisp_track"+str(track_num)+vers+".png")
            
                fig,ax = plt.subplots()
                ax.plot(times, mass, marker="o")
                ax.set_xlabel("t [Myr]")
                ax.set_ylabel("$m$ [M$_\odot$]")
                fig.savefig("Mass_track"+str(track_num)+vers+".png")

                break
            time = time + dt
