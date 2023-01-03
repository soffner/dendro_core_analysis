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
import imageio

#-------------------------------------------------------
# Code to read in leaf histories and plot the time 
# evolution of the properties for the tracks
#------------------------------------------------------

# STARFORGE snapshot files
dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'

# Run name
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'

# Version of analysis files
vers = '_v4'

# Leaf history file
hist_file = 'leaf_history_'+run+vers+'.csv'
print(hist_file)

# Core property files
files = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_snapshot_*_prop_v4.csv'

# Split Tree
tree_file = 'tree_history_'+run+vers+'.csv'

# Specific leaves to look at
all_tracks = True # If True, plot all tracks
track_list = [0,1,30] # Else plot these

# Image box radius
boxrad = 0.2 
# Time increment between snapshots
dt = 0.013    

with open(hist_file, newline='') as file:
    data = reader(file)
    leaf_history = list(data)

with open(tree_file, newline='') as file:
    data = reader(file)
    leaf_tree = list(data)

if all_tracks:
    ntracks = len(leaf_history)
    track_list = list(np.linspace(0, ntracks-1, ntracks))

# Get some statistics about mergers and splits
tree_ind = []     # Indicies of all parent and child tracks
splits = []       # Leaf names where splits occur
num_nosplit = 0  # Count number of tracks that never split 
for hist in leaf_tree:
    tree_ind.append(hist[0])
    splits.append(hist[1:])
    if hist[1] == '':
        num_nosplit += 1

# Indicies of parents (may or may not have childern)
unique_ind = np.unique(tree_ind) 
# Leaf names where split occurs
split_ind = np.unique(splits) # Each appears 2 or more times)

nsplit = len(split_ind)    # How many times a split occured (into 2 or more)
nunique = len(unique_ind)  # How many parents 
print("Number of tracks that never split %i (frac %f)" %(num_nosplit, num_nosplit/ntracks))
print("Number of times splits occur %i, num unique indicies %i, num parents %i (frac %f, %f) =" %(nsplit, nunique, nunique-num_nosplit, nsplit/ntracks, nunique/ntracks))

last_leaf = []
for track in leaf_history:
    last_leaf.append(np.unique(track)[-1])


# Get indicies of the merging tracks
u, mergeind, counts = np.unique(last_leaf, return_inverse=True, return_counts=True)
i = 0
matches = []
for leaf in u:
    matches.append([leaf])
    for j, track in enumerate(leaf_history):
        ind = [k for k,x in enumerate(track) if x==leaf]
        if ind:
            matches[i].append([j, ind])
    i=i+1
print("matches =", matches)

mergers = np.where(counts > 1)[0]
print("# unique, Final # merged tracks, # merging, counts:", len(u), len(mergers), np.sum(counts[mergers]), counts[mergers])
print("u[mergers]=", len(u[mergers]), u[mergers])
print(np.array(last_leaf).sort())
for m in u[mergers]:
    ind = [i for i,x in enumerate(last_leaf) if x==m]
    print("m, ind =", m, ind)
    for i in ind:
        print(" :", i, leaf_history[i])
    

# Read in core properties
fns = glob.glob(files)
fns.sort()
profiles = read_properties(fns)

# Store properties in arrays
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
#leigenvals = profiles['Eigenvals [pc]'].values
#leigenvecs = profiles['Eigenvecs'].values
lshape = profiles['Shape [pc]'].values
lhalf = profiles['Half Mass R[pc]'].values
lb = profiles['Mean B [G]'].values
lke = profiles['LeafKe'].values #Still in code units
lgrave = np.abs(profiles['LeafGrav'].values) #Still in code units
lmage = profiles['Mag. Energy'].values #Still in code units
lsink = profiles['Num. Sinks'].values
lsinkmass = profiles['Sink Masses [Msun]'].values
lsound = profiles['Sound speed [cm/s]'].values # in m/s
lproto = profiles['Protostellar'].values

IDnum = []  #Get IDs in numeric form

maxden = [] # To store this leaf tracks properties
mass = []
disp = []
half = []
time = [] # Hold time for each track
vir = []
virmag = []
aspect = []
coher = []
proto = []
tracklength = np.zeros(len(track_list))
stars = np.zeros(len(track_list))
lowmass = np.zeros(len(track_list))

for i in IDstr:
    IDnum.append(int(i))

for num, track_num in enumerate(track_list):

    track_num = int(track_num) 
    track_ids = leaf_history[track_num]
   
    t = 0
    i  = 0
    
    for leafid in track_ids:
        #print("leafid =", leafid)
        if leafid:
            snapshot = leafid[2:5]
            snap = int(snapshot)
            leaf_idx = int(leafid[5:])
            ind = np.where(IDstr == leafid)[0]
            #print("Snapshot =", snap, " leaf_idx=", leaf_idx, " IDstr =", IDstr[ind])
            #print("Leaf properties =", ldisp[ind], rcoh[ind], lmass[ind])
            if i == 0:
                maxden.append([lmax[ind][0]])
                mass.append([lmass[ind][0]])
                disp.append([ldisp[ind][0]])
                half.append([lhalf[ind][0]])
                time.append([i*dt])
                proto.append([lproto[ind][0]])
                vir.append([lke[ind][0]/lgrave[ind][0]])
                virmag.append([lmage[ind][0]/lgrave[ind][0]])
                asp = lshape[ind][0]
                aspect.append([asp[1]/asp[2]])
                coher.append([rcoh[ind][0]])
        
            else:
                maxden[num].append(lmax[ind][0])
                mass[num].append(lmass[ind][0])
                disp[num].append(ldisp[ind][0])
                half[num].append(lhalf[ind][0]) 
                time[num].append(i*dt)
                vir[num].append(lke[ind][0]/lgrave[ind][0])
                virmag[num].append(lmage[ind][0]/lgrave[ind][0])
                asp = lshape[ind][0]
                aspect[num].append(asp[1]/asp[2])
                coher[num].append(rcoh[ind][0])
                proto[num].append(lproto[ind][0])
            i+=1
    tracklength[num] = np.max(time[num])+dt/2.0
    stars[num] = np.max(proto[num])
    if np.min(mass[num]) < 0.1:
        lowmass[num] = 1.0


# Find, remove tracks with low-mass cores
print("Number of tracks with a low mass core =", np.sum(lowmass))
n_proto = np.sum(np.where(stars > 0)[0])
print("Number of tracks with a star =", n_proto)
print("Total number of stars in cores =", np.sum(stars))
print("Fraction of cores that disperse with no star = ", n_proto/len(mass)) 
#ind = np.where(lowmass == 0)[0]
#tracklength = tracklength[ind]
#maxden = maxden[ind]
#mass = mass[ind]
#disp = disp[ind]
#half = half[ind]
#vir = vir[ind]
#virmag = virmag[ind]
#stars = stars[ind]
#coher = coher[ind]
#proto = proto[ind]
#time = time[ind]
#aspect = aspect[ind]

# Make plots
fig,ax = plt.subplots()
ax.set_xlabel("t [Myr]")
ax.set_ylabel("N")
ax.set_xscale('log')
hist, bins = np.histogram(tracklength, bins=12)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#ax.hist(tracklength, color='blue', bins=logbins)
n, bins, patches = ax.hist(tracklength[np.where(stars > 0)[0]], color='green', bins=logbins, stacked=True, label='Protostellar', alpha=0.5)
n, bins, patches = ax.hist(tracklength[np.where(stars == 0)[0]], color='grey',bins=logbins, stacked=True, label='Starless', alpha=0.5)
ax.legend()
plt.xlim([1e-2, 7])
fig.savefig("Track_lengths"+vers+".png")

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel("$R_{coh}$ [pc]")
#ax.set_yscale('log')
ax[0].set_xscale('log')
i =0
for t, c in zip(time,coher):
    if stars[i] == 0:
        ax[0].plot(t, c, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, c, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, c, alpha=0.05, color='green')
    i=i+1

fig.savefig("Rcoh_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel(r'$\alpha_{vir}$')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i = 0
for t, v in zip(time,vir):
    if stars[i] == 0:
        ax[0].plot(t, v, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, v, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, v, alpha=0.05, color='green')
    i=i+1

fig.savefig("Vir_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel(r'$\alpha_{vir,B}$')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i = 0
for t, v in zip(time,virmag):
    if stars[i] == 0:
        ax[0].plot(t, v, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, v, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, v, alpha=0.05, color='green')
    i=i+1

fig.savefig("VirMag_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel('Aspect Ratio $b/a$')
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i=0
for t, a in zip(time,aspect):
    if stars[i] == 0:
        ax[0].plot(t, a, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, a, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, a, alpha=0.05, color='green')
    i=i+1

fig.savefig("Asp_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel("$n_{max}$ [cm$^{-3}$]")
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i=0
for t, den in zip(time,maxden):
    if stars[i] == 0:
        ax[0].plot(t, den, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, den, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, den, alpha=0.05, color='green')
    i=i+1

fig.savefig("Max_den_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel("$R_{half}$ [pc]")
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i=0
for t, h in zip(time,half):
    if stars[i] == 0:
        ax[0].plot(t, h, alpha=0.05, color='blue')
    elif stars[i] == 1:
        ax[1].plot(t, h, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, h, alpha=0.05, color='green')
    i=i+1

fig.savefig("Rhalf_alltracks"+vers+".png")

#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel("$\sigma$ [km/s]")
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i=0
for t, d in zip(time,disp):
    if stars[i] == 0:
        ax[0].plot(t, np.array(d)/1e5, alpha=0.05, color='blue')
    elif stars[i] == 1:
        ax[1].plot(t, np.array(d)/1e5, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, np.array(d)/1e5, alpha=0.05, color='green')
    i=i+1        

fig.savefig("Vdisp_alltracks"+vers+".png")
            
#fig,ax = plt.subplots()
fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("t [Myr]")
ax[0].set_ylabel("$m$ [M$_\odot$]")
ax[0].set_yscale('log')
ax[0].set_xscale('log')
i = 0
for t, m in zip(time,mass):
    if stars[i] == 0:
        ax[0].plot(t, m, alpha=0.05, color='grey')
    elif stars[i] == 1:
        ax[1].plot(t, m, alpha=0.05, color='blue')
    else:
        ax[1].plot(t, m, alpha=0.05, color='green')
    i=i+1

fig.savefig("Mass_alltracks"+vers+".png", color='blue')
