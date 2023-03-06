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
# Author: S. Offner
#------------------------------------------------------

# STARFORGE snapshot files
dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'

# Run name
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'

# Version of analysis files
vers = '_v7'

# Leaf history file
hist_file = 'leaf_history_'+run+vers+'.csv'
print(hist_file)

# Core property files
files = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_all_prop_v7.csv'

# Split History
tree_file = 'tree_history_'+run+vers+'.csv'

# Merge History
merge_file = 'merge_history_'+run+vers+'.csv'

# Specific leaves to look at
all_tracks = True # If True, plot all tracks
track_list = [0,1] # Else plot these

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

with open(merge_file, newline='') as file:
    data = reader(file)
    merge_tree = list(data)

if all_tracks:
    ntracks = len(leaf_history)
    track_list = list(np.linspace(0, ntracks-1, ntracks))

# Get some statistics about mergers and splits
tree_ind = []     # Indicies of all parent and child tracks
splits = []       # Leaf names where splits occur
num_nosplit = 0   # Count number of tracks that never split 
for hist in leaf_tree:
    tree_ind.append(hist[0])
    splits.append(hist[1:])
    if hist[1] == '':
        num_nosplit += 1

merge_leaf = []     # Leaf that is in common
merge_parent = []   # Index in leaf_history of merge parent
merge_children = [] # Index in leaf_history of merge child (terminated)
n_merge = []        # Num of leaves that merge together at the same time
for hist in merge_tree:
    short_hist = [int(float(x)) for x in hist if x !=""]
    merge_leaf.append(short_hist[0]) 
    merge_parent.append(short_hist[1])
    merge_children.extend(short_hist[2:]) # Add indicies at the end
    n_merge.append(len(short_hist)-1)     # Don't count leaf name [0]
    #print("merge_children =", merge_children)

print("Median, mean, max number of merges =", np.median(n_merge), np.mean(n_merge), np.max(n_merge))
# Indicies of parents (may or may not have childern)
unique_ind = np.unique(tree_ind) 
# Leaf names where split occurs
split_ind = np.unique(splits) # Each appears 2 or more times)

nsplit = len(split_ind)    # How many times a split occured (into 2 or more)
nunique = len(unique_ind)  # How many parents 
print("Number of tracks that never split %i (frac %f)" %(num_nosplit, num_nosplit/ntracks))
print("Number of times splits occur %i, num unique indicies %i, num parents %i (frac %f, %f) =" %(nsplit, nunique, nunique-num_nosplit, nsplit/ntracks, nunique/ntracks))
print("Number of times merges occur: %i (%i unique parents) )" %(len(merge_leaf), len(np.unique(merge_parent))))
    
# Read in core properties
fns = glob.glob(files)
fns.sort()
profiles = read_properties(fns)

r_grid = np.logspace(np.log10(2e-3), np.log10(0.2), 20)

# Store properties in arrays
IDstr = profiles['ID'].values # ID in padded string form
density = profiles['Density [cm^-3]'].values
veldisp = profiles['Dispersion [cm/s]'].values
# Bulk properties
lradius = np.array(profiles['Reff [pc]'].values)
rcoh = profiles['CoherentRadius [pc]'].values # Calculated from profile
lrhopow = profiles['DensityIndex'].values # Calculated from profile
lvbulk = profiles['V Bulk [cm/s]'].values
ldisp = profiles['LeafDisp [cm/s]'].values
lmass = profiles['LeafMass [msun]'].values
lcenter = profiles['Center Position [pc]'].values
lidx = profiles['Center index'].values
lmax = profiles['Max Den [cm^-3]'].values
#leigenvals = profiles['Eigenvals [pc]'].values # <=v5
#leigenvecs = profiles['Eigenvecs'].values
lshape = profiles['Shape [pc]'].values
lhalf = profiles['Half Mass R[pc]'].values
lb = profiles['Mean B [G]'].values
lke = profiles['LeafKe'].values 
#print(" *** Correcting lgrave for G value -> /10... (required for vers <=v6)")
lgrave = np.abs(profiles['LeafGrav'].values)
lmage = profiles['Mag. Energy'].values 
lsink = profiles['Num. Sinks'].values
lsinkmass = profiles['Sink Masses [Msun]'].values
lsound = profiles['Sound speed [cm/s]'].values # in m/s
lproto = profiles['Protostellar'].values
lmax = profiles['Max Den [cm^-3]'].values
lkeonly = profiles['LeafKe only'].values 

IDnum = []  #Get IDs in numeric form
maxden = []
mass = []
disp = []
half = []
proto = []
time = []
vir = []
virmag = []
virkeonly = []
grav = []
ke = []
mag = []
rpow = []
coher = []
shape = []
aspect = []
aspect2 = []
denprof = []
velprof = []
rhopow = []

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
                aspect2.append([asp[0]/asp[2]])
                coher.append([rcoh[ind][0]])
                rhopow.append([lrhopow[ind][0]])
                virkeonly.append([lkeonly[ind][0]/lgrave[ind][0]])
        
            else:
                #print(i, num, ind, leafid, len(maxden))
                maxden[num].append(lmax[ind][0])
                mass[num].append(lmass[ind][0])
                disp[num].append(ldisp[ind][0])
                half[num].append(lhalf[ind][0]) 
                time[num].append(i*dt)
                vir[num].append(lke[ind][0]/lgrave[ind][0])
                virmag[num].append(lmage[ind][0]/lgrave[ind][0])
                asp = lshape[ind][0]
                aspect[num].append(asp[1]/asp[2])
                aspect2[num].append(asp[0]/asp[2])
                coher[num].append(rcoh[ind][0])
                proto[num].append(lproto[ind][0])
                rhopow[num].append(lrhopow[ind][0])
                virkeonly[num].append(lkeonly[ind][0]/lgrave[ind][0])
        
            i+=1
    tracklength[num] = np.max(time[num])+dt/2.0
    stars[num] = np.max(proto[num])
    #print("stars =", num, stars[num], proto[num])
    #if np.min(mass[num]) < 0.1: #Check
    #    lowmass[num] = 1.0

# Check
#print("Number of tracks with a low mass core =", np.sum(lowmass))

n_proto = len(np.where(stars > 0)[0])
print("Total number of tracks, number of tracks with stars =", ntracks, n_proto)
# This includes merged tracks whose stars get added to their parents.
# Need to count and remove the merging tracks with stars
print("Total number of stars in cores =", np.sum(stars))
print("Fraction of starless cores that disperse or merge  = ", (ntracks-n_proto)/ntracks) 

starless_merge =  common_member(merge_children, np.where(stars == 0)[0]) 
protostellar_merge =  common_member(merge_children, np.where(stars > 0)[0]) 

print("Number, fraction of starless tracks that merge =", len(starless_merge), len(starless_merge)/ntracks)
print("Number, fraction of starless cores that disperse (subtract merges) =", (ntracks-n_proto-len(starless_merge)), (ntracks-n_proto-len(starless_merge))/ntracks)
# Note this only counts the children, i.e., excludes parent that continues after the merge 
print("Number of merged tracks with stars, fraction of merges =", len(protostellar_merge), len(protostellar_merge)/len(merge_children))
print("Number of stars in merging tracks =", np.sum(stars[protostellar_merge]))

# Make plots
fig,ax = plt.subplots()
ax.set_xlabel("t [Myr]")
ax.set_ylabel("N")
ax.set_xscale('log')
hist, bins = np.histogram(tracklength, bins=12)
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#ax.hist(tracklength, color='blue', bins=logbins)
n, bins, patches = ax.hist(tracklength, color='grey',bins=logbins, label='All', alpha=0.5)
n, bins, patches = ax.hist(tracklength[np.where(stars > 0)[0]], color='green', bins=logbins,  label='Protostellar', alpha=0.5)
n, bins, patches = ax.hist(tracklength[np.unique(merge_children)], color='blue', bins=logbins,  label='Merging', alpha=0.5)

ax.legend()
plt.xlim([1e-2, 7])
fig.savefig("Track_lengths"+vers+".png")

# General Plotting Function
def make_plot(xdata, ydata, stars, vers, xlabel="t [Myr]", ylabel="$R_{coh}$ [pc]", xlog=True, ylog=True, filename="Rcoh"):

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    ax = gs.subplots(sharex=True,sharey=True)
    ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    if xlog: 
        ax[0].set_yscale('log')
    if ylog:
        ax[0].set_xscale('log')
    i =0
    for t, c in zip(xdata,ydata):
        if stars[i] == 0:
            line1, =  ax[0].plot(t, c, alpha=0.05, color='grey', label='Starless')
        elif stars[i] == 1:
            line2, = ax[1].plot(t, c, alpha=0.05, color='blue', label='Single Protostar')
        else:
            line3, = ax[1].plot(t, c, alpha=0.05, color='green', label='Multple Protostars')
            i=i+1

    ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], handlelength=1)
    ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], hanglelength=1)
    fig.savefig(filename+"_alltracks"+vers+".png")

make_plot(time, vir, stars, vers, xlabel="t [Myr]", ylabel=r'$\alpha_{vir}$', xlog=True, ylog=True, filename="Vir")

make_plot(time, coher, stars, vers, xlabel="t [Myr]", ylabel="$R_{coh}$ [pc]", xlog=True, ylog=True, filename="Rcoh")

make_plot(time, maxden, stars, vers, xlabel="t [Myr]", ylabel=r'$n_{max}$ [cm$^{-3}$]', xlog=True, ylog=True, filename="Maxden")

make_plot(time, virmag, stars, vers, xlabel="t [Myr]", ylabel=r'$\alpha_{vir,B}$', xlog=True, ylog=True, filename="VirMag")

make_plot(time, virkeonly, stars, vers, xlabel="t [Myr]", ylabel=r'$\alpha_{vir}$', xlog=True, ylog=True, filename="Virkeonly")

make_plot(time, aspect, stars, vers, xlabel="t [Myr]", ylabel='Aspect Ratio $b/a$', xlog=True, ylog=True, filename="Asp")

make_plot(time, aspect2, stars, vers, xlabel="t [Myr]", ylabel='Aspect Ratio $c/a$', xlog=True, ylog=True, filename="Asp2")

make_plot(time, half, stars, vers, xlabel="t [Myr]", ylabel="$R_{half}$ [pc]", xlog=True, ylog=True, filename="Rhalf")

make_plot(time, disp, stars, vers, xlabel="t [Myr]", ylabel=r'$\sigma$ [km/s]', xlog=True, ylog=True, filename="Vdisp")

make_plot(time, mass, stars, vers, xlabel="t [Myr]", ylabel=r'$m$ [M$_\odot$]', xlog=True, ylog=True, filename="Mass")

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("r [pc]")
ax[0].set_ylabel("Log $n$ [cm$^{-3}$]")
#ax[0].set_yscale('log')
ax[0].set_xscale('log')
i = 0
for rho in density:
    prof, ind = np.unique(rho, return_index=True)
    if lproto[i] == 0:
        line1, = ax[0].plot(r_grid[ind], rho[ind], alpha=0.005, color='grey',label='Starless')
    elif lproto[i] == 1:
        line2, = ax[1].plot(r_grid[ind], rho[ind], alpha=0.005, color='blue', label='Single')
    else:
        line3, = ax[1].plot(r_grid[ind], rho[ind], alpha=0.005, color='green', label='Multiple')
    i=i+1

ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], handlelength=1)
ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], handlelength=1)
fig.savefig("Density_profiles_all"+vers+".png", color='blue')

fig = plt.figure()
gs = fig.add_gridspec(2, hspace=0)
ax = gs.subplots(sharex=True,sharey=True)
ax[1].set_xlabel("r [pc]")
ax[0].set_ylabel("$\sigma$ [km/s]")
#ax[0].set_yscale('log')
ax[0].set_xscale('log')
i = 0
for vel in veldisp:
    prof, ind = np.unique(vel, return_index=True)
    if lproto[i] == 0:
        ax[0].plot(r_grid[ind], np.array(vel[ind])/1000., alpha=0.005, color='grey',label='Starless')
    elif lproto[i] == 1:
        ax[1].plot(r_grid[ind], np.array(vel[ind])/1000., alpha=0.005, color='blue',label='Single')
    else:
        ax[1].plot(r_grid[ind], np.array(vel[ind])/1000., alpha=0.005, color='green',label='Multiple')
    i=i+1

ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], handlelength=1)
ax[0].legend(handles=[line1, line2, line3], loc='upper right', labelcolor=['grey', 'blue', 'green'], handlelength=1)
fig.savefig("Veldisp_profiles_all"+vers+".png", color='blue')



