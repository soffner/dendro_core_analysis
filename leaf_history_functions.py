from astrodendro import Dendrogram
import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
import gc as gc

# Load only particle ids
def load_data_ids(file):
    # Load snapshot data
    f = h5py.File(file, 'r')
    ids = f['PartType0']['ParticleIDs'][:]
    return ids


# Load only coordinates and densities
def load_pos_den(file):
    # Load snapshot data
    f = h5py.File(file, 'r')
    x = f['PartType0']['Coordinates'][:]
    den = f['PartType0']['Density'][:]
    return x, den

def load_star_pos_mass(file):
    # Load snapshot data
    f = h5py.File(file, 'r')
    starpos = f['PartType5']['Coordinates'][:]
    starmass = f['PartType5']['Masses'][:]
    return starpos, starmass


#create leaf histories
def create_leaf_history(fns,run): #, ntimes, time_file):
    leaf_histories = []

    # Load first snapshot
    snap = fns[0][-8:-5]
    dendro_file = run+'_snapshot_'+snap+'_min_val1e3.fits' 
    dendro = Dendrogram.load_from(dendro_file)
    ids = load_data_ids(fns[0])
    print("*** Starting from snapshot ", snap)

    # Loop through snapshots
    for j,snapshot in enumerate(fns[:-1]):
        next_snap = fns[j+1][-8:-5]
        next_dendro_file = run+'_snapshot_'+next_snap+'_min_val1e3.fits' 
        next_dendro = Dendrogram.load_from(next_dendro_file)
        next_ids = load_data_ids(fns[j+1])
        print(" --- ", next_snap)

        leaves = dendro.leaves
        if j == 0: # We're at the beginning, start fresh
            for leaf in leaves:
                leaf_histories.append([str("%05i"%(int(snap))) + str(leaf.idx)])

        next_leaves = next_dendro.leaves
        sz_next_leaves = len(next_leaves)
        # Store ids of particles in each leaf
        next_leaves_indices = np.zeros(shape=(sz_next_leaves,), dtype=np.ndarray)
        # Whether leaf at next snapshot has a match with the current one
        matched = np.zeros(sz_next_leaves)

        for m,leaf in enumerate(next_leaves):
            # Get the ids of the particles in this leaf 
            indices = next_ids[leaf.get_mask()]
            next_leaves_indices[m] = indices


        # Get location of current leaf in the leaf_history
        for leaf in leaves:
            matched_history = [index[0] for index in enumerate(leaf_histories) if (str("%05i")%(int(snap)) + str(leaf.idx)) in leaf_histories[index[0]]]
            if not matched_history:
                print('Start leaf history')
                leaf_histories.append([str("%05i"%(int(snap))) + str(leaf.idx)])
                matched_history = [len(leaf_histories)-1]

            # Old position criterion
            #found_match = bool([ele for ele in proj_x if(ele in next_snapshot_x)] ) & bool([ele for ele in proj_y if(ele in next_snapshot_y)] )

            # Number of cells in common with each leaf in next snapshot
            overlap = np.zeros(sz_next_leaves) 

            # Fraction of current leaf in common
            fraction_leaf = np.zeros(sz_next_leaves)

            # Fraction of next leaf in common
            fraction_nextleaf = np.zeros(sz_next_leaves)
            
            for m in range(0,sz_next_leaves):
                set = [i for i in ids[leaf.get_mask()] if i in next_leaves_indices[m]]
                if set:
                    overlap[m] = len(set)
                    fraction_leaf[m] = overlap[m]/len(leaf.get_mask())
                    fraction_nextleaf[m] = overlap[m]/len(next_leaves_indices[m])
    
            print("Matching fraction for leaf i =", leaf.idx, np.sort(fraction_leaf)[0:5])
            # Match where more than half of cells match a leaf at the next time
            # Direction: small to bigger leaf 
            # Merging: next_leaf could be matched to more than one leaf
            match_ind = np.where(fraction_leaf > 0.5)[0] 

            # Add to end of appropriate row
            if match_ind:
                print('Match found for leaf %i' %leaf.idx)
                # Add to the appropriate row
                matched_next_leaf = next_leaves[match_ind].idx
                matched[match_ind] = 1
                matched_hist_index = matched_history[0]
                leaf_histories[matched_hist_index].append(str("%05i"%(int(next_snap)))+str(matched_next_leaf))
                # Mergers will appear as the same leaf in multiple rows
            else:
                # Match where more than half of a nextleaf matches a current one
                # Direction: bigger to smaller leaf 
                # Splitting: leaf could be matched to more than one next_leaf
                match_ind = np.where(fraction_nextleaf > 0.5)[0] 
                if match_ind.any():
                    print('Match found for leaf %i: %i' %(leaf.idx,next_leaves[match_ind[0]].idx))
                    i = match_ind[0]
                    matched[i] = 1
                    matched_next_leaf = next_leaves[i].idx
                    matched_hist_index = matched_history[0]
                    # Append to located history
                    tmphist = leaf_histories[matched_hist_index].copy() # Store in case of split (below)
                    leaf_histories[matched_hist_index].append(str("%05i"%(int(next_snap)))+str(matched_next_leaf)) 

                    # If there's a second match, copy history and append to the end
                    if len(match_ind) > 1:
                        print("    Multi-matches found: Leaf %i splitting into %i parts" %(leaf.idx,len(match_ind)))
                        for i in match_ind[1:]:
                            matched[i] = 1
                            print("... matching with next leaf %i" %next_leaves[i].idx)
                            matched_next_leaf = next_leaves[i].idx
                            #matched_hist_index = matched_history[0]
                            split = tmphist.copy()
                            print(" Split history =", split)
                            split.append(str("%05i"%(int(next_snap)))+str(matched_next_leaf)) 
                            print(" Split history append = ", split)
                            leaf_histories.append(split)          

        # Put remaining unmatched leaves in a new row
        unmatched = np.where(matched == 0)[0]    
        if unmatched.any():
            for item_idx in unmatched:
                leaf_histories.append([str("%05i"%(int(next_snap))) + str(next_leaves[item_idx].idx)])

                
        dendro = next_dendro
        ids = next_ids
        snap = next_snap
        del next_dendro
        del next_ids
        gc.collect()

    return leaf_histories


# Read in the file
def read_leaf_history(file):

    # Retrieve leafs
    read_data=reader(file)
    leaf_history = list(read_data)

    return leaf_history

# Loop through files and plot all the leaves + numbers
def plot_leaves(fns, run):
    print("Plotting leaves ...")

    # Load first snapshot
    for i in range(len(fns)):
        print(".. Plotting ", i)
        snap = fns[i][-8:-5]
        dendro_file = run+'_snapshot_'+snap+'_min_val1e3.fits' 
        dendro = Dendrogram.load_from(dendro_file)

        x,den = load_pos_den(fns[i])
        fig,ax = plt.subplots()
        ax.scatter(x[:,0], x[:,1], alpha=0.01,s=1)
        leaves = dendro.leaves
        for leaf in leaves:
            mask = leaf.get_mask()
            ax.scatter(x[mask,0], x[mask,1], alpha=0.05,s=1)
            meanx = np.mean(x[mask,0])
            meany = np.mean(x[mask,1])
            ax.annotate(str(leaf.idx), (meanx,meany)) #, xycoords=data)
        plt.xlim([11,19])
        plt.ylim([11,19])
        fig = plt.gcf()
        fig.set_size_inches(12,12)
        fig.savefig('Snapshot_'+snap+'_leaves.png', dpi=100)
        plt.close()


   
   
