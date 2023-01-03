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
    starpos = []
    starmass = []
    if 'PartType5' in f.keys():
        starpos = f['PartType5']['Coordinates'][:]
        starmass = f['PartType5']['Masses'][:]
    return starpos, starmass


#create leaf histories
def create_leaf_history(fns,run): #, ntimes, time_file):
    leaf_histories = []
    tree = []
    nhist = 0

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

            # Start building the merger tree
            for m,leaf in enumerate(leaves):
                leaf_histories.append([str("%05i"%(int(snap))) + str(leaf.idx)])
                tree.append([nhist])
                nhist+=1

        next_leaves = next_dendro.leaves
        sz_next_leaves = len(next_leaves)
        # Store ids of particles in each leaf
        next_leaves_indices = np.zeros(shape=(sz_next_leaves,), dtype=np.ndarray)
        # Whether leaf at next snapshot has a match with the current one
        # Increment by 1 for each match
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
                nhist+=1
                tree.append([nhist])

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
                    fraction_leaf[m] = overlap[m]/len(ids[leaf.get_mask()])
                    fraction_nextleaf[m] = overlap[m]/len(next_leaves_indices[m])
            # Match where more than half of cells match for either snapshot
            # Merging: next_leaf could be matched to more than one leaf
            # Split: This leaf matches to more than one next_leaf
            match_ind = np.where(np.logical_or(fraction_leaf > 0.5, fraction_nextleaf > 0.5))[0]
            
            # Add to end of appropriate row
            if len(match_ind) > 0:
                print('Match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind], match_ind)
                # Add to the appropriate row
                match = match_ind[0]
                matched_next_leaf = next_leaves[match].idx
                matched[match] += 1
                matched_hist_index = matched_history[0]
                leaf_histories[matched_hist_index].append(str("%05i"%(int(next_snap)))+str(matched_next_leaf))

                if len(match_ind)>1:
                    tmphist = leaf_histories[matched_hist_index].copy() # Store in case of split (below)
                    print(" "+str(len(match_ind))+" splits found at ",str("%05i"%(int(snap)))+str(leaf.idx))
                    tree[matched_hist_index].append("s"+str("%05i"%(int(snap)))+str(leaf.idx))
                    for i in match_ind[1:]:
                        matched[i] += 1
                        print("... matching with next leaf %i" %next_leaves[i].idx)
                        matched_next_leaf = next_leaves[i].idx
                        split = tmphist.copy()
                        split.append(str("%05i"%(int(next_snap)))+str(matched_next_leaf)) 
                        print(" New split history = ", split)
                        leaf_histories.append(split)
                        nhist+=1
                        tree.append([matched_hist_index, "s"+str("%05i"%(int(snap)))+str(leaf.idx)])
            else:
                print('No match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind], match_ind)

        # Put remaining unmatched leaves in a new row
        unmatched = np.where(matched == 0)[0]
        print(" Final splits/mergers = ", np.where(matched > 1)[0])
        if unmatched.any():
            for item_idx in unmatched:
                if overlap[item_idx]> 0:
                    print("Unmated leaf: ", next_leaves[item_idx].idx, fraction_leaf[item_idx], fraction_nextleaf[item_idx])
                leaf_histories.append([str("%05i"%(int(next_snap))) + str(next_leaves[item_idx].idx)])
                tree.append([nhist])
                nhist+=1

                
        dendro = next_dendro
        ids = next_ids
        snap = next_snap
        del next_dendro
        del next_ids
        gc.collect()

    return leaf_histories, tree


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

def read_properties(fns):
    # Read in all the properties in the files.
    converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)
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
    
    profiles = pd.concat(frames)
    return profiles

   
   
