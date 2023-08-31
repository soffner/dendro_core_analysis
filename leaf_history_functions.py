from astrodendro import Dendrogram
import h5py
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sys
import gc as gc
import glob

# Load only particle ids
def load_data_ids(file, res_limit = 0.0):
    # Load snapshot data
    f = h5py.File(file, 'r')
    mask = (f['PartType0']['Masses'][:] >= res_limit*0.999)
    ids = f['PartType0']['ParticleIDs'][:]*mask
    return ids

# Load only coordinates and densities
def load_pos_den(file, res_limit=0.0):
    # Load snapshot data
    f = h5py.File(file, 'r')
    mask = (f['PartType0']['Masses'][:] >= res_limit*0.999)
    mask3d = np.array([mask, mask, mask]).T
    x = f['PartType0']['Coordinates'][:]*mask3d
    den = f['PartType0']['Density'][:]*mask
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


#create leaf histories, generate all 4 files (slow)
def create_leaf_history(fns,run,res_limit=0.0): #, ntimes, time_file):
    leaf_histories = []
    tree = []
    nhist = 0
    merge_histories = []
    nodes_edges = []
    nsplits = 0

    # Load first snapshot
    snap = fns[0][-8:-5]
    dendro_file = run+'_snapshot_'+snap+'_min_val1e3_res1e-3.fits' 
    dendro = Dendrogram.load_from(dendro_file)
    ids = load_data_ids(fns[0], res_limit=res_limit)
    print("*** Starting from snapshot ", snap)

    # Loop through snapshots
    for j,snapshot in enumerate(fns[:-1]):
        next_snap = fns[j+1][-8:-5]
        next_dendro_file = run+'_snapshot_'+next_snap+'_min_val1e3_res1e-3.fits' 
        next_dendro = Dendrogram.load_from(next_dendro_file)
        next_ids = load_data_ids(fns[j+1],res_limit=res_limit)
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
        # Increment by 1 for each match found
        matched = np.zeros(len(next_leaves))
        matched_ind = []

        for m,leaf in enumerate(next_leaves):
            # Get the ids of the particles in this leaf 
            indices = next_ids[leaf.get_mask()]
            next_leaves_indices[m] = indices
            matched_ind.append([str("%05i"%(int(next_snap))) + str(leaf.idx)]) #First one will be this leaf's index 
            

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
                for i in match_ind:
                    nodes_edges.append([str("%05i"%(int(snap)))+str(leaf.idx), str("%05i"%(int(next_snap)))+str(next_leaves[i].idx)])
                print('Match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind], match_ind)
                # Add to the appropriate row
                match = match_ind[0]
                matched_next_leaf = next_leaves[match].idx
                matched[match] += 1
                # Append row in leaf history rather than the leaf it matches with
                matched_ind[match].append(matched_history[0])#str("%05i"%(int(snap))) + str(leaf.idx))
                matched_hist_index = matched_history[0]
                tmphist = leaf_histories[matched_hist_index].copy() # Store in case of split (below)
                leaf_histories[matched_hist_index].append(str("%05i"%(int(next_snap)))+str(matched_next_leaf))

                if len(match_ind)>1:
                    print(" "+str(len(match_ind))+" splits found at ",str("%05i"%(int(snap)))+str(leaf.idx))
                    tree[matched_hist_index].append("s"+str("%05i"%(int(snap)))+str(leaf.idx))
                    for i in match_ind[1:]:
                        matched[i] += 1
                        matched_ind[i].append(len(leaf_histories))#str("%05i"%(int(snap))) + str(leaf.idx))

                        print("... matching with next leaf %i" %next_leaves[i].idx)
                        matched_next_leaf = next_leaves[i].idx
                        split = tmphist.copy()
                        split.append(str("%05i"%(int(next_snap)))+str(matched_next_leaf)) 
                        print(" New split history = ", split)
                        leaf_histories.append(split)
                        nhist+=1
                        tree.append([matched_hist_index, "s"+str("%05i"%(int(snap)))+str(leaf.idx)])
                        nsplits += 1
            else:
                print('No match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind], match_ind)

        # Put remaining unmatched leaves in a new row
        unmatched = np.where(matched == 0)[0]
        if unmatched.any():
            for item_idx in unmatched:
                #if overlap[item_idx]> 0:
                #    print("Unmated leaf: ", next_leaves[item_idx].idx, fraction_leaf[item_idx], fraction_nextleaf[item_idx])
                leaf_histories.append([str("%05i"%(int(next_snap))) + str(next_leaves[item_idx].idx)])
                tree.append([nhist])
                nhist+=1

        # Get the number of mergers and splits
        mergers = np.where(matched >1)[0]
        print(" Number of merges = ", len(mergers))
        print(" Number of splits = %i (nleaves: %i, nhist-nleaves: %i)" %(nsplits, len(leaves), nhist-len(leaves))) 
        for i in mergers: 
            print("  merged: = ",  matched_ind[i]) # These should all have at least three entries (first one is merged leaf, the second two are the current leaves matched to it)
            merge_histories.append(matched_ind[i])
        
        #print(" leaf matches =", [matched_ind[i] for i in np.where(matched >1)[0])
                
        dendro = next_dendro
        ids = next_ids
        snap = next_snap
        del next_dendro
        del next_ids
        gc.collect()

    return leaf_histories, tree, merge_histories, nodes_edges

#create leaf histories (generate nodes_edges file only)
def create_leaf_history_fast(fns,run,res_limit=0.0, search_radius=1.0): #, ntimes, time_file):
        
    # Load first snapshot
    snap = fns[0][-8:-5]
    dendro_file = run+'_snapshot_'+snap+'_min_val1e3_res1e-3.fits' 
    dendro = Dendrogram.load_from(dendro_file)
    ids = load_data_ids(fns[0], res_limit=res_limit)
    print("*** Starting from snapshot ", snap)

    # Load density peak locations
    corefile = glob.glob(run+'_snapshot_*_'+snap+'_prop_v2.csv')
    coredata = read_properties(corefile)
    centpos_snap = coredata['Center Position [pc]'].values
    print("centpos =", centpos_snap)

    # Loop through snapshots
    for j,snapshot in enumerate(fns[:-1]):
        nodes_edges = []
        next_snap = fns[j+1][-8:-5]
        next_dendro_file = run+'_snapshot_'+next_snap+'_min_val1e3_res1e-3.fits' 
        next_dendro = Dendrogram.load_from(next_dendro_file)
        next_ids = load_data_ids(fns[j+1],res_limit=res_limit)

        # Load density peak locations
        corefile_next = glob.glob(run+'_snapshot_*'+next_snap+'_prop_v2.csv')
        coredata_next = read_properties(corefile_next)
        centpos_nextsnap = coredata_next['Center Position [pc]'].values

        print(" --- ", next_snap)
        out_nodefile = 'nodes_edges_'+run+'_'+snap+'_'+next_snap+'.csv'

        leaves = dendro.leaves
            
        next_leaves = next_dendro.leaves
        sz_next_leaves = len(next_leaves)

        # Store ids of particles in each leaf
        next_leaves_indices = np.zeros(shape=(sz_next_leaves,), dtype=np.ndarray)
        for m,leaf in enumerate(next_leaves):
            # Get the ids of the particles in this leaf 
            next_leaves_indices[m] = next_ids[leaf.get_mask()]

        # Get location of current leaf in the leaf_history
        for thisleafidx, leaf in enumerate(leaves):

            # Number of cells in common with each leaf in next snapshot
            overlap = np.zeros(sz_next_leaves) 

            # Fraction of current leaf in common
            fraction_leaf = np.zeros(sz_next_leaves)

            # Fraction of next leaf in common
            fraction_nextleaf = np.zeros(sz_next_leaves)        

            # Get center pos for current leaf
            print("centpost =", centpos_snap)
            pos = centpos_snap[thisleafidx]
            print("pos =", pos)

            for m in range(0,sz_next_leaves):
            
                nextpos = centpos_nextsnap[m]
                diffpos = np.sqrt(np.sum((pos - nextpos)**2, axis=0))

                # If central position between leaves is more than 1 pc 
                # continue to next leaf
                if diffpos > search_radius:
                    continue

                print("nextsnap pos, nextpos, diff", next_snap, pos, nextpos, diffpos)
                set = [i for i in ids[leaf.get_mask()] if i in next_leaves_indices[m]]
                if set:
                    overlap[m] = len(set)
                    fraction_leaf[m] = overlap[m]/len(ids[leaf.get_mask()])
                    fraction_nextleaf[m] = overlap[m]/len(next_leaves_indices[m])
            # Match where more than half of cells match for either snapshot
            match_ind = np.where(np.logical_or(fraction_leaf > 0.5, fraction_nextleaf > 0.5))[0]
            
            # Add to end of appropriate row
            if len(match_ind) > 0:
                for i in match_ind:
                    nodes_edges.append([str("%05i"%(int(snap)))+str(leaf.idx), str("%05i"%(int(next_snap)))+str(next_leaves[i].idx)])
                print('Match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind])
               
            else:
                print('No match found for leaf ', leaf.idx, fraction_leaf[match_ind], fraction_nextleaf[match_ind])

        nodes_edges_df = pd.DataFrame(nodes_edges)
        nodes_edges_df.to_csv(out_nodefile, index=False, header=False)
        print("Saved nodes to ", out_nodefile)

        dendro = next_dendro
        ids = next_ids
        snap = next_snap
        centpos_snap = centpos_nextsnap

        del next_dendro
        del next_ids
        gc.collect()

    return


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
        snap = fns[i][-8:-5].replace("_","")
        dendro_file = run+'_snapshot_'+snap+'_min_val1e3_res1e-3.fits' 
        dendro = Dendrogram.load_from(dendro_file)

        x,den = load_pos_den(fns[i],res_limit=res_limit)
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

# Helper function to find common items   
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        common = list(a_set & b_set)
    else:
        common = []

    return common
