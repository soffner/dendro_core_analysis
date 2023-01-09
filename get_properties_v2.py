from scipy.spatial import cKDTree
from astrodendro import Dendrogram, pp_catalog
from astropy.io import fits
import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
import sys

np.set_printoptions(threshold=False)
from scipy import ndimage
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import os.path
import gc

import h5py
import glob
from scipy.spatial import cKDTree
import time
import copy as cp
import pytreegrav

def KE(xc, mc, vc, uc): 
    """ xc - array of positions 
        mc - array of masses 
        vc - array of velocities 
        uc - array of internal energies 
    """
    ## velocity w.r.t. com velocity of leaf
    v_bulk = np.average(vc, weights=mc,axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well**2,axis=1)
    return (mc*(vSqr/2 + uc)).sum()

def KEonly(xc, mc, vc, uc): 
    """ xc - array of positions 
        mc - array of masses 
        vc - array of velocities 
        uc - array of internal energies 
    """
    ## velocity w.r.t. com velocity of leaf
    v_bulk = np.average(vc, weights=mc,axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well**2,axis=1)
    return (mc*vSqr/2).sum()

def BE(mc, bc, rhoc):
    """ mc - array of masses 
        bc - array of magnetic field strengths
        rhoc - array of densities
    """
    ## magnetic energy; sum over volume
    bSqr = np.sum(np.sum(bc*bc, axis=1) * (mc/rhoc)) #unit_base['UnitLength_in_cm'])**3 *(unit_base['UnitMagneticField_in_gauss'])**2
    return bSqr.sum() / (8 * np.pi)

def PE(xc, mc, hc):
    """ xc - array of positions 
        mc - array of masses 
        hc - array of smoothing lengths 
        bc - array of magnetic field strengths
    """
    ## gravitational potential energy
    phic = pytreegrav.Potential(xc, mc, hc, G=4.301e4, theta=0.7) # G in code units
    return 0.5*(phic*mc).sum()

def get_evals(dxc):
    """ dxc - distance from center (density max) position 
    """
    ## Return eigenvalues and eigenvectors               
    evals, evecs = np.linalg.eig(np.cov(dxc.T)) # This seems very slow ...
    sort_mask = np.argsort(-evals)
    return evals[sort_mask],evecs[sort_mask]

def get_shape(dxc):
    """ dxc - distance from center (density max) position 
    """
    ## Return length of principle axes               
    evals, evecs = np.linalg.eig(np.cov(dxc.T)) # This seems very slow ...
    shape = np.array([np.max(evecs[0].T*dxc)-np.min(evecs[0].T*dxc),
                      np.max(evecs[1].T*dxc)-np.min(evecs[1].T*dxc),
                      np.max(evecs[2].T*dxc)-np.min(evecs[2].T*dxc)])              
    shape.sort()
    return shape

def load_data(file, res_limit = 0.0):

    # Load snapshot data
    f = h5py.File(file, 'r')

    # Mask to remove any cells with mass below some value 
    # (e.g., such as feedback cells) 
    mask = (f['PartType0']['Masses'][:] >= res_limit*0.999)

    den = f['PartType0']['Density'][:]*mask #* unit_base['UnitMass_in_g'] / unit_base['UnitLength_in_cm']**3 # Need to remember to convert to match dendro
    mask3d = np.array([mask, mask, mask]).T
    x = f['PartType0']['Coordinates']*mask3d
# * unit_base['UnitLength_in_cm'] # [pc]
    m = f['PartType0']['Masses'][:]*mask # * unit_base['UnitMass_in_g'] # [Msun]
    h = f['PartType0']['SmoothingLength'][:]*mask
    u = f['PartType0']['InternalEnergy'][:]*mask
    b = f['PartType0']['MagneticField'][:]*mask3d
    fmol = f['PartType0']['MolecularMassFraction'][:]*mask
    fneu = f['PartType0']['NeutralHydrogenAbundance'][:]*mask
    t = f['PartType0']['Temperature'][:]*mask
    v = f['PartType0']['Velocities']*mask3d
    print("Max/min temp =", np.max(t), np.min(t))
  
    if 'PartType5' in f.keys():
        partlist = f['PartType5']['Coordinates'][:]
        partmasses = f['PartType5']['Masses'][:]
        partvels = f['PartType5']['Velocities'][:]
        partids = f['PartType5']['ParticleIDs'][:]
    else:
        partlist = []
        partmasses = [0]
        partids = []
        partvels = [0, 0, 0]

    time = f['Header'].attrs['Time']
    #for att in f['Header'].attrs:
    #    print(att)
    unitlen = f['Header'].attrs['UnitLength_In_CGS']
    unitmass = f['Header'].attrs['UnitMass_In_CGS']
    unitvel = f['Header'].attrs['UnitVelocity_In_CGS']
    unitb = 1e4 #f['Header'].attrs['UnitMagneticField_In_CGS'] #Not defined

    tcgs = time*(unitlen/unitvel)/(3600.0*24.0*365.0*1e6)
    unit_base = {'UnitLength' : unitlen, 'UnitMass': unitmass, 'UnitVel': unitvel, 'UnitB': unitb}

    del f
    return den, x, m, h, u, b, v, t, fmol, fneu, partlist, partmasses, partvels, partids, tcgs, unit_base

def calc_nh2(den, fmol, fneu, unit_base, helium_mass_fraction=0.284, mh=1.67e-24):
    ## Calculate n_H2
    return unit_base['UnitMass']/unit_base['UnitLength']**3*den*fmol*fneu*(1-helium_mass_fraction)/(2.0*mh)
    
def get_leaf_properties(dendro, den, x, m, h, u, b, v, t, snapshot_no, partlist, partmasses, partvels, partids, veltol=1730, postol=0.043):
    """ dendro - dendrogram
        den, c, m, h, u, b, v - snapshot density, position, masses, smoothing
             length, magnetic field, velocities
        partlist, partmasses, partvels, partids - star particle coordines, masses, velocities, ids
        veltol -- maximum difference between core bulk velocity and star velocity to save sinks
        postol -- maximum difference between density maximum and star position to count as protostellar
    """
    ## Return properties of all dendrogram leaves
    ## Note: input/output values are all in code units by default

    # Initialize data arrays
    leaf_masses = [] 
    leaf_maxden = [] 
    leaf_centidx = [] 
    leaf_centpos = [] 
    leaf_vdisp = []    
    leaf_vbulk = []   
    leaf_shape = []   
    leaf_halfmass = [] 
    leaf_reff = []  
    leaf_bmean = [] 
    leaf_mage = []  
    leaf_ke = []      
    leaf_grave = [] 
    leaf_sink = []   
    leaf_sinkallm = [] 
    leaf_sinkallid = [] 
    leaf_protostellar = []
    leaf_id = []  
    leaf_cs = []      # Mass-wieghted temperature
    leaf_keonly = []  # COM KE without thermal component
    csconst = 0.188e3 # T=10 K [m/s], currently not used
    kb = 1.38e-16
    mh = 1.67e-24

    # Loop through all leaves in the dendrogram
    for leaf in dendro:
        if leaf.is_leaf:
            print("Analyzing leaf ", leaf, " size", leaf.get_npix())
            mask = leaf.get_mask()
            mass = np.sum(m[mask])
            leaf_masses.append(mass)    # code units [msun]
            idx, maxd = leaf.get_peak() # [cm^-3]
            leaf_id.append(str("%05i")%(int(snapshot_no)) + str(leaf.idx))
            leaf_centidx.append(idx)
            leaf_centpos.append(x[idx]) # code units [pc]
            leaf_maxden.append(maxd)
        
            # Get size information
            dx = x[mask]-x[idx]
           
            shape = get_shape(dx)
            leaf_shape.append(shape)
                  
            r = np.sqrt(np.sum(dx**2,axis=1)) # code units [pc]
            leaf_halfmass.append(np.median(r)) # code units [pc]
            leaf_reff.append(np.sqrt(5./3 * np.average(r**2,weights=m[mask]))) # code units [pc] 

            # Get velocity, ke information
            v = np.array(v)
            #mask3d = np.column_stack((mask, mask, mask))
            
            v_bulk = np.average(v[mask,:], weights=m[mask],axis=0)
            v_well = v[mask,:] - v_bulk
            #print('vbulk = ', v_bulk)
            vSqr = np.sum(v_well**2,axis=1)
            leaf_vdisp.append(np.sqrt((m[mask]*vSqr).sum()/mass)) # [m/s]
            leaf_vbulk.append(v_bulk)  # code units [m/s]
            leaf_ke.append((m[mask]*(vSqr/2 + u[mask])).sum()) # code units [Msun m^2/s^2]
            leaf_keonly.append((m[mask]*(vSqr/2 + u[mask])).sum()) # code units [Msun m^2/s^2]
            
            # Get temperature          
            cs = np.sqrt(kb*np.sum(m[mask]*t[mask])/mass/(2.33*mh))/1e5
            #leaf_cs.append(csconst) #cg
            leaf_cs.append(cs)             
     
            # Get grav, magnetic info
            leaf_grave.append(PE(x[mask], m[mask], h[mask]))
            leaf_bmean.append(np.sqrt(np.average(b[mask,:]*b[mask,:], weights=m[mask],axis=0))) #*unit_base["UnitMagneticField_in_gauss"]
            leaf_mage.append(BE(m[mask], b[mask], den[mask])) #code units
            
            # Statistics for sinks contained within the leaf boundary
            sinkm = []
            sinkids = []
            numsinks = 0
            numproto = 0        
            if np.any(partlist):
                for loc, s in enumerate(partlist):
                    minx = np.array([np.min(x[mask,0]), np.min(x[mask,1]), np.min(x[mask,2])])
                    maxx = np.array([np.max(x[mask,0]), np.max(x[mask,1]), np.max(x[mask,2])])
                    if np.sum(s < maxx) + np.sum(s > minx) == 6:
                        diffv = np.sqrt(np.sum((v_bulk - partvels[loc])**2, axis=0))
                        # Velocity difference check
                        #print("Vel diff [m/s]= ", diffv)
                        if diffv < veltol:
                            sinkm.append(partmasses[loc])
                            sinkids.append(partids[loc])
                            numsinks += 1
                        # Spatial position relative to peak density
                        diffp = np.sqrt(np.sum((x[idx]- partlist[loc])**2, axis=0))
                        #print("Diff p =", diffp, x[idx], x[idx]-partlist[loc])
                        if diffp < postol:
                           numproto += 1
                    
        
            leaf_sink.append(numsinks)  # N sink inside core boundary
            leaf_sinkallm.append(sinkm) # Save sink masses
            leaf_sinkallid.append(sinkids) # Keep sink ids
            leaf_protostellar.append(numproto) # N sink close to density peak
        
    return leaf_masses, leaf_maxden, leaf_centidx, leaf_centpos, leaf_vdisp, leaf_vbulk, leaf_shape, leaf_halfmass, leaf_reff, leaf_bmean, leaf_mage, leaf_ke, leaf_grave, leaf_sink, leaf_sinkallm, leaf_sinkallid, leaf_protostellar, leaf_id, leaf_cs, leaf_keonly
    
def load_dendrogram(dendro_file, nh2, x, num):
    """  dendro_file - dendrogram location saved or to save
    """
    ## Make/Load Dendrogram
    if os.path.isfile(dendro_file):
        print("Found dendrogram", dendro_file)
        d = Dendrogram.load_from(dendro_file)
    else:
        print("Warning: Need to make the fix to use kDTree to run dendrogram on Mac.") # See git repo branch, frontera/_yt_scripts/test_kdtree.py     

        print("Constructing dendrogram... Saving to", dendro_file)
        start = time.time()
        d = Dendrogram.compute(nh2, min_value=1e3, min_delta=1e4, min_npix=100,pos=x) 
        end = time.time()
        
        print("Done computing dendogram .... Time to make dendrogram [min] is", (end-start)/60.0)
        d.save_to(dendro_file)
    
    print("Number of leaves is", len(d.leaves))

    fig = plt.figure(figsize=(15,5))
    p2 = d.plotter()

    ax2 = fig.add_subplot(2, 1, 1)
    p2.plot_tree(ax2, color='black')
    ax2.hlines(1e4, *ax2.get_xlim(), color='b', linestyle='--')
    ax2.hlines(2e4, *ax2.get_xlim(), color='b', linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xlabel("Structure")
    ax2.set_ylabel("Flux")
    ax2.set_title(dendro_file+" Dendrogram")
    plt.savefig("Dendrogram_"+num+".png")
    
    return d


def calc_profiles(dendro, nh2, x, v, nbin, num, maxsize=0.5, plotleaf=False, saveall=False):
    """
    dendro - leaf structure
    nh2  -  derived number density (cgs)
    x, v, shape - pos,vel, semi-major axes length (code units)
    maxsize - max size for profile (code units)
    plotleaf - make a plot of each leaf
    saveall - save all the densities included in the profile for debugging
    """
    
    ## Calculate profiles from leaf structures
    leaves = dendro.leaves
    loop_length = len(leaves)
    radii_eq = np.zeros((loop_length, len(nbin)))
    veldisp = np.zeros((loop_length, len(nbin)))
    n_part = np.zeros((loop_length, len(nbin)))
    radii_eq[radii_eq == 0] = 'nan'

    allden = []   # Store all relevant densities if saveall=True
    allradii = [] # Store all relevant radii if saveall=True

    # Loop over each leaf
    for i in range(loop_length): #get leaf, find profiles
        
        print("Leaf "+str(i)+" of "+str(loop_length))
        center = leaves[i].get_peak()[0] #find center
        self_id = leaves[i].idx #identify leaf of focus
        
        #initialize mask       
        mask = leaves[i].get_mask()
        if plotleaf:
            fig, ax = plt.subplots()
            ax.scatter(x[mask,0], x[mask,1], alpha=0.1,s=1)
            
        leafden = nh2[mask]
        minleaf = np.min(leafden) # Min density in leaf

        dx = x-x[center]   #location of peak in indicy
        r = np.sqrt(np.sum(dx**2, axis=1))
        masksz = (r < maxsize) 
        
        # cap all other densities at minimum of target leaf
        leaf_nh2 = cp.deepcopy(nh2)
        leaf_nh2[leaf_nh2 > minleaf] = minleaf 
        leaf_nh2[mask] = nh2[mask]

        # Crop data arrays
        leaf_nh2 = leaf_nh2[masksz] 
        r = r[masksz]
        v = np.array(v)
        smv = v[masksz,:]
        
        if plotleaf:
            plt.savefig('Leaf_'+num+'%i.png' %i)
            plt.close()
            
        if saveall:
            allden.append(leaf_nh2[mask2])  #To check whether density distribution matches
            allradii.append(r)

        for j in range(len(nbin)):  
            ind = (leaf_nh2 > nbin[j])
            if len(leaf_nh2[ind]) > 1:
                #print(" nbin, ", j)
                mass = np.sum(leaf_nh2[ind])
                v_com = np.average(smv[ind,:], weights=leaf_nh2[ind],axis=0)
                v_well = smv[ind,:] - v_com
                vSqr = np.sum(v_well**2,axis=1)
                veldisp[i,j] = np.sqrt((leaf_nh2[ind]*vSqr).sum()/mass) # [m/s]
                radii_eq[i,j] = np.sqrt(5./3 * np.average(r[ind]**2)) #,weights=leaf_nh2[mask2][ind])) #np.cbrt(vol_iso)
                n_part[i,j] = len(r[ind])

    # allden, allradii will be empty unless saveall == True
    return veldisp, radii_eq, n_part, allden, allradii


# This original method searches the full tree
def calc_profiles_slow(dendro, nh2, x, v, nbin, maxsize=0.5, plotleaf=False, saveall=False):
    """
    dendro - leaf structure
    nh2  -  derived number density (cgs)
    x, v, evals - semi-major axes, pos, vel (code units)
    maxsize - max size for profile (code units)
    plotleaf - make a plot of each leaf
    saveall - save all the densities included in the profile for debugging
    """
    
    ## Calculate profiles from leaf structures
    leaves = dendro.leaves
    loop_length = len(leaves)
    radii_eq = np.zeros((loop_length, len(nbin)))
    veldisp = np.zeros((loop_length, len(nbin)))
    n_part = np.zeros((loop_length, len(nbin)))
    radii_eq[radii_eq == 0] = 'nan'

    allden = []   # Store all relevant densities if saveall=True
    allradii = [] # Store all relevant radii if saveall=True

    # Loop over each leaf
    for i in range(loop_length): #get leaf, find profiles
        
        print("Leaf "+str(i)+" of "+str(loop_length))
        center = leaves[i].get_peak()[0] #find center
        self_id = leaves[i].idx #identify leaf of focus
        
        #initialize parent and mask
        ancestors_idx = []
        parent = leaves[i].parent
        
        mask = leaves[i].get_mask()
        if plotleaf:
            fig, ax = plt.subplots()
            ax.scatter(x[mask,0], x[mask,1], alpha=0.1,s=1)
            
        leafden = nh2[mask]
        minleaf = np.min(leafden) # Min density in leaf
        
        if parent is not None:
            ancestors_idx.append(parent.idx)

        leaf_nh2 = cp.deepcopy(nh2)
        # Impt: cap all other densities above this leaf mindensity
        leaf_nh2[leaf_nh2 > minleaf] = 0.0 # minleaf, 0 = mask out 
        leaf_nh2[mask] = nh2[mask]
        
        # loop through ancestral substructures to identify masked coordinates
        while parent is not None: # while there is a parent...
            
            for child in parent.children: # iterate through its children
                
                if child.is_leaf:
                    if child.idx is not self_id: #check if child is a leaf and not the self
                        #print("condition 1")
                        mask2 = child.get_mask()
                        mask = np.logical_or(mask, mask2)
                        if plotleaf:
                            ax.scatter(x[mask2,0], x[mask2,1], alpha=0.1,s=1)
                       
                    #else:
                        #print("This is the self leaf")
                elif child.is_branch and (child.idx not in ancestors_idx):
                    #print("condition 2")
                    #branch values include their substructures, so can chop at branch "base".
                    mask2 = child.get_mask()
                    mask = np.logical_or(mask, mask2)
                    
                    if plotleaf:
                        ax.scatter(x[mask2,0], x[mask2,1], alpha=0.1,s=1)
    
            parent = parent.parent #step up to the next parent structure
            if parent is not None:
                ancestors_idx.append(parent.idx)

        if plotleaf:
            plt.savefig('Leaf_%i.png' %i)
            plt.close()
            
        #print("Finished profile computation")
        dx = x-x[center]   #location of peak in indicy
        r = np.sum(dx**2, axis=1)**0.5
        
        mask2 = np.logical_and(mask, r < maxsize)  # Truncate leaf 
        dx = x[mask2]-x[center]
        r = np.sum(dx**2, axis=1)**0.5
        
        if saveall:
            allden.append(leaf_nh2[mask2])  #To check whether density distribution matches
            allradii.append(r)

        #fig, ax = plt.subplots()
        for j in range(len(nbin)):
            print(" nbin ", j)
            ind = (leaf_nh2[mask2] > nbin[j])
            if len(r[ind]) > 3:          
                # Note nh2 is just a weighting factor;
                # it doesn't need to be in the same units as velocity
                mass = np.sum(leaf_nh2[mask2][ind])
                v_com = np.average(v[mask2,:][ind], axis=0)
                v_well = v[mask2,:][ind] - v_com
                vSqr = np.sum(v_well**2,axis=1)
                veldisp[i,j] = np.sqrt((leaf_nh2[mask2][ind]*vSqr).sum()/mass)
                #vol_iso = (evals[i])[0]*(evals[i])[1]*(evals[i])[2]
                radii_eq[i,j] = np.sqrt(5./3 * np.average(r[ind]**2)) #,weights=leaf_nh2[mask2][ind])) #np.cbrt(vol_iso)
                n_part[i,j] = len(r[ind])
               
        plt.close()

    # allden, allradii will be empty unless saveall == True
    return veldisp, radii_eq, n_part, allden, allradii


def interpolate_profiles(nbin, veldisp, radii, leaf_cs, num, maxsize=0.5):
    """ nbin - array of number densities (n_h2)
        veldisp - velocity dispersion profiles
        radii - radii profiles
        leaf_cs - mean leaf sound speed
    """
    ## Interpolate the profiles onto a common size grid
    

    # grid used for the interpolation of density, vdisp profiles
    # Changed from Offner et al. 2022, since STARFORGE cores tend to be more compact and denser
    size_grid = np.log10(np.logspace(np.log10(2e-3), np.log10(maxsize/2.0), len(nbin)))
    i_density_interp = []
    i_veldisp_interp = []
    leaf_rpow = []
    leaf_rcoh = []

    for i in range(len(leaf_cs)):
        i_size = np.log10(radii[i][::-1])
        i_density = np.log10(nbin[::-1])
        veldisp[i][::-1][veldisp[i][::-1]<1] = 0
        i_veldisp = np.log10(veldisp[i][::-1]) #Convert to km/s
        mask_clean = np.isfinite(i_size)&np.isfinite(i_density)&np.isfinite(i_veldisp)
        sortind = i_size[mask_clean].argsort()
        i_density_interp.append(np.interp(size_grid, (i_size[mask_clean]), (i_density[mask_clean])))
        #print(" ?", i_size[mask_clean])
        i_veldisp_interp.append(np.interp(size_grid, (i_size[mask_clean]), (i_veldisp[mask_clean])))
        idx = np.where(np.array(i_veldisp_interp[i]) < np.log10(leaf_cs[i]))[0] 
        if idx.size > 0: 
            leaf_rcoh.append(10**size_grid[np.max(idx)])
        else:
            leaf_rcoh.append(0.0)

        # fit the profile: ~0.03 -0.1
        #poly = np.polyfit(size_grid[2:17],i_density_interp[i][2:17], 1, rcond=None, full=False, w=None, cov=False) #1 = Degree of fit

        # Fit the raw data and only where it's defined
        rtmp = i_size[mask_clean]
        rhotmp = i_density[mask_clean]
        
        ind = np.logical_and(rtmp > np.log10(0.0025), rtmp < np.log10(0.1))
        if len(rtmp[ind]) > 2:
            poly = np.polyfit(rtmp[ind], rhotmp[ind], 1, rcond=None, full=False, w=None, cov=False)
            leaf_rpow.append(np.min([0.0, poly[0]]))
        else:
            leaf_rpow.append(0.0)

    # Plot interpolated profiles
    fig, ax = plt.subplots()
    for i in range(len(leaf_cs)):
        ax.plot(10**size_grid, 10**i_density_interp[i], alpha=0.1)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.001, maxsize/2.0])
    plt.savefig("Interpolated_Densities_"+num+".png")


    fig, ax = plt.subplots()

    for i in range(len(leaf_cs)):
        ax.plot(10**size_grid, 10**i_veldisp_interp[i], alpha=0.1)

    ax.plot(np.array([1e-3,maxsize/2.0]),np.array([0.188e3, 0.188e3]), color='black')                
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([0.001, 2e-1])
    plt.savefig("Interpolated_veldisp_"+num+".png")

    return i_density_interp, i_veldisp_interp, size_grid, leaf_rcoh, leaf_rpow
