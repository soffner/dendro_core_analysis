import numpy as np
import pandas as pd
from get_properties_v0 import *
import gc

file = 'M2e3_mid'
snapshot = 'temp' # Temporary file number
outdir = ''
dendro_file = file+'_min_val1e2.fits' # min_value = 1e2
prop_file = file+'_prop.csv'
nbin = 10.0

nbin = np.logspace(3.5,5.5, 10)  # n_H2 number density cuts for profiles
maxsize = 0.5 # Max size used to construct profile (code units)

unit_base = {'UnitMagneticField_in_gauss':  1e+4,
             'UnitLength_in_cm'         : 3.08568e+18,
             'UnitMass_in_g'            :   1.989e+33,
             'UnitVelocity_in_cm_per_s' :      100}

print("Loading data from snapshot %s..." %file)
den, x, m, h, u, b, v, fmol, fneu, partlist, partmasses, partids = load_data(file, unit_base)

print("Calculating nH2...")
nh2 = calc_nh2(den, fmol, fneu, unit_base)
del fmol
del fneu
gc.collect()

print("Computing or Loading dendrogram from %s..." %dendro_file)
start = time.time()
dendro = load_dendrogram(dendro_file, nh2)
print(" Complete:", (time.time() - start)/60.)

print("Computing bulk properties for each leaf ...")
start = time.time()
leaf_masses, leaf_maxden, leaf_centidx, leaf_centpos, leaf_vdisp, leaf_vbulk, leaf_evals, leaf_evecs, leaf_halfmass, leaf_reff, leaf_bmean, leaf_mage, leaf_ke, leaf_grave, leaf_sink, leaf_sinkallm, leaf_sinkallid, leaf_ids, leaf_cs = get_leaf_properties(dendro, den, x, m, h, u, b, v, snapshot, partlist, partmasses, partids)
print(" Complete:", (time.time() - start)/60.)


print("Computing profiles, including info outside leaf ...")
# all* will be empty unless saveall = True
start = time.time()
veldisp, radii, n_part, allden, allradii = calc_profiles(dendro, nh2, x, v, nbin, maxsize=maxsize)
print(" Complete:", (time.time() - start)/60.)


print("Interpolating profiles onto a common grid, fiting for rho powerlaw index, determining radius of coherence...")
start = time.time()
i_density_interp, i_veldisp_interp, size_grid, leaf_rcoh, leaf_rpow = interpolate_profiles(nbin, veldisp, radii, leaf_cs)
print(" Complete:", (time.time() - start)/60.)

print("Combining and saving information in %s ..." %prop_file)
leaf_count = len(leaf_masses) # Here just one snapshot
combined_all = np.empty(shape=(leaf_count,22), dtype=object)
for leaf in range(leaf_count):
    combined_all[leaf][0] = leaf_ids[leaf] #
    combined_all[leaf][1] = i_density_interp[leaf]    # interpolated log density
    combined_all[leaf][2] = i_veldisp_interp[leaf]    # interpolated log velocity Note:  compare with droplets, need to convert to sigtot = (signt^2 + cs^2)**0.5
    combined_all[leaf][3] = leaf_reff[leaf]   
    combined_all[leaf][4] = leaf_vdisp[leaf]  
    combined_all[leaf][5] = leaf_masses[leaf]  
    combined_all[leaf][6] = leaf_rcoh[leaf]   
    combined_all[leaf][7] = leaf_rpow[leaf]
    combined_all[leaf][8] = leaf_vbulk[leaf]
    combined_all[leaf][9] = leaf_centpos[leaf]
    combined_all[leaf][10] = leaf_centidx[leaf][0]
    combined_all[leaf][11] = leaf_ke[leaf]
    #combined_all[leaf][12] = leaf_grave[leaf]
    combined_all[leaf][13] = leaf_sink[leaf]
    combined_all[leaf][14] = leaf_maxden[leaf]
    combined_all[leaf][15] = leaf_evals[leaf]
    combined_all[leaf][16] = leaf_evecs[leaf]
    combined_all[leaf][17] = leaf_halfmass[leaf]
    combined_all[leaf][18] = leaf_bmean[leaf]
    combined_all[leaf][19] = leaf_mage[leaf]
    combined_all[leaf][20] = leaf_sinkallm[leaf]
    combined_all[leaf][21] = leaf_cs[leaf]

np.set_printoptions(threshold=sys.maxsize)
save_df = pd.DataFrame(combined_all, columns = ['ID','Density [cm^-3]','Dispersion [m/s]', 'Reff [pc]', 'LeafDisp [m/s]', 'LeafMass [msun]', 'CoherentRadius [pc]',
                                                'DensityIndex','V Bulk [m/s]','Center Position [pc]','Center index','LeafKe', 'LeafGrav', 'Num. Sinks',
                                                'Max Den [cm^-3]', 'Eigenvals [pc]', 'Eigenvecs', 'Half Mass R[pc]', 'Mean B [T]', 'Mag. Energy', 'Sink Masses [Msun]',
                                                'Sound speed [m/s]'])
print("Saving cores in ", prop_file)
print("Total number of leaves = ", leaf_count)
save_df.to_csv(prop_file,index=False)
