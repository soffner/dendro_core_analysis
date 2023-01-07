import numpy as np
import pandas as pd
from get_properties_v2 import *
import gc
import glob
import matplotlib
matplotlib.use('Agg')

# Location of STARFORGE outputs
dir  = '/scratch3/03532/mgrudic/STARFORGE_RT/production2/M2e3_R3/M2e3_R3_S0_T1_B0.01_Res126_n2_sol0.5_42/output/'

# Run name
run = 'M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42'
outdir = './'

nbin = np.logspace(3.5,7.0, 20)  # n_H2 number density cuts for profiles
maxsize = 0.4 # Max size used to construct profile (code units)

res_limit = 1e-3 # Only consider cells with masses at or above min resolution

#unit_base = {'UnitMagneticField_in_gauss':  1e+4,
#             'UnitLength_in_cm'         : 3.08568e+18,
#             'UnitMass_in_g'            :   1.989e+33,
#             'UnitVelocity_in_cm_per_s' :      100}

fns = glob.glob(dir+'snapshot_*.hdf5')
fns.sort()

for fn in fns:
    
    snap = fn[-8:-5] # File number
    #file = dir+snap # Input file
    dendro_file = outdir+run+'_snapshot_'+snap+'_min_val1e3_res1e-3.fits' # min_value = 1e3 [cm^-3]

    print("Loading data from snapshot %s..." %fn)
    den, x, m, h, u, b, v, t, fmol, fneu, partlist, partmasses, partvels, partids, tcgs,  unit_base = load_data(fn, res_limit=res_limit)
    prop_file=outdir+run+'_snapshot_%1.3f_'%tcgs+snap+'_prop_v6.csv'
    print("Run Time =",tcgs)

    print("Calculating nH2...")
    nh2 = calc_nh2(den, fmol, fneu, unit_base)
    del fmol
    del fneu
    gc.collect()

    print("Computing or Loading dendrogram from %s..." %dendro_file)
    start = time.time()
    dendro = load_dendrogram(dendro_file, nh2, x, snap)
    print(" Complete:", (time.time() - start)/60.)

    print("Computing bulk properties for each leaf ...")
    start = time.time()
    leaf_masses, leaf_maxden, leaf_centidx, leaf_centpos, leaf_vdisp, leaf_vbulk, leaf_shape, leaf_halfmass, leaf_reff, leaf_bmean, leaf_mage, leaf_ke, leaf_grave, leaf_sink, leaf_sinkallm, leaf_sinkallid, leaf_protostellar, leaf_ids, leaf_cs, leaf_keonly = get_leaf_properties(dendro, den, x, m, h, u, b, v, t, snap, partlist, partmasses, partvels, partids)
    print(" Complete:", (time.time() - start)/60.)

    print("Computing profiles, including info outside leaf ...")
    # all* will be empty unless saveall = True
    start = time.time()
    veldisp, radii, n_part, allden, allradii = calc_profiles(dendro, nh2, x, v, nbin, snap, maxsize=maxsize)
    print(" Complete:", (time.time() - start)/60.)


    print("Interpolating profiles onto a common grid, fiting for rho powerlaw index, determining radius of coherence...")
    start = time.time()
    i_density_interp, i_veldisp_interp, size_grid, leaf_rcoh, leaf_rpow = interpolate_profiles(nbin, veldisp, radii, leaf_cs, snap, maxsize=maxsize)
    print(" Complete:", (time.time() - start)/60.)

    print("Combining and saving information in %s ..." %prop_file)
    leaf_count = len(leaf_masses) # Here just one snapshot
    combined_all = np.empty(shape=(leaf_count,23), dtype=object)
    for leaf in range(leaf_count):
        combined_all[leaf][0] = leaf_ids[leaf] #
        combined_all[leaf][1] = i_density_interp[leaf]    # interpolated log density
        combined_all[leaf][2] = np.array(i_veldisp_interp[leaf])*unit_base['UnitVel']    # interpolated log velocity Note:  compare with droplets, need to convert to sigtot = (signt^2 + cs^2)**0.5
        combined_all[leaf][3] = leaf_reff[leaf]   
        combined_all[leaf][4] = leaf_vdisp[leaf]*unit_base['UnitVel']   
        combined_all[leaf][5] = leaf_masses[leaf]  
        combined_all[leaf][6] = leaf_rcoh[leaf]   
        combined_all[leaf][7] = leaf_rpow[leaf]
        combined_all[leaf][8] = leaf_vbulk[leaf]*unit_base['UnitVel'] 
        combined_all[leaf][9] = leaf_centpos[leaf]
        combined_all[leaf][10] = leaf_centidx[leaf][0]
        combined_all[leaf][11] = leaf_ke[leaf]*unit_base['UnitMass']*unit_base['UnitVel']**2
        combined_all[leaf][12] = leaf_grave[leaf]*unit_base['UnitMass']*unit_base['UnitVel']**2
        combined_all[leaf][13] = leaf_sink[leaf]
        combined_all[leaf][14] = leaf_maxden[leaf]
        combined_all[leaf][15] = leaf_shape[leaf]
        combined_all[leaf][16] = leaf_halfmass[leaf]
        combined_all[leaf][17] = leaf_bmean[leaf]*unit_base['UnitB']
        combined_all[leaf][18] = leaf_mage[leaf]*unit_base['UnitB']**2*unit_base['UnitLength']**3
        combined_all[leaf][19] = leaf_sinkallm[leaf]
        combined_all[leaf][20] = leaf_cs[leaf]
        combined_all[leaf][21] = leaf_keonly[leaf]
        combined_all[leaf][22] = leaf_protostellar[leaf]

    np.set_printoptions(threshold=sys.maxsize)
    save_df = pd.DataFrame(combined_all, columns = ['ID','Density [cm^-3]','Dispersion [cm/s]',
                                                    'Reff [pc]', 'LeafDisp [cm/s]',
                                                    'LeafMass [msun]', 'CoherentRadius [pc]',
                                                'DensityIndex','V Bulk [cm/s]',
                                                'Center Position [pc]','Center index','LeafKe',
                                                 'LeafGrav', 'Num. Sinks',
                                                'Max Den [cm^-3]', 'Shape [pc]',
                                                'Half Mass R[pc]', 'Mean B [G]',
                                                'Mag. Energy', 'Sink Masses [Msun]',
                                                    'Sound speed [cm/s]', 'LeafKe only', 'Protostellar'])
    print("Saving cores in ", prop_file)
    print("Total number of leaves = ", leaf_count)
    save_df.to_csv(prop_file,index=False)

    del den,x,v,h,u,t,m,b
    gc.collect()
