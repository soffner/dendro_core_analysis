from astropy.io import fits
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import yt

# TO EDIT

loadfile = '/u/kneralwar/ptmp_link/dendrogram_cores_starforge/M2e4a2_fiducial/_summary_files/M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_all_prop_v2.pkl' #the file with core properties
savefile = '/u/kneralwar/ptmp_link/dendrogram_cores_starforge/M2e4a2_fiducial/_summary_files/M2e4a2_sne_saved.pkl' #save path and filename
data_path = '/u/kneralwar/ptmp_link/dendrogram_cores_starforge/M2e4a2_fiducial/_dendrograms/' #where the dendrogram for each folder are stored
hd_path = '/u/kneralwar/ptmp_link/starforge/M2e4_alpha2_fiducial/' #hdf5 files path

with open(loadfile, 'rb') as f:
    data = pd.read_pickle(f)

for snap in range(397,455):
    snap1 = str(snap).zfill(3)    
    indx_map = fits.getdata(data_path + 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_snapshot_'+snap1+'_min_val1e3_res1e-3.fits', 2) #need to update for different simulations


    ## 2

    snapshot = (data['ID']).str[:5].astype(int)
    core_prop = data[snapshot == snap]

    ## 3
    
    ds = yt.load(hd_path + 'snapshot_' + str(snap).zfill(3) +'.hdf5', unit_base = {'UnitMagneticField_in_gauss':  1e+4,
                 'UnitLength_in_cm'         : 3.08568e+18,
                 'UnitMass_in_g'            :   1.989e+33,
                 'UnitVelocity_in_cm_per_s' :      100})

    ad = ds.all_data()

    m11 = ad['PartType0', 'Metallicity_11']
    m12 = ad['PartType0', 'Metallicity_12']
    m13 = ad['PartType0', 'Metallicity_13']

    m_req = m13/(1-m11-m12)


    ## 4

    df = pd.DataFrame({'indx_map' : indx_map,
                      'm_req' : m_req,})

    indx_map_ids_mean = df.groupby(['indx_map']).mean()
    indx_map_ids_median = df.groupby(['indx_map']).median()

    char_cut = 5 #len(str(snap)) #cutting characters belonging to snapshot #5 because the snapshot 450 is 00450 in index list
    new_id1 = core_prop['ID'].astype(str).map(lambda x: x[char_cut:])
    new_id = new_id1.astype(int)

    m_req_mean = []
    m_req_median = []

    for value in new_id:
        m_req_mean.append(indx_map_ids_mean.m_req[value])
        m_req_median.append(indx_map_ids_median.m_req[value])

    data.loc[snapshot == snap, 'Median SNe Mass Frac'] = m_req_median
    data.loc[snapshot == snap, 'Mean SNe Mass Frac'] = m_req_mean

    # Append the combined DataFrame

    # Save the updated DataFrame to the pickle file

    print('Done: ', snap)
    
with open(savefile, 'wb') as f:
    pickle.dump(data, f)
