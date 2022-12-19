import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
#from natsort import natsorted # not compatible with current setup 

#----------------------------------------------------------
# Code to concatenate property files and generate time file
#---------------------------------------------------------

unit_base = {'UnitMagneticField_in_gauss':  1e+4,
             'UnitLength_in_cm'         : 3.08568e+18,
             'UnitMass_in_g'            :   1.989e+33,
             'UnitVelocity_in_cm_per_s' :      100}


# Read in Data From Folder of csv files

tag = '_M2e3_cores_v4/M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_snapshot_*_prop_v4.csv'
file_save = '_M2e3_cores_v4/M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_all_prop_v4.csv'
time_save = '_M2e3_cores_v4/M2e3_R3_S0-T1_B0.01_Res126_n2_sol0.5_42_times_v4.csv'

fns = glob.glob(tag)
fns.sort() # Will need natsort for snapshot > 1000 (unless it sorts on time ...)

converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)

frames = []
times = []
snapshot_no = []  #Need to format snapshot numbers better for easier reading
for fn in fns:
    data = pd.read_csv(fn, 
                       converters = { 'ID': str,
                           'Radii': converter_nparray,
                                     'Density [cm^-3]': converter_nparray,
                                    'Dispersion [cm/s]': converter_nparray,
                                    'V Bulk [cm/s]': converter_nparray,
                                    'Center Position [pc]':converter_nparray,
                                    'Eigenvals [pc]':converter_nparray,
                                     'Mean B [G]':converter_nparray
                                    })
    frames.append(data)
    time = np.float(fn[-21:-16])
    times.append(time)
    
profiles = pd.concat(frames)


if 1: # Save all as a new data frame
    
    save_df = pd.DataFrame(profiles, columns = ['ID','Density [cm^-3]','Dispersion [cm/s]', 'Reff [pc]', 'LeafDisp [cm/s]', 'LeafMass [msun]', 'CoherentRadius [pc]',
                                                'DensityIndex','V Bulk [cm/s]','Center Position [pc]','Center index','LeafKe', 'LeafGrav', 'Num. Sinks',
                                                'Max Den [cm^-3]', 'Shape [pc]','Half Mass R[pc]', 'Mean B [G]', 'Mag. Energy', 'Sink Masses [Msun]',
                                                'Sound speed [cm/s]', 'Protostellar'])
    print("Saving cores in ", file_save)
    save_df.to_csv(file_save,index=False)

times = []
for fn in fns:
    time = fn[-21:-16]
    times.append(time)
    
df = pd.DataFrame({'Time [Myr]': times})
df.to_csv(time_save,index=False)

# Print some statistics
# Grab the arrays
#'ID','Density [cm^-3]','Dispersion [m/s]', 'Reff [pc]', 'LeafDisp [m/s]', 'LeafMass [msun]', 'CoherentRadius [pc]',
#'DensityIndex','V Bulk [m/s]','Center Position [pc]','Center index','LeafKe', 'LeafGrav', 'Num. Sinks',
#'Max Den [cm^-3]', 'Shape [pc]', 'Half Mass R[pc]', 'Mean B [T]', 'Mag. Energy', 'Sink Masses [Msun]',
#'Sound speed [cm/s]', 'Protostellar'

ID = profiles['ID'].values
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
lgrave = profiles['LeafGrav'].values #Still in code units
lmage = profiles['Mag. Energy'].values #Still in code units
lsink = profiles['Num. Sinks'].values
lsinkmass = profiles['Sink Masses [Msun]'].values
lsound = profiles['Sound speed [cm/s]'].values # in m/s
lproto = profiles['Protostellar'].values # in m/s
#lkeonly = profiles['LeafKe only'].values # in m/s

nleaves = len(lsound)
indsink = np.where(lsink > 0)[0]
indnosink = np.where(lsink == 0)[0]
sinkfrac = len(indsink)/nleaves
# Interpolate the profiles and store the results (in a format compatible with sklearn PCA)
## the common grid used for the interpolation
#size_grid = np.log10(np.logspace(np.log10(1.0e-2), np.log10(1.5e-1), 50))

lgrave = np.abs(lgrave)
lmeanv = []
lmeanb = []
for i in range(len(lsound)):
    lmeanv.append(np.sqrt(np.sum(lvbulk[i]**2)))
    lmeanb.append(np.sqrt(np.sum(lb[i]**2)))

lmeanb = np.array(lmeanb)
lmeanv = np.array(lmeanv)

np.set_printoptions(precision=2, suppress=2, threshold=2)

kms = 1e3
ug = 1e6 # G -> uG
print("Simulation Core statistics (%i cores):" %len(lmass))
print(" M (Msun)        = %5.3f %5.3f %5.3f" %(np.mean(lmass), np.median(lmass), np.std(lmass)))
print(" R (pc)          = %5.3f %5.3f %5.3f" %(np.mean(lradius), np.median(lradius), np.std(lradius) ))
print(" Max den (cm^-3) = %5.3f %5.3f %5.3f" %(np.mean(lmax), np.median(lmax), np.std(lmax) ))
print(" Half Max (pc)   = %5.3f %5.3f %5.3f" %(np.mean(lhalf), np.median(lhalf), np.std(lhalf) ))
print(" Mean B (uG)     = ", np.average(lb, axis=0)*ug), np.sqrt(np.sum(np.average(lb, axis=0)*ug,axis=0)**2)#, np.median(lb, axis=0)/uG) #, np.std(lb, axis=0)/ug )
print(" La,b,c (pc)          = %5.3f %5.3f %5.3f" %(np.average(lshape[:,0]), np.median(lshape[:,0]), np.std(lshape[:,0]) ))
#print(" Mean Brms (uG)     = %5.3f" %np.mean(lmeanb)*ug)
print(" 3D Vdisp (km/s) = %5.3f %5.3f %5.3f" %(np.mean(ldisp)/kms, np.median(ldisp)/kms, np.std(ldisp)/kms))
print(" Rcoh (pc)       = %5.3f %5.3f %5.3f" %(np.mean(rcoh), np.median(rcoh), np.std(rcoh)))
print(" Den Power Law   = %5.3f %5.3f %5.3f" %(np.mean(rhopow), np.median(rhopow), np.std(rhopow)))
print(" Sound Speed     = %5.3f %5.3f %5.3f" %(np.mean(lsound)/kms, np.median(lsound), np.std(lsound)/kms))
print("**")
print(" KE    = %5.3e %5.3e %5.3e" %(np.mean(lke), np.median(lke), np.std(lke)))
print(" Mage  = %5.3e %5.3e %5.3e" %(np.mean(lmage), np.median(lmage), np.std(lmage))) 
print(" Grave  = %5.3e %5.3e %5.3e" %(np.mean(lgrave), np.median(lgrave), np.std(lgrave))) 
print(" Mage/KE  = %5.4f %5.4f %5.4f" %(np.mean(lmage/lke), np.median(lmage/lke), np.std(lmage/lke))) 
print(" Grave/KE  = %5.4f %5.4f %5.4f" %(np.mean(lke/lgrave), np.median(lke/lgrave), np.std(lke/lgrave))) 
print(" SinkFrac = %5.3f" %sinkfrac)

