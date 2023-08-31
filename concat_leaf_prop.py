import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(precision=5,threshold=sys.maxsize)
import glob as glob
from natsort import natsorted 

#----------------------------------------------------------
# Code to concatenate property files and generate time file
# Author: S. Offner
#---------------------------------------------------------

unit_base = {'UnitMagneticField_in_gauss':  1e+4,
             'UnitLength_in_cm'         : 3.08568e+18,
             'UnitMass_in_g'            :   1.989e+33,
             'UnitVelocity_in_cm_per_s' :      100}


# Read in Data From Folder of csv files
vers = 'v2'

tag = 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_isrf10_snapshot_*_prop_'+vers+'.csv'
file_save = 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_isrf10_all_prop_'+vers+'.csv'
time_save = 'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_isrf10_times_'+vers+'.csv'
save = True  # Whether to save the profiles
pkl_save =  'M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_42_isrf10_all_prop_'+vers+'.pkl'

fns = glob.glob(tag)
fns = natsorted(fns) #fns.sort() # Need natsort for snapshot > 1000, otherwise do time sort

converter_nparray = lambda x: np.array(x[1:-1].split(), dtype = float)

frames = []
times = []
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
    time = np.float(fn[-21:-16])
    times.append(time)
    
profiles = pd.concat(frames)

# For v7
#lkeonly = profiles['LeafKe only'].values # x unitmass x unitvel^2
#print("Fixing units of KE only for v7....")
#for i,ke in enumerate(lkeonly):
#    (profiles['LeafKe only'].values)[i] = ke*2e33*(100.0)**2

if save: # Save all as a new data frame

    save_df = pd.DataFrame(profiles, columns = ['ID','Density [cm^-3]','Dispersion [cm/s]', 'Reff [pc]', 'LeafDisp [cm/s]', 'LeafMass [msun]', 'CoherentRadius [pc]',
                                                'DensityIndex','V Bulk [cm/s]','Center Position [pc]','Center index','LeafKe', 'LeafGrav', 'Num. Sinks',
                                                'Max Den [cm^-3]', 'Shape [pc]','Half Mass R[pc]', 'Mean B [G]', 'Mag. Energy', 'Sink Masses [Msun]',
                                                'Sound speed [cm/s]', 'LeafKe only','Protostellar','Median Outflow Mass Frac', 'Mean Outflow Mass Frac', 'Median Wind Mass Frac', 'Mean Wind Mass Frac'])

    print("Saving cores in ", file_save)
    save_df.to_csv(file_save,index=False)
    save_df.to_pickle(pkl_save)


    print("Saving times in ", time_save)
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
lke = profiles['LeafKe'].values #total EKE + Eth, cgs
lgrave = profiles['LeafGrav'].values # cgs
lmage = profiles['Mag. Energy'].values # cgs
lsink = profiles['Num. Sinks'].values
lsinkmass = profiles['Sink Masses [Msun]'].values
lsound = profiles['Sound speed [cm/s]'].values 
lproto = profiles['Protostellar'].values 
lkeonly = profiles['LeafKe only'].values #cgs 
medoutmassfrac = profiles['Median Outflow Mass Frac'].values
meanoutmassfrac = profiles['Mean Outflow Mass Frac'].values
medwindmassfrac = profiles['Median Wind Mass Frac'].values
meanwindmassfrac = profiles['Mean Wind Mass Frac'].values

nleaves = len(lsound)
indsink = np.where(lsink > 0)[0]
indnosink = np.where(lsink == 0)[0]
sinkfrac = len(indsink)/nleaves

## the common grid used for the interpolation (maxsize = 0.4)
#size_grid = np.log10(np.logspace(np.log10(2.0e-3), np.log10(maxsize/2.0), 20))
print("Check: min leaf mass =", np.min(lmass), np.max(lmass), np.median(lmass))

lgrave = np.abs(lgrave)
lmeanv = []
lmeanb = []
lmajor = []
lminor = []
for i in range(len(lsound)):
    lmeanv.append(np.sqrt(np.sum(lvbulk[i]**2)))
    lmeanb.append(np.sqrt(np.sum(lb[i]**2)))
    lmajor.append(lshape[i][1]/lshape[i][2]) # Sorted smallest to largest
    lminor.append(lshape[i][0]/lshape[i][2])



lmeanb = np.array(lmeanb)
lmeanv = np.array(lmeanv)
lmajor = np.array(lmajor)
lminor = np.array(lminor)

np.set_printoptions(precision=2, suppress=2, threshold=2)

kms = 1e5
ug = 1e6 # G -> uG
print("Simulation Core statistics (%i cores):" %len(lmass))
print(" M (Msun)        = %5.3f %5.3f %5.3f" %(np.mean(lmass), np.median(lmass), np.std(lmass)))
print(" R (pc)          = %5.3f %5.3f %5.3f" %(np.mean(lradius), np.median(lradius), np.std(lradius) ))
print(" Max den (cm^-3) = %5.3f %5.3f %5.3f" %(np.mean(lmax), np.median(lmax), np.std(lmax) ))
massden = lmass*2e33/(4.0/3.0*np.pi*(lradius*3.09e18)**3)
print(" Mean den (cm^-3) = %5.3e %5.3e %5.3e" %(np.mean(massden), np.median(massden), np.std(massden) ))
print(" Half Max (pc)   = %5.3f %5.3f %5.3f" %(np.mean(lhalf), np.median(lhalf), np.std(lhalf) ))
print(" Mean B (uG)     = %5.3f %5.3f %5.3f" %(np.mean(lmeanb)*ug, np.median(lmeanb)*ug, np.std(lmeanb)*ug ))
#np.average(lb, axis=0)*ug), np.sqrt(np.sum(np.average(lb, axis=0)*ug,axis=0)**2)#, np.median(lb, axis=0)/uG) #, np.std(lb, axis=0)/ug )
print(" Major Ratio (b/a) = %5.3f %5.3f %5.3f" %(np.average(lmajor), np.median(lmajor), np.std(lmajor) ))
print(" Minor Ratio (b/a) = %5.3f %5.3f %5.3f" %(np.average(lminor), np.median(lminor), np.std(lminor) ))
print(" 3D Vdisp (km/s) = %5.3f %5.3f %5.3f" %(np.mean(ldisp)/kms, np.median(ldisp)/kms, np.std(ldisp)/kms))
print(" Rcoh (pc)       = %5.3f %5.3f %5.3f" %(np.mean(rcoh), np.median(rcoh), np.std(rcoh)))
print(" Den Power Law   = %5.3f %5.3f %5.3f" %(np.mean(rhopow), np.median(rhopow), np.std(rhopow)))

ind = np.where(np.abs(rhopow) > 0)[0]
print("   Number of non-0 density indicies =", len(ind))
print(" Den Power Law (!=0) = %5.3f %5.3f %5.3f" %(np.mean(rhopow[ind]), np.median(rhopow[ind]), np.std(rhopow[ind])))

print(" Sound Speed     = %5.3f %5.3f %5.3f" %(np.mean(lsound)/kms, np.median(lsound)/kms, np.std(lsound)/kms))
print("**")
print(" KE    = %5.3e %5.3e %5.3e" %(np.mean(lke), np.median(lke), np.std(lke)))
print(" KE only= %5.3e %5.3e %5.3e" %(np.mean(lkeonly), np.median(lkeonly), np.std(lkeonly)))

print(" Mage  = %5.3e %5.3e %5.3e" %(np.mean(lmage), np.median(lmage), np.std(lmage))) 
print(" Grave  = %5.3e %5.3e %5.3e" %(np.mean(lgrave), np.median(lgrave), np.std(lgrave))) 
print(" Mage/KE  = %5.4f %5.4f %5.4f" %(np.mean(lmage/lke), np.median(lmage/lke), np.std(lmage/lke))) 
print(" Mage/Grave  = %5.4f %5.4f %5.4f" %(np.mean(lmage/lgrave), np.median(lmage/lgrave), np.std(lmage/lgrave))) 
print(" KE/Grave  = %5.4f %5.4f %5.4f" %(np.mean(lke/lgrave), np.median(lke/lgrave), np.std(lke/lgrave))) 
print(" KEonly/Grave  = %5.4f %5.4f %5.4f" %(np.mean(lkeonly/lgrave), np.median(lkeonly/lgrave), np.std(lkeonly/lgrave)))
print(" Outflow Mass Fract  = %5.4f %5.4f %5.4f" %(np.mean(medoutmassfrac), np.median(medoutmassfrac), np.std(medoutmassfrac)))
print(" Wind Mass Fract  = %5.3e %5.3e %5.e" %(np.mean(medwindmassfrac), np.median(medwindmassfrac), np.std(medwindmassfrac)))

print(" SinkFrac = %5.3f" %sinkfrac)





# the histograms of the data

# Radius
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lradius[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lradius[indsink]), 30, stacked=True, color='purple', label='Protostellar')
n, bins, patches = ax.hist(np.log10(lhalf[indnosink]), 30, stacked=True, label='Starless (half mass)', histtype='step', color='black')
n, bins, patches = ax.hist(np.log10(lhalf[indsink]), 30, stacked=True, color='blue', label='Protostellar (half mass)', histtype='step')
ax.set_xlabel('Log R [pc]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rad_'+vers+'.png')

# Aspect Ratio
fig, ax = plt.subplots()
n, bins, patches = ax.hist(lmajor[indnosink], 30, stacked=True, label='b/a: Starless')
n, bins, patches = ax.hist(lmajor[indsink], 30, stacked=True, color='purple', label='b/a: Protostellar')
n, bins, patches = ax.hist(lminor[indnosink], 30, stacked=True, label='c/a: Starless', histtype='step', color='black')
n, bins, patches = ax.hist(lminor[indsink], 30, stacked=True, color='blue', label='c/a: Protostellar', histtype='step')
ax.set_xlabel('Aspect Ratio')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Asp_'+vers+'.png')

# Mass
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lmass[indnosink]), 30, stacked=True,label='Starless')
n, bins, patches = ax.hist(np.log10(lmass[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log M [Msun]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('M_'+vers+'.png')

# Velocity Dispersion
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(ldisp[indnosink]/kms), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(ldisp[indsink]/kms), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $\sigma_{3D}$ [km/s]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Vdisp_'+vers+'.png')

# Brms
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lmeanb[indnosink]*ug), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lmeanb[indsink]*ug), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $B_{rms}$ [$\mu$G]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Brms_'+vers+'.png')

# Vrms bulk 
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lmeanv[indnosink]/kms), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lmeanv[indsink]/kms), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $V_{bulk}$ [km/s]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Vbulk_'+vers+'.png')

# Ratio of EB/KE
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lmage[indnosink]/lke[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lmage[indsink]/lke[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $E_B/E_K$')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rat_be_ke_'+vers+'.png')

# Ratio of EB/Grave
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lmage[indnosink]/lgrave[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lmage[indsink]/lgrave[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $E_B/E_G$')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rat_be_grave_'+vers+'.png')

# Ratio of Mass-to-Flux 
crit = 1.0/(3.0*np.pi)*(5.0/6.67e-8)**0.5 #E.g., Hu, Wibking, Krumholz 2023
mtf = lmass*2e33/(np.pi*(lradius*3.09e18)**2*lmeanb/np.sqrt(3.0)) #lmeanb = Brms
# Correct by a factor of 5 used in the radius definition for some reason
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(mtf[indnosink]/crit), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(mtf[indsink]/crit), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log Mass to Flux Ratio')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rat_MasstoFlux_'+vers+'.png')

# Ratio of EB/KE
# Note CloudPhinder defines virial parameter as 2*KE/PE
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lke[indnosink]/lgrave[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lke[indsink]/lgrave[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $E_K/E_G$')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rat_ke_grave_'+vers+'.png')

# Ratio of EB/KE
# Note CloudPhinder defines virial parameter as 2*KE/PE
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(lkeonly[indnosink]/lgrave[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(lkeonly[indsink]/lgrave[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log $E_K/E_G$')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rat_ke_only_grave_'+vers+'.png')

# Rcoh
fig, ax = plt.subplots()
n, bins, patches = ax.hist(rcoh[indnosink], 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(rcoh[indsink], 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Rcoh [pc]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rcoh_'+vers+'.png')


# Mean Temp
temp = (lsound)**2*2.33*1.667e-24/1.38e-16
fig, ax = plt.subplots()
n, bins, patches = ax.hist(np.log10(temp[indnosink]), 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(np.log10(temp[indsink]), 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Log T [K]')
ax.set_ylabel('N')
plt.legend()
fig.savefig('T_'+vers+'.png')

fig, ax = plt.subplots()
n, bins, patches = ax.hist(rhopow[indnosink], 30, stacked=True, label='Starless')
n, bins, patches = ax.hist(rhopow[indsink], 30, stacked=True, color='purple', label='Protostellar')
ax.set_xlabel('Density exponent: p')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Rhopow_'+vers+'.png')

fig, ax = plt.subplots()
medoutmassfractmp = medoutmassfrac
tmp = np.where(medoutmassfrac == 0)[0]
medoutmassfractmp[tmp] = 1.0
medwindmassfractmp = medwindmassfrac
tmp = np.where(medwindmassfrac == 0)[0]
medwindmassfractmp[tmp] = 1.0

n, bins, patches = ax.hist(np.log10(medoutmassfractmp[indnosink]), 30, stacked=True, label='Starless Outflow')
n, bins, patches = ax.hist(np.log10(medoutmassfractmp[indsink]), 30, stacked=True, color='purple', label='Protostellar Outflow')
n, bins, patches = ax.hist(np.log10(medwindmassfractmp[indnosink]), 30, stacked=True, label='Starless Wind', color='grey')
n, bins, patches = ax.hist(np.log10(medwindmassfractmp[indsink]), 30, stacked=True, color='green', label='Protostellar Wind')
ax.set_xlabel('Log Feedback Mass Fraction')
ax.set_ylabel('N')
plt.legend()
fig.savefig('Feedback_'+vers+'.png')


coeff, cov = np.polyfit(np.log10(lradius), np.log10(lmass), 1, cov='True')
print("Starless: slope, offset:", coeff, np.sqrt(np.diag(cov)))

coeff, cov = np.polyfit(np.log10(lradius[indsink]), np.log10(lmass[indsink]), 1, cov='True')
print("Protostellar: slope, offset:", coeff, np.sqrt(np.diag(cov)))

coeff, cov = np.polyfit(np.log10(lradius), np.log10(lmass), 1, cov='True')
print("All: slope, offset:", coeff, np.sqrt(np.diag(cov)))

rx = np.logspace(-2, 0, 10)
ry = 10**(coeff[0]*np.log10(rx)+coeff[1])
fig, ax = plt.subplots()
ax.scatter(lradius[indnosink], lmass[indnosink], label='Starless', s=20, alpha=0.02)
ax.scatter(lradius[indsink], lmass[indsink], s=20, alpha=0.1, color='purple', label='Protostellar')
ax.loglog(rx, ry, color='grey')
ax.set_xlabel('R [pc]')
ax.set_ylabel('M [Msun]')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend()
fig.savefig('MvsR_'+vers+'.png')

virx = np.array([5e38, 3e45])
viry = np.array([5e38,3e45])

fig, ax = plt.subplots()
ax.scatter(lkeonly[indnosink], lgrave[indnosink], label='Starless', s=20, alpha=0.02)
ax.scatter(lkeonly[indsink], lgrave[indsink], s=20, alpha=0.1,color='purple', label='Protostellar')
ax.loglog(virx, virx,  color='grey')
ax.set_xlabel('KE')
ax.set_ylabel('PE')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend()
fig.savefig('KEonlyvsPE_'+vers+'.png')

fig, ax = plt.subplots()
ax.scatter(lke[indnosink], lgrave[indnosink], label='Starless', s=20, alpha=0.02)
ax.scatter(lke[indsink], lgrave[indsink], s=20, alpha=0.1,color='purple', label='Protostellar')
ax.loglog(virx, virx,  color='grey')
ax.set_xlabel('KE')
ax.set_ylabel('PE')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend()
fig.savefig('KEvsPE_'+vers+'.png')

fig, ax = plt.subplots()
ax.scatter(lmage[indnosink], lgrave[indnosink], label='Starless', s=20, alpha=0.02)
ax.scatter(lmage[indsink], lgrave[indsink], s=20, alpha=0.1,color='purple', label='Protostellar')
ax.loglog(virx, virx,  color='grey')
ax.set_xlabel('BE')
ax.set_ylabel('PE')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend()
fig.savefig('BEvsPE_'+vers+'.png')

fig, ax = plt.subplots()
ax.scatter(massden[indnosink], lmeanb[indnosink]*ug, label='Starless', s=20, alpha=0.02)
ax.scatter(massden[indsink], lmeanb[indsink]*ug, s=20, alpha=0.1,color='purple', label='Protostellar')
ax.loglog(np.array([1e-22, 1e-16]), np.array([2.0, 2000.]),  color='grey')
ax.set_xlabel('rho [g cm$^{-3}$]')
ax.set_ylabel('B (uG)')
ax.set_yscale("log")
ax.set_xscale("log")
plt.legend()
fig.savefig('Bvsrho_'+vers+'.png')


