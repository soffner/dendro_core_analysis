# dendro_core_analysis

Overview: Set of analysis scripts to analyze STARFORGE cores.

_____________________
Author: Stella Offner

Code Contributions: Carleen Markey, Mike Grudic
_____________________

Main analysis scripts: 

* *analyze_core_props_v2.py* (identify cores via dendrograms, compute their properties, and save result)

  *get_properties_v2.py* (functions to compute core properties)
* *make_leaf_history_fast.py* (get linked cores between consecutive snapshots)
  
  *leaf_history_functions.py* (functions to identify parent/children cores)
_________________
Packages Required
-----------------
* *scipy (cKDTree), astropy, pandas, h5py, matplotlib*
* *astrodendro* with tree modification (see my astrodendro branch)
* *pytreegrav* to compute core gravitational energy (from mgrudic repository)

_________________
 Usage
-----------------

Parameters:
* Set file location, name, output directory set of snapshots in *analyze_core_props_v2.py, make_leaf_history_fast.py*
* Set mass resolution limit, *res_limit*, in code units -- minimum particle mass to consider -- in *analyze_core_props_v2.py, make_leaf_history_fast.py*
  This excludes feedback gas with cells mass below this limit from the core identification. Make sure dendrogram output/input filename is consistent.
* Set the number of bins (*nbin*) and core radius (*maxsize*) used to compute the profiles in *analyze_core_props_v2.py*
* Set *search_radius* for parent/child linkage in call to *create_leaf_history_fast* (default 1.0 code units). Limit speeds up search by excluding distant cores.
  
