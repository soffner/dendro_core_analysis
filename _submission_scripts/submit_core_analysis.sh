#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 56 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J core_anaysis # Job name
#SBATCH -o core.o%j # Name of stdout output file 
#SBATCH -e core.e%j # Name of stderr error file 
#SBATCH -A AST21002
#SBATCH -t 24:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

## To run this file: sbatch submit_radmc.sh

export OMP_NUM_THREADS=56
#./radmc3d image npix 256 loadlambda fluxcons inclline linelist nostar sizepc 20.0 phi 0 incl 0 

python analyze_cores_v2.py


