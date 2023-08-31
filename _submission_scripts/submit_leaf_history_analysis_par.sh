#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 5 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J ISRF_leaf # Job name
#SBATCH -o node.o%j # Name of stdout output file 
#SBATCH -e node.e%j # Name of stderr error file 
#SBATCH -A AST21002
#SBATCH -t 48:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

## To run this file: sbatch submit_radmc.sh
module unload python3/3.7.0

export PYTHONPATH=$HOME/anaconda3/lib/python3.8/site-packages

#export OMP_NUM_THREADS=2
#./radmc3d image npix 256 loadlambda fluxcons inclline linelist nostar sizepc 20.0 phi 0 incl 0 

#python analyze_cores_v2.py
#python analyze_cores_v2_2.py
#python analyze_cores_v2_4.py
#python analyze_cores_v2_6.py
#python analyze_cores_v2_8.py

module load launcher
export LAUNCHER_RMI=SLURM
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins


#                         # JOB_FILE is a list of executions to run

export LAUNCHER_JOB_FILE=`pwd`/commands_nodes
export LAUNCHER_SCHED=interleaved
export LAUNCHER_WORKDIR=`pwd`

$LAUNCHER_DIR/paramrun
