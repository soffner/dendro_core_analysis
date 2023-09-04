#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 3 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J ISRFx10_ # Job name
#SBATCH -o core.o%j # Name of stdout output file 
#SBATCH -e core.e%j # Name of stderr error file 
#SBATCH -A AST21002
#SBATCH -t 48:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

export PYTHONPATH=$HOME/anaconda3/lib/python3.8/site-packages

#export OMP_NUM_THREADS=10
#./radmc3d image npix 256 loadlambda fluxcons inclline linelist nostar sizepc 20.0 phi 0 incl 0 

module load launcher
export LAUNCHER_RMI=SLURM
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins


#                         # JOB_FILE is a list of executions to run

export LAUNCHER_JOB_FILE=`pwd`/commands_core
export LAUNCHER_SCHED=interleaved
export LAUNCHER_WORKDIR=`pwd`

$LAUNCHER_DIR/paramrun
