#!/bin/bash

#SBATCH --account=def-escheme
#SBATCH --mem-per-cpu=1.5G                                          # increase as needed
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00                                              # walltime in d-hh:mm or hh:mm:ss format
#SBATCH --mail-user=saeed.kazemi@unb.ca
#SBATCH --mail-type=ALL

module load python/3.8
virtualenv ~/env
source ~/env/bin/activate
pip install --upgrade pip


pip install --no-index pandas scikit_learn matplotlib seaborn
pip install --no-index tensorflow jupyterlab


python ./Codes/test2.py


## $ chmod 755 bash.sh
## $ seff {Job_ID}                                                                                       # list resources used by a completed job 
## $ sacct -j jobID [--format=jobid,maxrss,elapsed]                                                      # list resources used by a completed job
## $ scancel <jobid>                                                                                     # Cancelling jobs
## $ sbatch simple_job.sh                                                                                # submit jobs

## salloc --account=def-escheme --cpus-per-task=1 --mem=1000M --time=0:10:00                             # intractive mode


## $ scp filename saeed67@cedar.computecanada.ca:/path/to                                               # File transfer
## $ scp saeed67@cedar.computecanada.ca:/path/to/filename localPath                                     # File transfer





