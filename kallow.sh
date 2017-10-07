#!/bin/bash
# Job name:
#SBATCH --job-name=kallow
#
# Account:
#SBATCH --account=fc_mlsec
#
# Wall clock limit:
#SBATCH --time=00:02:30
#
# Partition:
#SBATCH --partition=savio_bigmem
#
# Memory:
#SBATCH --mem-per-cpu=20G
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=alex_vlissidis@berkeley.edu
#
## Command(s) to run:
python3 kallow.py
