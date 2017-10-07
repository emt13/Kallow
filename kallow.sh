#!/bin/bash
# Job name:
#SBATCH --job-name=kallow
#
# Account:
#SBATCH --account=alex_vlissidis
#
# Wall clock limit:
#SBATCH --time=00:10:00
#
# Memory:
#SBATCH --mem-per-cpu=20G
#
# Mail type:
#SBATCH --mail-type=all
#
# Mail user:
#SBATCH --mail-user=alex_vlissidis@berkeley.edu
## Command(s) to run:
python kallow.py
