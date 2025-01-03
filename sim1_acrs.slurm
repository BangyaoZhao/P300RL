#!/bin/bash
# The interpreter used to execute the script

#SBATCH --job-name=bci
#SBATCH --mail-user=byzhao@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=2
#SBATCH --mem=1000m
#SBATCH --time=2:00:00
#SBATCH --account=jiankang0
#SBATCH --partition=standard
#SBATCH --array=0-149

module load python3.10-anaconda/2023.03
python sim1_acrs.py