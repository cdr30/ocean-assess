#!/bin/bash -l
#SBATCH --mem=500
#SBATCH --output=./test_complete.out
#SBATCH --error=./test_complete.err
#SBATCH --time=20
#SBATCH --ntasks=1

# Run on SPICE with "sbatch test_complete.sh"

../../ocean_assess.py --expt experiment.ini \
                      --metrics metrics.csv --items items.ini \
                      --paths paths.ini --domains domains.ini \
                      --obs observations.ini 

rm -r $LOCALTEMP/ocean_assess_runtime_data

../../ocean_assess.py --expt experiment.ini --cntl control.ini \
                      --metrics metrics.csv --items items.ini \
                      --paths paths.ini --domains domains.ini \
                      --obs observations.ini 

rm -r $LOCALTEMP/ocean_assess_runtime_data
