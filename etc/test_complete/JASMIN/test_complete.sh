#! /bin/bash
#BSUB -R "rusage[mem=500]"
#BSUB -o ./test_complete.out
#BSUB -e ./test_complete.err
#BSUB -W 00:20
#BSUB -n 1
#BSUB -q lotus

# Run on LOTUS with "bsub < test_complete_JASMIN.sh"

../../../ocean_assess.py --expt experiment.ini \
                         --metrics metrics.csv --items items.ini \
                         --paths paths.ini --domains domains.ini \
                         --obs observations.ini

rm -r /work/scratch/$USER/ocean_assess_runtime_data

../../../ocean_assess.py --expt experiment.ini --cntl control.ini \
                         --metrics metrics.csv --items items.ini \
                         --paths paths.ini --domains domains.ini \
                         --obs observations.ini

rm -r /work/scratch/$USER/ocean_assess_runtime_data
