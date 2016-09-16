#! /bin/bash

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
