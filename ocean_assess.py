#!/usr/bin/env python2.7

import argparse
import os
from src.model import Model


def main():
    '''
    ocean_assess.py
    Main program to run ocean assessment
    Method:
        1) Call get_arguments to parse command line args
        2) Initiate the model class
        3) Run the assessment
    '''
    #Parse input arguments
    args = get_arguments()
    expt = Model(args.expt, 
                 args.metrics, 
                 args.domains, 
                 args.paths, 
                 args.obs, 
                 args.items, 
                 cntl = args.cntl)
    expt.assess()
    
def get_arguments():
    '''
    Parse command line arguments
    '''
    
    usage = """
    
    ocean_assess.py:
    Welcome to the ocean_assess package. A flexible tool developed 
    to automatically assess your global NEMO ocean model.
    
    Author: Tim Graham
    
    Usage: 
    ocean_assess.py --expt EXPT [--cntl CNTL --metrics METRICS --obs OBS --paths PATHS --domains DOMAINS --items ITEMS]
    """
    
    prefix = os.path.abspath(os.path.dirname(__file__))
    
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('--expt',
                        help = 'Config file describing the model to be \
                                assessed',
                        action = 'store', type=str, required = True)
    parser.add_argument('--cntl',
                        help = 'Config file describing the control model \
                        to compare against',
                        action = 'store', type = str)
    parser.add_argument('--metrics',
                        help='csv file specifying the metrics to assess',
                        action = 'store', type = str,
                        default=prefix+'/etc/metrics.csv')
    parser.add_argument('--obs',
                        help='Config file describing observations',
                        action ='store', type = str,
                        default = prefix+'/etc/observations.ini')
    parser.add_argument('--paths',
                        help="""
                        Override the default config file specifying paths (e.g. output directory).
                        """,
                        action ='store', type = str,
                        default = prefix+'/etc/paths.ini')
    parser.add_argument('--items',
                        help="""
                        Override the default config file specifying item definitions.
                        """,
                        action ='store', type = str,
                        default = prefix+'/etc/items.ini')
    parser.add_argument('--domains',
                        help="""
                        Override the default config file describing domains over \
                        which to do averaging""",
                        action ='store', type = str,
                        default = prefix+'/etc/domains.ini')
   
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    main()
