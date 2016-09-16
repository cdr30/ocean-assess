'''
Created on Oct 24, 2013

@author: hadtd
'''
from glob import glob
from utils import mask_gen
import os
import csv
import run_metrics
from netCDF4 import netcdftime as nctime, Dataset
from FileDates import File_Dates
from collections import namedtuple
import mesh_tools
import io_utils

class ModelError(Exception):
    pass

class Model(object):
    '''
    A class to represent information about a model and the assessment to be carried out on the model
   '''

    def __init__(self, inifile, metric_file, dom_file, path_file, obs_file, item_file, cntl = None):
        '''
        
        Create a Model instance
        
        Input parameters:
            inifile: An ini file containing the runid, description, start/end dates, data directories
            metric_file: csv file specifying tasks to be carried out with one task per line of form task, period, type, obs, domain(s)
            dom_file: An ini file specifying masks and regions to be used for each domain
            path_file: An ini file containing paths such as the output directory
            obs_file: An ini file containing information about all observations datasets used (not yet implemented)
            item_file: An ini file giving instructions on what to do for each metric
        Keyword parameters:
            An inifile containing information about a control model to be compared to the main model run
        '''
        
        # Read inifile to get information about the model
        (info,self.datadirs) = self._read_model_files(inifile)
        self.runid = info['runid']
        self.description = info['description']
        self.meshmask = info['mesh_mask']
        self.mesh = mesh_tools.mesh(self.meshmask)
        self.subbasins = info['basin_mask_file']
        self.grid = info['grid']
        start = info['mean_start']       
        self.mean_start = nctime.datetime(int(start[0:4]), int(start[4:6]),int(start[6:8]))
        end = info['mean_end']       
        self.mean_end = nctime.datetime(int(end[0:4]), int(end[4:6]),int(end[6:8]))
        
        #Read metric file
        self.metrics = self._read_metrics(metric_file)
        
        #paths
        self.paths = self._read_paths(path_file)
        self._make_assess_dir()
        
        #Read in the domains
        self.domains = self._read_domains(dom_file)
        self._make_domain_masks()
        
        #Set the obs_file and item_file as attributes to access later
        self.obs_file = obs_file
        self.item_file = item_file
        
        # Fill in lists of files and dates
        self._get_dates()
        
        #Set up path to fortran code to be used for faster data processing
        self.f90dir = os.path.abspath(os.path.dirname(__file__))+'/../f90/'
        
        if cntl is not None:
            self.cntl = Model(cntl, metric_file, dom_file, path_file, obs_file, item_file)
        else: self.cntl = None
        
    def _get_files(self, grid, period):
        '''
        Generate lists of files in each data directory
        '''
        files = glob(self.datadirs[period]+'/*'+period+'*'+grid+'.nc')
        files.sort()
            
        return files

    def _get_dates(self):
        '''
        Generate dictionaries of FileDateLists for each meaning period and grid type
        '''
                
        self.filedates={}
        for grid in ['T','U','V','W','ptr']:
            self.filedates[grid]={}
            for period in self.datadirs:
                if period != 'assess':
                    files = self._get_files(grid, period)
                    if len(files) != 0: 
                        self.filedates[grid][period] = File_Dates(files)
    
    def _make_assess_dir(self):
        '''
        Create directories to use for storing netcdf files created 
        '''
        try:
            assess_dir = self.datadirs['assess']
            if os.path.isdir(assess_dir):
                if not os.access(assess_dir, os.W_OK) and os.access(assess_dir, os.R_OK):
                    raise ModelError("assess directory must be writable and Readable")
            else:
                os.makedirs(assess_dir) 
        except KeyError:
            raise ModelError('assess directory must be specified in experiment.ini file')
        except OSError as err:
            raise ModelError('Could not create assess directory '+assess_dir, err)
        
    def _make_domain_masks(self):
        '''
        Make a netCDF file in ocean_assess_dir containing masks of each domain name
        '''
#???: Does this belong in this module? Seems meshy
#NOTE: There is the potential for conflict between two notes attempting to write to this file.
#NOTE: The meshmask is necessary but the subbasins file is not; we ought to allow self.subbasins 
#      to be empty and create a global default subdomain in this case
        ncid_basin = Dataset(self.subbasins)
        ncid_mesh = Dataset(self.meshmask)
#???: Is this assumption of a T grid problematic?
        mask0 = ncid_mesh.variables['tmaskutil'][:].squeeze()
        lat = ncid_mesh.variables['gphit'][:].squeeze()
        lon = ncid_mesh.variables['glamt'][:].squeeze()

        self.domain_masks = self.datadirs['assess'] + '/' + self.runid + '_domain_masks.nc'
        ncid_out = Dataset(self.domain_masks, 'w', clobber = True)
        
        nx = len(ncid_basin.dimensions['x'])
        ny = len(ncid_basin.dimensions['y'])
        ncid_out.createDimension('x', nx)
        ncid_out.createDimension('y', ny)
        
        for key,domain in self.domains.iteritems():
            if key.lower() != 'all':
                if domain.mask is not None:
#TODO: Test this bug fix for no haloes in the subbasin-based masks
                    bmask = ncid_basin.variables[domain.mask] * mask0
                else: bmask = mask0
                if domain.region is not None:
                    region_mask = mask_gen(lon, lat, domain.region) * bmask
                else: region_mask = bmask
                
                #Write to file
                ncid_out.createVariable(key, 'i', dimensions=('y','x'))
                ncid_out.variables[key][:] = region_mask
        ncid_out.close()
    
    def _read_model_files(self,inifile):
        '''
        Read information from model ini file
        Returns:
            info: dictionary containing info about the model
            datadirs: dictionary containing paths to datadirs
        '''
    
        conf = io_utils.OAConfigParser()

        #Read config file for experiment
        try:
            with open(inifile) as fid: 
                conf.readfp(fid)
                info = conf.items('info')
                datadirs = conf.items('directories')
        except IOError as err:
            raise ModelError("Cannot open experiment.ini file ("+ inifile + ")" + err.strerror)
        except Exception as err:
            print err
            raise ModelError("Unable to read experiment.ini file ("+ inifile +")")
    
        return info, datadirs
       
    def _read_metrics(self, csvfile):
        '''
        Read in Metrics from CSV file and parse into dictionaries
        '''
        
        try:
            with open(csvfile) as fid:
                reader = csv.reader(fid)
                metrics=[]
                for i,row in enumerate(reader):
                    if len(row) == 0:
                        #check for empty row
                        pass
                    elif row[0][0] == '#':
                        #Check for comment line
                        pass
                    elif len(row) != 6:
                        raise ModelError('Incorrect format of metrics csv file ' + csvfile + 
                                           ' at line '+str(i+1) )
                    else:
                        metrics.append({'metric': row[0].strip(), 
                                        'period': row[1].strip().split(), 
                                        'type': row[2].strip(), 
                                        'obsname': row[3].strip(), 
                                        'plot_domain': row[4].strip().split(),
                                        'metric_domain': row[5].strip().split()})
        except IOError:
            raise ModelError("Specified metrics csv file ("+csvfile+") cannot be opened")
        
        return metrics

    def _read_domains(self,inifile):
        '''
        Read name of mask var and regions for all domains
        Return a dictionary of named tuples?
        '''
        
        domain = namedtuple('domains','mask region')
        domains = {}
        try:
            conf = io_utils.OAConfigParser()
            
            with open(inifile) as fid:
                conf.readfp(fid)
                masks = conf.items('masks')
                regions = conf.items('regions')
            keys=masks.keys()
            keys.extend(regions.keys())  
            keys=list(set(keys)) #Get a unique set of keys
            for key in keys: 
                if masks.get(key) == 'None':
                    mask=None
                else: mask=masks.get(key)
                if regions.get(key) == 'None':
                    region=None
                else: 
                    region=regions.get_ndarray(key, dtype=float).tolist()
                domains[key] = domain( mask, region )
        except IOError:
            raise ModelError("domains.ini definition file ("+inifile+") cannot be opened")
        except Exception as err:
            raise ModelError("Error reading domains.ini definition file (" + inifile +")",err)
       
        return domains

    def _read_paths(self,inifile):
        '''
        Read paths from path_file
        '''
        try:
            conf = io_utils.OAConfigParser()
            
            with open(inifile) as fid:
                conf.readfp(fid)
                paths = conf.items('paths')
        except IOError:
            raise ModelError("paths.ini definition file ("+inifile+") cannot be opened")
        except Exception as err:
            raise ModelError("Error reading paths.ini definition file (" + inifile +")",err)
        
        return paths

    def get_domain_mask(self, domain):
        '''
        Read and return domain mask from file
        '''
        ncid = Dataset(self.domain_masks)
        try:
            mask = ncid.variables[domain][:]
        except KeyError:
            raise KeyError('domain_masks.nc does not contain a mask named "%s"' % domain)
        finally:
            ncid.close()
        return mask
        
    def assess(self):
        '''
        Carry out the assessment by calling run_metrics.run()
        '''
        run_metrics.run(self)
        
