import numpy as np
try:
    import configparser as ConfigParser #Python version 3 and later
except ImportError:
    import ConfigParser #Python version 2.7
import os
from FileDates import File_Dates
import netCDF4 as nc4
import utils
import io_utils

class ObsError(Exception): pass

class Obs(object):
    '''
    Class to handle observational data
    '''
    def __init__(self, expt, obsname, items):
        '''
        Input arguments:
        expt: A model class
        obsname: The name of the observations to read
        items: Dictionary of items read in from items.ini file for metric being assessed
        '''
        
        self.expt = expt
        self.obsname = obsname
        self.items = items
        
        if not os.path.isfile(self.expt.obs_file):
            raise ObsError('observations file '+ self.expt.obs_file + ' not found')
        
        #Read the config file
        conf = io_utils.OAConfigParser()
        
        try:
            with open(self.expt.obs_file) as fid:
                conf.readfp(fid)
                self.obsinput = conf.items(obsname)
        except IOError as err:
            raise ObsError("Cannot open observations file " + self.expt.obs_file, err)
        except ConfigParser.NoSectionError:
            raise ObsError("No observations defined in " + self.expt.obs_file + " for "+obsname)
        
        try:
            self.long_name = self.obsinput['long_name']
        except KeyError:
            self.long_name = self.obsname
            
#TODO: We want to check dimensionality against the model data here

#IDEA: This should subclass a common Data class.
#      We could make the model and observation classes subclasses of a main Data class. This would be the data "layer" of Model and Obs and
#      should contain methods needed by the metric classes to access the data and certain common attributes describing the data. 
#      For example, there is no need for Obs to contain the same cross section code as the CrossSection class; CrossSection should be able
#      to handle both. 
#      Similarly, a lot of functions in Obs are generic indexing and averages over spatiotemporal axes that apply equally validly to 
#      data referred to by the Model class. In general, the Data class would handle spatiotemporal data (t, z, y, x).
#      This is where iris would come in handy, as the Data class could load and store the data lazily and pass it to the metric classes, 
#      which analyze and plot it. Basically we would be moving the generic data handling methods from the metric classes to Model and Obs 
#      via the Data class.
    def get_mean_field(self, period, minlev = None, maxlev = None, zavg = False, xsect_class = None):
        '''
        Read in mean field from observations file
	    period: String period for which observations are required
	        Options are 1y, djf, mam, jja, son, jan, feb, mar, apr,
	                    may, jun, jul, aug, sep, oct, nov, dec
	                    
	    Returns a numpy array of dimensions (z, x).
	    '''
        if xsect_class is not None:
            xsect_points = xsect_class.points
        
        if period == '1y':
            key = '1y_'+self.expt.grid
        elif period in ['djf','mam','jja','son']:
            key = '1s_'+self.expt.grid
        elif period in ['jan', 'feb', 'mar', 'apr','may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
            key = '1m_'+self.expt.grid
        else:
            raise ValueError('Invalid period ('+period+') specified')
        
        try:
            fname = self.obsinput[key]
        except KeyError:
            raise ObsError('No obs file specified for '+self.obsname+' and period '+key)
        
        fdates = File_Dates(fname)
        if period in ['djf','mam','jja','son']:
            fdates = fdates.in_season(period)
        elif period in ['jan', 'feb', 'mar', 'apr','may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
            fdates = fdates.in_month(period)
        
        if len(fdates) > 1:
            raise ObsError('Too many records found for period '+period)
        
        #Read the field
        ncid = nc4.Dataset(fname)
        var = ncid.variables[self.obsinput['varname']]
        
        dims = utils.query_dims(var)
        ndims = len(var.dimensions)
        index = [slice(None)] * ndims
        
        # Get time and depth records 
        if 't' in dims:
            record = fdates[0].record
            index[dims['t']] = slice(record, record + 1)
        if 'z' in dims:
            zdim = dims['z']
        else:
            zdim = None

#NOTE: This should be done within a central I/O routine, as most files will need this check
#???: A lot of observations have rather appalling metadata; should this be corrected or should assumptions be made of dimensionality? 
        # All non-singlet dimensions must be recognised as a standard spatiotemporal coordinate 
        n_dims_gt1 = sum([i > 1 for i in var.shape])
        
        if len(dims) < n_dims_gt1:
            bad_dims = set(range(ndims)) - set(dims.values())
            err = 'Observation "%s" has unrecognised dimensions of size > 1 (indices %s)'
            raise ObsError(err % (self.long_name, tuple(bad_dims)))
        
        # Cross section indexing
        if xsect_class is not None:
            
            if xsect_class.mean_axis is None:
#HACK: See notes from cross_section_plot
                index[dims['y']] = np.append(xsect_points[0][0], xsect_points[0])
                index[dims['x']] = np.append(xsect_points[1][0], xsect_points[1])
            else:
                index[dims['y']] = xsect_points[0]
                index[dims['x']] = xsect_points[1]
                axis = dims[xsect_class.mean_axis]
                
                var = utils.mask_and_crop(var, index)[0]

        # Perform slice / vertical average
        field = self.read_level_field(var, index, zdim, zavg, minlev, maxlev, xsect_class=xsect_class)
        
        # Further cross section actions
        if xsect_class is not None:
                    
            # Initialize output field
            obs_field = {}
            
            if xsect_class.mean_axis is None:
#HACK: See notes from cross_section_plot  
                index = [slice(None)] * field.ndim
                index[field.shape.index(xsect_points[0].size + 1)] = slice(1, None)
                
                field = field[index]
        
            if type(field) is np.ndarray:
                field = np.ma.array(field)
            
            # Save original mask
            mask_orig = field.mask.copy()
                
            # Loop over subdomains and produce one mask per domain
            for domain in xsect_class.domains:
                
#NOTE: This modifies the mask of var, due to var = var[index] in read_level_field. 
#      We can avoid this by using a .copy(), but really there is no need as var is no longer used.
                # Combine both masks in case the subdomain / mesh mask does not match that of the observations
                field.mask = mask_orig | xsect_class.get_domain_mask(domain)

#NOTE: Both this and the depth_mean code produce mdtol = 1 masking. If the observation and model masks
#      differ however, the average will not be consistent. This is avoided with mdtol = 0 masking.    
                if xsect_class.mean_axis is not None:
                    obs_field[domain] = xsect_class.mean_over_axis(field, axis).squeeze()
                else:
                    obs_field[domain] = field.squeeze()
        
        else:
            obs_field = field.squeeze()

#TODO: Need something here to re-order dimensions if necessary so z is the leading dimension (due to plotting requirements)
#      xsect_class provides the order of dimensions for the xsect, so can use this to reorder the obs xsect if needed
 
        return obs_field
        
        
    def read_level_field(self, var, index, zdim, zavg, minlev, maxlev, xsect_class=None):
#IDEA: This could be removed; the vertical indexing and slice command should be in get_mean_field

#HACK: Temporarily, we reset index for the mean cross section case as we have already done the indexing in x and y.
#      What we want in the long term is for mask_and_crop to be used for all indexing as it doesn't broadcast dimensions and therefore
#      retains the field dimensionality. This function can then be retired in favor of just the zavg lines.
        if (xsect_class is not None) and (xsect_class.mean_axis is not None):
            index = [slice(None)] * var.ndim

        #Not sure about this section as if there is a depth dimension with unrecognised name then it won't be found!
        if zdim is not None:
            if minlev is not None:
                zmin = minlev
            else:
                zmin = 0
            if maxlev is not None:
                zmax = maxlev + 1
            else:
                zmax = var.shape[zdim]
            
            index[zdim] = slice(zmin, zmax)

#NOTE: This should instead be done by an initial grid comparison against the experiment model in the calling metrics module
        # Check whether the cross section points are valid for the observational field
        try:
            var = var[index]
        except IndexError:
            err = 'Cross section indices cannot index observation "%s" with shape %s:\n %s'
            raise ObsError(err % (self.long_name, var.shape, index))
        
        if zavg and zdim is not None:
            var = self.depth_mean(var, zdim, zmin, zmax - 1)
        
        return var
 
    
    def get_max_min_field(self, period, function):
        '''
        Get the minimum, maximum or index of the min/max of a set of fields (e.g maximum annual maximum monthly field)
        '''
        #BUG: Need to translate period to '1s','1m' etc
        key = period + '_' + self.expt.grid
        fname = self.obsinput[key]
        ncid = nc4.Dataset(fname)
        var = ncid.variables[self.obsinput['varname']]
        
        dims = utils.query_dims(var)
        if 't' in dims:
            tdim = dims['t']
        else:
            err = 'Observations variable "%s" has no time coordinate; cannot calculate time function "%s"'
            raise ObsError(err % (self.long_name, function))
        
        # The mask may be time-varying: mask result where data is not valid for all times
        if type(var) is np.ma.core.MaskedArray:
            mask = var[:].mask.max(tdim)
        else:
            mask = None
        
        if function == 'max':
            data = var[:].max(tdim)
        elif function == 'min':
            data = var[:].min(tdim)
        elif function == 'argmax':
            data = var[:].argmax(tdim)
        elif function == 'argmin':
            data = var[:].argmin(tdim)
        else:
            raise ObsError('Function {} is not implemented'.format(function))

        return np.ma.array(data, mask=mask)

    def depth_mean(self, var, zdim, minlevel, maxlevel):
        '''
	    Return weighted depth average of var
	    zdim is dimension to average over
	    minlevel,maxlevel = min and max levels to average over
	    '''

#TODO: Replace eventually with a generalised mean_over_axis function
        # Read in mask and e3 variable from mesh file
        z_lev = np.arange(minlevel, maxlevel + 1)
        grid = self.items['grid'].lower()
        e3 = self.expt.mesh.get_e3(grid, z_lev)
        
        dims = self.expt.mesh.get_e3(grid, return_dims=True)
        
        # Apply mask to e3
        mask = self.expt.mesh.get_mask(grid, z_lev)
        e3 = np.ma.masked_where(mask == 0, e3)

        # Apply mesh mask to observation field if required
        if type(var) is np.ndarray:
            if (mask.shape != var.shape) and (mask.ndim <= var.ndim):
                mask, var = np.broadcast_arrays(mask, var)
                
            var = np.ma.masked_where(mask == 0, var)
            
        return (var * e3).sum(zdim) / e3.sum(dims['z'])
