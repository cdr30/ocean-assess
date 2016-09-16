

# -*- coding: iso-8859-1 -*-
'''
Produce maps of scalar model fields
If there is only one model then make a map of the field and a difference plot with obs
If there are two models then make a 2x2 validation note style plot
'''
import matplotlib
matplotlib.use('Agg')
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import os.path
import plot_tools
import netCDF4 as nc4
import numpy as np
import observations
import utils
import meaning
import calendar
import textwrap
import io_utils

class LatLonError(Exception): pass

def lat_lon_mean(expt, metric, items, out_dir):
    periods = metric['period'][:]
    
    #Overwrite 1m or 1s with individual months or seasons to be looped through
    if '1m' in periods:
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        i = periods.index('1m')
        periods.remove('1m')
        m=0
        while m < 12: 
            periods.insert(i+m, months[m])
            m+=1
    if '1s' in periods:
        months = ['djf','mam','jja','son']
        i = periods.index('1s')
        periods.remove('1s')
        m=0
        while m < 4: 
            periods.insert(i+m, months[m])
            m+=1
    
    # Get name of obs
    obs = metric['obsname']
    
    #Initialise output lists
    plotlist = []
    plot_tuple = utils.plot_tuple()
    tablelist = []
    table_tuple = utils.table_tuple()

    #Loop through each period making plots and stats tables
    for period in periods:
        field = map_field(expt, period, obs, items) 
        thisplotlist = make_figures(expt, metric, out_dir, field, period)
        plotlist.append(plot_tuple(period, thisplotlist))
        
        table = make_table(expt, metric, field)
        tablelist.append(table_tuple(period, table))
                              
    return plotlist, tablelist, None 

def lat_lon_maxmin(expt, metric, items, out_dir, function):
    '''
    Plot a map maximum monthly value of a field
    '''
    periods = metric['period'][:]
    if periods != ['1m']: 
        raise LatLonError("Only period '1m' is valid for lat_lon_maxmin")
   
    obs = metric['obsname']
    months = [calendar.month_abbr[i].lower() for i in np.arange(12)+1 ]
    
    field = map_field(expt, months, obs, items, multi_field_function = function)
        
    #Initialise output lists
    plot_tuple = utils.plot_tuple()
    table_tuple = utils.table_tuple()
    
    thisplotlist = make_figures(expt, metric, out_dir, field, periods[0])
    plotlist = [plot_tuple(periods[0], thisplotlist)]
        
    table = make_table(expt, metric, field)
    tablelist = [table_tuple(periods[0], table)]
                              
    return plotlist, tablelist, None 
        

def lat_lon_max(expt, metric, items, out_dir):
    '''
    Plot map of maximum monthly value of a field
    '''        
    return lat_lon_maxmin(expt, metric, items, out_dir, 'max')


def lat_lon_min(expt, metric, items, out_dir):
    '''
    Plot map of maximum monthly value of a field
    '''        
    return lat_lon_maxmin(expt, metric, items, out_dir, 'min')


def lat_lon_argmax(expt, metric, items, out_dir):
    '''
    Plot map of Month in which maximum monthly value of a field
    occurs
    '''        
    return lat_lon_maxmin(expt, metric, items, out_dir, 'argmax')


def lat_lon_argmin(expt, metric, items, out_dir):
    '''
    Plot map of Month in which minimum monthly value of a field
    occurs
    '''        
    return lat_lon_maxmin(expt, metric, items, out_dir, 'argmin')

def make_figures(expt, metric, out_dir, field, period):
    '''
    Plot figures and return a list of filenames
    '''
    
    plot_domains = metric['plot_domain']
    if plot_domains[0] == 'all': plot_domains = expt.domains.keys()
    
    thisplotlist = []
    for domain in plot_domains:
        fname = plot_tools.figure_filename(expt, metric, period, domain, out_dir)

#TODO: A one panel case is needed for when metric['obsname'] and expt.cntl is None
        if expt.cntl is not None:
            fig=plot_tools.figure_setup('four_panel')
        else:
            fig=plot_tools.figure_setup('two_panel')
        field.plot_maps(fig=fig, domain = domain )
        plt.savefig(fname)
        thisplotlist.append(fname)
        plt.close(fig)
        
    return thisplotlist

def make_table(expt, metric, field):
    '''
    Make table of metrics
    '''
    
    metric_domains = metric['metric_domain']
    if metric_domains[0] == 'all': metric_domains = expt.domains.keys()
    
    #Set up first row of table
    table = []
    if expt.cntl is not None:
        table.append(['Domain', 
                      expt.runid + '\n mean',
                      expt.cntl.runid + '\n mean',
                      metric['obsname'] + '\n mean',
                      expt.runid + '\n RMS error',
                      expt.cntl.runid + '\n RMS error',
                      'RMS model difference'])
    else:
        table.append(['Domain', 
                      expt.runid + '\n mean',
                      str(metric['obsname']) + '\n mean',
                      expt.runid + '\n RMS error'])
    
    for domain in metric_domains:
        if expt.cntl is not None:
            table.append([domain, 
                          field.mean(domain = domain),
                          field.cntl.mean(domain = domain),
                          field.obs_mean(domain = domain),
                          field.obs_rms(domain = domain),
                          field.cntl.obs_rms(domain = domain),
                          field.mod_rms(domain = domain)])
        else:
            table.append([domain, 
                          field.mean(domain = domain),
                          field.obs_mean(domain = domain),
                          field.obs_rms(domain = domain)])
    
        # Remove entirely nan rows, else blank out nan entries
        is_nan = ~np.isfinite(table[-1][1:])
        
        if is_nan.all():
            table.remove(table[-1])
        elif sum(is_nan) > 0: 
            table[-1][1:] = [row if np.isfinite(row) else '--' for row in table[-1][1:]]
            
    return table
 

class map_field():
    def __init__(self, expt, period, obs, items, multi_field_function = None):

        self.obs = obs
        self.items = items
        start = '%4i%02i%02i' %(expt.mean_start.year, expt.mean_start.month, expt.mean_start.day)
        end = '%4i%02i%02i' %(expt.mean_end.year, expt.mean_end.month, expt.mean_end.day)
        self.expt = expt
        self.multi_field_function = multi_field_function
        
        if multi_field_function is None:
            self.meandir = expt.datadirs['assess']+'/'+decode_period(period)+'/means/'
            if not os.path.isdir(self.meandir):
                os.makedirs(self.meandir)
            self.period = period
            meanfile = [expt.runid+'o', decode_period(period), start, end, self.items['variable']]
            if decode_period(period) != period:
                meanfile.insert(-1, period)
            self.meanfile = '_'.join(meanfile)+'.nc'
            self._make_mean()
        else:
            #Loop through periods making mean files
            self.meanfiles = []
            for per in period: 
                self.meandir = expt.datadirs['assess']+'/'+decode_period(per)+'/means/'
                if not os.path.isdir(self.meandir):
                    os.makedirs(self.meandir)
                self.period = per
#???: Pretty sure we don't need both self.meanfile and self.meanfiles. We can just have self.make_mean
#     and self._check_mean_file accept a filename argument, loop over self.meanfiles and remove 
#     self.meanfile
                meanfile = [expt.runid+'o', decode_period(per), start, end, self.items['variable']]
                if decode_period(per) != per:
                    meanfile.insert(-1, per)
                self.meanfile = '_'.join(meanfile)+'.nc'
                self._make_mean()
                self.meanfiles.append(self.meanfile)
                
        self._get_lat_lon()
        self._get_depth()
                
        self._read_field(multi_field_function)
        self.get_domain_mask = self.expt.get_domain_mask
        
        # Get data from observation field
        if (self.obs is not None) and (not hasattr(self, 'obs_field')):
            self.obs_class = observations.Obs(self.expt, self.obs, self.items)
            if self.multi_field_function is None:  
                self.obs_field = self.obs_class.get_mean_field(self.period, 
                                                               self.minlev, self.maxlev, 
                                                               zavg=(self.maxlev > self.minlev))
            else:
                self.obs_field = self.obs_class.get_max_min_field( decode_period(self.period),
                                                                   self.multi_field_function)
            
        if expt.cntl is not None:
            self.cntl = map_field(expt.cntl, period, obs, items, multi_field_function)
        else:
            self.cntl = None
    
    def _make_mean(self):
        '''
	    Check if meanfile exists (and check if it's readable?). If it doesn't then create it.
	    Should this be put in a separate module? - leave here for now and refactor later if required
	    '''
#TODO: This belongs in a central data handling module
        if not self._check_mean_file():
        #File does not exist (or needed to be remade and has been deleted)
        #Get list of input filedates
            period = decode_period(self.period)
            file_dates = self.expt.filedates[self.items['grid']]
            
            if period in file_dates:
                fdates = file_dates[period]
                if period == '1m':
                    fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end, month = self.period)
                elif period == '1s':
                    fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end, season = self.period)
                elif period == '1y': 
                    fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end)
                else:
                    err = 'Period "%s" is not implemented'
                    raise LatLonError(err % period)
            else:
                err = 'No files with a mean period "%s" were found'
                raise LatLonError(err % self.period)
        
            grid = self.items['grid'].lower()
            meaning.make_mean(self.meandir+'/'+self.meanfile, fdates, self.items['variable'], self.expt.mesh, grid)
    
    def _check_mean_file(self):
        '''
        Check if mean file exists and is readable (return True if it is) or False if not
	    If file exists and is not readable attempt to delete it so a new file can be made
	    '''
        try:
            exists = meaning.check_file(self.meandir + '/' + self.meanfile)
            return exists
        except meaning.MeaningError as err:
            raise LatLonError(err)

    def _check_cntl_grid(self):
        '''
        Check that both models are on the same grid. 
        '''
        
        if self.cntl is not None:
            if (not np.array_equal(self.lat, self.cntl.lat)) or (not np.array_equal(self.lon, self.cntl.lon)):
                raise LatLonError('Both models must be on the same horizontal grid')
            if not np.array_equal(self.depth, self.cntl.depth):
                raise LatLonError('Both models must be on the same vertical grid')

    def _get_lat_lon(self):
        '''
        Read latitude and longitude from mesh mask file.
        
        '''
        
        grid = self.items['grid'].lower()
        self.lat, self.lon = self.expt.mesh.get_latlon(grid) 

#NOTE: Copied from CrossSection
    def _get_depth(self):
        '''
        Read depth coordinate from mesh mask.
        '''

        grid = self.items['grid'].lower()
        
        # Get depth variable
        depth = self.expt.mesh.get_depth(grid)
            
        # Requested vertical levels
        self.minlev = int(self.items.get('minlev', 0))
        self.maxlev = int(self.items.get('maxlev', depth.size - 1))
 
        # Index depths by requested levels
        index = np.arange(self.minlev, self.maxlev + 1)
        self.depth = depth[index]

    def _read_field(self, multi_field_function=None ):
        '''
        Read in variable from self.mean_file for correct levels only (if 3D field)
        Or if multi_field_function is set then get the field (e.g.) maximum from the set of fields in meanfiles
	    '''
        # import pdb; pdb.set_trace()
        # Get all required mean files
        if multi_field_function is None:
            files = [self.meandir + '/' + self.meanfile]
        else:
            files = [self.meandir + '/' + fname for fname in self.meanfiles]
            fields = []
        
        for fname in files:
            ncid = nc4.Dataset(fname)
            var = ncid.variables[self.items['variable']]
            
            # Get dimension indices
            dims = utils.query_dims(var)
            
            # Time and depth slices
            index = [slice(None)] * var.ndim
            if ('t' not in dims) and (multi_field_function is not None):
                err = 'Model variable "%s" has no time coordinate; ' + \
                      'cannot calculate time function "%s"'
                raise LatLonError(err % (self.items['variable'], multi_field_function))
                
            if 'z' in dims:
                index[dims['z']] = slice(self.minlev, self.maxlev + 1)
            
            # Slicing needed prior to self.depth_mean 
            var = var[index]

            # If necessary do vertical mean
            if ('z' in dims) and (self.maxlev > self.minlev):
                var = self.depth_mean(var, dims['z'])
            
            # Stack fields in time
            if multi_field_function is None:
                self.field = var.squeeze()
            else:
                fields.append(var)
                
            ncid.close()

#???: These embedded "catch-all" errors are not very readable and 
#     prevent direct tracebacks; better to have just low-level ones IMO                

#TODO: Another set of functions that could be replaced with a generalised operation_over_axis
#      (i.e. operation = ['max', 'min', 'mean'] etc).
        if multi_field_function == 'max':
            self.field = np.ma.concatenate(fields).max(dims['t'])
        elif multi_field_function == 'min':
            self.field = np.ma.concatenate(fields).min(dims['t'])
        elif multi_field_function == 'argmax':
            fields = np.ma.concatenate(fields)
            mask = fields[0].mask
            self.field = np.ma.array(fields.argmax(dims['t']) , mask=mask)
        elif multi_field_function == 'argmin':
            fields = np.ma.concatenate(fields)
            mask = fields[0].mask
            self.field = np.ma.array(fields.argmin(dims['t']) , mask=mask)

#NOTE: Here I have replaced "e3 * tmask" with masked e3. While neater, it may be slightly slower
    def depth_mean(self, var, zdim):
        '''
	    Return weighted depth average of var
	    zdim is dimension to average over
	    '''
        
#TODO: Replace eventually with a generalised mean_over_axis function
        # Read in mask and e3 variable from mesh file
        z_lev = np.arange(self.minlev, self.maxlev + 1)
        grid = self.items['grid'].lower()
        e3 = self.expt.mesh.get_e3(grid, z_lev)
        
        dims = self.expt.mesh.get_e3(grid, return_dims=True)
        
        # Apply mesh mask to e3
        mask = self.expt.mesh.get_mask(grid, z_lev)
#NOTE: The IDL validation note uses mdtol = 0 type masking for the model plot. In testing, this code
#      produces 0-700m RMS global values that are ~0.2 degree different to the IDL code for mdtol = 0. 
#      This line is needed for mdtol = 0:
#         mask = utils.replicate_ndarray(self.expt.mesh.get_mask(grid, self.maxlev), 
#                                        self.maxlev - self.minlev + 1)
        e3 = np.ma.masked_where(mask == 0, e3)                

        # Apply mesh mask to model field if required    
        if type(var) is np.ndarray:
            if (mask.shape != var.shape) and (mask.ndim <= var.ndim):
                mask, var = np.broadcast_arrays(mask, var)
                
            var = np.ma.masked_where(mask == 0, var)

        return (var * e3).sum(zdim) / e3.sum(dims['z'])

    def plot_maps(self, fig=None, domain = None):
        '''
        Produce multipanel plot
	    If there is no control run then plot the field and field-obs.
	    If there is a control run then plot the field, field-cntl, cntl-obs, field-obs
	    '''
        if fig is None: 
            fig = plt.gcf()
        
        projection = self.items.get('projection')
        if projection is not None:
            try:
                exec "projection = ccrs." + projection + "()"
            except Exception as err:
                raise UserWarning('Unable to use specified projection - will use PlateCarree instead',err)
        if projection is None: projection = ccrs.PlateCarree()
        
        if domain is not None:
            dom_tuple = self.expt.domains.get(domain)
            if dom_tuple is not None:
                region = dom_tuple.region
            else:
                #TODO: This will cause the code to crash for all regions
                #Is this a good way to do it?
                raise LatLonError('Unknown domain specified ('+domain+')')
        else: region = None
        
        # Plotting method
        method = self.items.get('contour_method', 'mesh')

        #import pdb; pdb.set_trace()
        if self.cntl is None:
            self.map_self(subplot=211, projection=projection, region=region, method=method)
            if self.obs is not None:
                self.map_obs_diff(subplot=212, projection=projection, region=region, method=method)

        else:
            self.map_self(subplot=221, projection=projection, region=region, method=method)
            self.map_model_diff(subplot=222, projection=projection, region=region, method=method)
            
            if self.obs is not None:
                self.cntl.map_obs_diff(subplot=223, projection=projection, region=region, method=method)
                self.map_obs_diff(subplot=224, projection=projection, region=region, method=method)

        

    def map_self(self, subplot=111, region=None, method='mesh', **kwargs):
        '''
        Make a map of the field
        '''
        #Get levels
        levs = self.items.get('contour_levels_fill')
        if levs is not None:
            exec "levs="+levs
        else: levs = None

        #Read om colormap
        cmap_file = self.items.get('cmap_file',None)
        if cmap_file is not None: 
            cmap = plot_tools.read_cmap(cmap_file)
        else: 
            cmap = plt.cm.jet
 
        plot_tools.plot_xy_fill(self.lon, self.lat, self.field, method=method,
                                levels=levs, subplot=subplot, cmap=cmap, **kwargs)
        plt.gca().set_title(textwrap.fill(self.expt.runid,40),fontsize=12)
        if region is not None:
            plt.gca().set_xlim([region[2], region[3] ])
            plt.gca().set_ylim([region[1], region[0] ])
        
    def map_model_diff(self, subplot=111, region=None, method='mesh', **kwargs):
        '''
        Make a map of differences between models
        '''
        
        #The error below shouldn't be needed as the metrics code wouldn't call this method but no harm in keeping it
        if self.cntl is None: raise LatLonError('No control model to calculate difference against')
        
        try:
            self._check_cntl_grid()
        except LatLonError as err:
            io_utils.warn('Cannot create model - control plot; ' + err.message)
            return
        
        levs = self.items.get('contour_levels_diff')
        if levs is not None:
            exec "levs="+levs
        else: levs = None
        
        #Read in colormap
        cmap_file = self.items.get('cmap_file_diff',None)
        if cmap_file is not None: 
            cmap = plot_tools.read_cmap(cmap_file)
        else: 
            cmap = plt.cm.RdBu_r

        diff_field = self.field-self.cntl.field
        if self.multi_field_function=='argmax' or self.multi_field_function=='argmin':
        # Fields are months so make maximum difference +/- 6
            diff_field[diff_field > 6] -= 12
            diff_field[diff_field < 6] += 12

        plot_tools.plot_xy_fill(self.lon, self.lat, diff_field, method=method, set_middle=True,
                                levels=levs, subplot=subplot, cmap=cmap, **kwargs)
        plt.gca().set_title(textwrap.fill(self.expt.runid + ' minus ' +self.cntl.expt.runid,40),fontsize=12)
        if region is not None:
            plt.gca().set_xlim([region[2], region[3] ])
            plt.gca().set_ylim([region[1], region[0] ])
            
        
    def map_obs_diff(self, subplot=111, cntl=False, region=None, method='mesh', **kwargs):
        '''
        Make map of model - obs
        Should we regrid the model to obs grid here?
        Probably should it's harder to do (and slower)
        '''
        
        
        levs = self.items.get('contour_levels_obs')
        if levs is not None:
            exec "levs="+levs
        else: levs = None

        #Read in colormap
        cmap_file = self.items.get('cmap_file_diff',None)
        if cmap_file is not None: 
            cmap = plot_tools.read_cmap(cmap_file)
        else: 
            cmap = plt.cm.RdBu_r
 
        #import pdb; pdb.set_trace()
        diff_field = self.field-self.obs_field
        
        if self.multi_field_function=='argmax' or self.multi_field_function=='argmin':
        # Fields are months so make maximum difference +/- 6
            diff_field[diff_field > 6] -= 12
            diff_field[diff_field < -6] += 12

        plot_tools.plot_xy_fill(self.lon, self.lat, diff_field, method=method, set_middle=True, 
                                levels=levs, subplot=subplot, cmap=cmap, **kwargs)
        #TODO: can just refer to obs_class.long_name 
        plt.gca().set_title(textwrap.fill(self.expt.runid + ' minus ' 
            + self.obs_class.obsinput.get('long_name',self.obs),40),fontsize=12)
        if region is not None:
            plt.gca().set_xlim([region[2], region[3] ])
            plt.gca().set_ylim([region[1], region[0] ])

    def mean(self, domain = 'global'):
        '''
        Calculate the area weighted mean of field over specified domain
        '''
        
        err = 'cannot calculate model mean'
        if domain is not 'global' and domain not in self.expt.domains:
            io_utils.warn('Undefined domain specified; ' + err)
            return np.nan

        area = self.get_area() * self.get_domain_mask(domain)
        
        return (self.field * area).sum()/area.sum()
    
    def obs_mean(self, domain = 'global'):
        '''
        Calculate the area weighted mean of obs field over specified domain
        '''
        if self.obs is None:
            return np.nan
        else:
            err = 'cannot calculate observation mean'
            if domain is not 'global' and domain not in self.expt.domains:
                io_utils.warn('Undefined domain specified; ' + err)
                return np.nan
                
            area = self.get_area() * self.get_domain_mask(domain)
            return (self.obs_field * area).sum()/area.sum()
 

    def obs_rms(self, domain = 'global'):
        '''
        Calculate the RMS difference between field and observations for specified domain
        '''
        if self.obs is None:
            return np.nan
        else:
            err = 'cannot calculate model RMS error'
            if domain is not 'global' and domain not in self.expt.domains:
                io_utils.warn('Undefined domain specified; ' + err)
                return np.nan
        
            area = self.get_area() * self.get_domain_mask(domain)
            return (((self.field-self.obs_field)**2 * area).sum() / area.sum())**0.5

    def mod_rms(self, domain = 'global'):
        '''
        Calculate the RMS difference between the model and the control experiment
        '''
        
        err = 'cannot calculate model - control RMS difference'
        if domain is not 'global' and domain not in self.expt.domains:
            io_utils.warn('Undefined domain specified; ' + err)
            return np.nan
        if self.cntl is None:
            io_utils.warn('No control model specified; ' + err)
            return np.nan
        
        try:
            self._check_cntl_grid()
            area = self.get_area() * self.get_domain_mask(domain)
            return (((self.field - self.cntl.field) ** 2 * area).sum() / area.sum()) ** 0.5
        except LatLonError as exc:
            io_utils.warn('{}; {}'.format(exc.message, err))
            return np.nan
    
    def get_area(self):
        '''
        Read in area from mesh file and mask for minumum level specified in items
        '''
        grid = self.items['grid'].lower()
        
#NOTE: This line is needed if using depth_mean mdtol = 0
#         area = self.expt.mesh.get_area(grid) * self.expt.mesh.get_mask(grid, self.maxlev)
        area = self.expt.mesh.get_area(grid) * self.expt.mesh.get_mask(grid, self.minlev)
        
        return area
        

def decode_period(period):
    '''
    Parse period to work out what needs to be done
    ''' 
    #If individual months or seasons are specified then need to do monthly or seasonal averaging
    if period in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
        mean_period = '1m'
    elif period in ['djf', 'mam', 'jja', 'son']:
        mean_period = '1s'
    elif period == '1y':
        mean_period = '1y'
    else: raise LatLonError('Period specified ('+period+') is not valid')

    return mean_period



    
