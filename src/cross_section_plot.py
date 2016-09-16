'''
Created on Oct 23, 2014

@author: hadtd
'''
import os
import meaning
import netCDF4 as nc4
import numpy as np
import utils
import plot_tools
import matplotlib.pyplot as plt
import observations
import io_utils
from matplotlib.ticker import MaxNLocator

#!!!: If at any point files store data in a different order to (t, z, y, x), we will need to call a reordering function within
#     every function that accesses a netCDF file. This would also eliminate the need to query_dims later on. Again, this
#     supports the need for a central I/O class.

class CrossSectError(Exception):
    pass


def cross_section(expt, metric, items, out_dir):
    '''
    Create plots of and calculate metrics for cross sections
    '''
    #TODO: This preamble common to the metrics modules could be condensed 
    
    # Get metric periods
    periods = metric['period'][:]
    
    # Overwrite '1m' or '1s' with individual months or seasons to be looped through
    if '1m' in periods:
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        i = periods.index('1m')
        periods.remove('1m')
        m = 0
        while m < 12: 
            periods.insert(i + m, months[m])
            m += 1
    if '1s' in periods:
        months = ['djf','mam','jja','son']
        i = periods.index('1s')
        periods.remove('1s')
        m = 0
        while m < 4: 
            periods.insert(i + m, months[m])
            m += 1
            
    # Get name of observational dataset
    obs = metric['obsname']
    
    # Subdomains masks to apply to the cross section
    domains = list(set(metric['plot_domain']) | set(metric['metric_domain']))
    if 'all' in domains:
        domains = expt.domains.keys()
    
    # Initialise output lists
    plotlist = []
    plot_tuple = utils.plot_tuple()
    tablelist = []
    table_tuple = utils.table_tuple()
    
#IDEA: It would be particularly beneficial to have some way of reusing existing plots in the assess directory, via some keyword in metrics.csv.
#      This is because plots are automatically always made, and only metrics in metrics.csv appear in the webpage. Really metrics.csv should control
#      what the webpage displays and should recover plots where necessary. In this case, CrossSection should be skipped and make_figures becomes simpler.
#      The table metrics would need to be saved in a CSV file; they are not currently.
    # Get cross section data, make plots and metric table for each period 
    for period in periods:
        field = CrossSection(expt, period, obs, items, domains) 

        thisplotlist = make_figures(expt, metric, out_dir, field, period)
        plotlist.append(plot_tuple(period, thisplotlist))
        
        table = make_table(expt, metric, field)
        tablelist.append(table_tuple(period, table))
    
    return plotlist, tablelist, None


def make_figures(expt, metric, out_dir, field, period):
    '''
    Plot figures and return a list of filenames
    '''
    
    plot_domains = metric['plot_domain']
    if 'all' in plot_domains: 
        plot_domains = expt.domains.keys()
    
    thisplotlist = []
    
    # Plot and save section for each plot domain
    for domain in plot_domains:
        
        fname = plot_tools.figure_filename(expt, metric, period, domain, out_dir)
                                            
        if expt.cntl is not None:
            fig=plot_tools.figure_setup('four_panel')
        else:
            fig=plot_tools.figure_setup('two_panel')
        
        field.plot_sections(fig=fig, domain=domain)
        plt.savefig(fname)
        thisplotlist.append(fname)

        # Close the figure instance to stop them accumulating
        plt.close(fig)
        
    return thisplotlist


def make_table(expt, metric, field):
    '''
    Make table of metrics(?) for a single metric
    '''
    
    metric_domains = metric['metric_domain']
    if 'all' in metric_domains: 
        metric_domains = expt.domains.keys()
    
    # Set up first row of table headers
    table = []
    
    if expt.cntl is not None:
        table.append(['Domain', 
                      expt.runid + '\n mean',
                      expt.cntl.runid + '\n mean',
                      str(metric['obsname']) + '\n mean',
                      expt.runid + '\n RMS error',
                      expt.cntl.runid + '\n RMS error',
                      'RMS model difference'])
    else:
        table.append(['Domain', 
                      expt.runid + '\n mean',
                      str(metric['obsname']) + '\n mean',
                      expt.runid + '\n RMS error'])
    
    # Add metric values to table, one row per metric domain
    for domain in metric_domains:
        if expt.cntl is not None:
            table.append([domain, 
                          field.mean(domain),
                          field.cntl.mean(domain),
                          field.obs_mean(domain),
                          field.obs_rms(domain),
                          field.cntl.obs_rms(domain),
                          field.mod_rms(domain)])
        else:
            table.append([domain, 
                          field.mean(domain),
                          field.obs_mean(domain),
                          field.obs_rms(domain)])

        # Remove entirely nan rows, else blank out nan entries
        is_nan = ~np.isfinite(table[-1][1:])
        
        if is_nan.all():
            table.remove(table[-1])
        elif sum(is_nan) > 0: 
            table[-1][1:] = [row if np.isfinite(row) else '--' for row in table[-1][1:]]
    
    return table


class CrossSection(object):
    '''
    Class for making cross section plots
    '''

    def __init__(self, expt, period, obs, items, domains):
        '''
        
        '''
        
        # Populate class attributes with arguments        
        self.expt = expt
        self.period = period
        self.obs = obs        
        self.items = items
        if 'all' in domains: 
            self.domains = self.expt.domains.keys()
        else:
            self.domains = domains
        
        # Meaning period
        start = '%4i%02i%02i' %(expt.mean_start.year, expt.mean_start.month, expt.mean_start.day)
        end = '%4i%02i%02i' %(expt.mean_end.year, expt.mean_end.month, expt.mean_end.day)
        
        # Mean directory and filename
        self.meandir = expt.datadirs['assess']+'/'+decode_period(period)+'/means/'
        if not os.path.isdir(self.meandir):
            os.makedirs(self.meandir)
        self.meanfiles = []
        meanfile = [expt.runid+'o', decode_period(period), start, end, self.items['variable']]
        if decode_period(period) != period:
            meanfile.insert(-1, period)
        self.meanfile = '_'.join(meanfile)+'.nc'
        
        # Loop through periods making mean files
        self._make_mean()
        
        # Get coordinates and vertical index limits
        self._get_lat_lon()
#IDEA: We can avoid the longitude wrap-around issue by writing a new function to:
#          * Wrap longitudes (input lon and nav_coords) to ind_i[0] / lon[0] and having self._get_lat_lon write sort indices to self.lon_sort_ind
#          * Use a biggus ArrayAdapter to rearrange the global field so that ind_i[0] / lon[0] is first
#          * self._find_points will operate on the rearranged coordinates
        self._get_depth()
        
        # Find section coordinate indices
        self._what_section()
        self._find_points()

        # Get cross section data and x coordinate
        self._read_field()
        self._make_xcoord()
        
        # Cross section cell areas
        self.face_areas = self.get_face_area()
        
        # Get cross section data from observation field
        if (self.obs is not None) and (not hasattr(self, 'obs_field')): 
            self.obs_class = observations.Obs(self.expt, self.obs, self.items)
            self.obs_field = self.obs_class.get_mean_field(self.period, 
                                                           minlev = self.minlev,
                                                           maxlev = self.maxlev,
                                                           xsect_class = self)

        # Guess an items title if necessary (edits items object)
        if (expt.cntl is None) and ('title' not in self.items):
            self.items['title'] = self._default_item_title()

        # Repeat for control experiment
        if expt.cntl is not None:
            self.cntl = CrossSection(expt.cntl, period, obs, items, domains)
        else:
            self.cntl = None
            
    def _default_item_title(self):
        '''
        Guess a 'title' entry for the items dictionary.
        '''        
        
        axis = {'y':'zonal', 'x':'meridional'}[self.xsect_axis]
        main_title = '%s %s cross section for variable \'%s\''
        
        return main_title % (self.points_repr, axis, self.items['variable'])

#TODO: Duplicated code with latlon
    def _get_lat_lon(self):
        '''
        Read latitude and longitude from mesh mask file.
        
	    '''
        
        grid = self.items['grid'].lower()
        latlon = self.expt.mesh.get_latlon(grid)
        dims = self.expt.mesh.get_latlon(grid, return_dims=True)

        # Force the dimensionality to be (y, x); this is expected by the rest of the module    
        for ind in xrange(2):
            if dims['y'] > dims['x']:
                latlon[ind] = np.rollaxis(latlon[ind], dims['y'], dims['x'])
        
        self.nav_lat = latlon[0]
        self.nav_lon = utils.wrap_longitude(latlon[1])

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
         
    def _what_section(self):
        '''
        Parse "items" for instructions on what type of section to plot
        '''
        
        try:
            self.section_type = self.items.get('section_type')
        except KeyError:
            raise CrossSectError('section_type not defined in items')
    
    def _read_field(self):
        '''
        Read in the data according to the type of section required.
        
        Returns a numpy array indexed by self.points.
        '''
        
        fname = self.meandir + '/' + self.meanfile

        try:            
            ncid = nc4.Dataset(fname)
            var = ncid.variables[self.items['variable']]
            ndims = var.ndim
                        
            # Initialize self.field
            self.field = {}
            
            # Get dimension indices
            dims = utils.query_dims(var)
            
            # Vertical slice: slice object to exclude dimension from numpy broadcasting
            index = [slice(None)] * ndims
            index[dims['z']] = slice(self.minlev, self.maxlev + 1)
            
            # Horizontal slice (path-like section): Collapse to a 2D section using numpy broadcasting (z, x)
            if self.mean_axis is None:
                
#HACK: Numpy fancy indexing only occurs when the indices are not monotonic, otherwise orthogonal indexing is used.
#      Here we duplicate the first horizontal point to force this type of indexing.
#      biggus.NumpyArrayAdapter doesn't seem to replicate fancy indexing but should in theory?
                index[dims['y']] = np.append(self.points[0][0], self.points[0])
                index[dims['x']] = np.append(self.points[1][0], self.points[1])
                field = var[index]
                
#HACK: Remove the extra horizontal point
                index = [slice(None)] * field.ndim
                index[field.shape.index(self.points[0].size + 1)] = slice(1, None)
                field = field[index]
                
            # Horizontal slice (mean section): First index to a 3D box (z, y, x)
            else:
                index[dims['y']] = self.points[0]
                index[dims['x']] = self.points[1]
                axis = dims[self.mean_axis]
                field = utils.mask_and_crop(var, index)[0]
                
            # Save original mask
            mask_orig = field.mask.copy()

            # Loop over subdomains and produce one mask per domain
            for domain in self.domains:

#???: Not sure if we should obscure a mismatch between the subdomain / mesh mask and missing model data like this?
                # Combine both masks in case the mesh mask does not match that of the model
                field.mask = mask_orig | self.get_domain_mask(domain)
            
                # Mean section: after the subdomain mask is applied, average to a 2D section (z, x)
                if self.mean_axis is not None:
                    self.field[domain] = self.mean_over_axis(field, axis).squeeze()
                else:
                    self.field[domain] = field.squeeze()

        except Exception as err:
            raise CrossSectError('Cross section for variable %s could not be produced' % self.items['variable'], err)

#IDEA: Might want something here to re-order dimensions if necessary so z is the leading dimension (due to plotting requirements)
#IDEA: We might also want to crop masked data in the horizontal; when lat/lon bounds are used to specify the extent of box_points,
#      this can result in an irregular unmasked area in i and j. The cropping over i / j can therefore leave a lot of missing data. 
        finally:
            ncid.close()

    def _find_points(self):
        '''
        Find grid points along a given section.
        
        Inputs (items.ini):
            lon = longitude of section - either a single value or end points of the section
            lat = latitude of section - either a single value or end points of the section
            index-i = i index of section - either a single value or end points of the section
            index-j = j index of section - either a single value or end points of the section
            
        NB: At least one of lon, lat, index-i and index-j must be provided
        
        Returns a tuple containing indices into (j, i) of data type:
            * 2D path-type sections:     numpy array
            * Average box-type sections: slice
        '''

        # Find points according to specific section type
        try:
            points_method = getattr(self, '_points_' + self.section_type)
        except AttributeError:
            raise RuntimeError('Section method %s unrecognised' % self.section_type)
        
        self.xsect_axis = None
        self.mean_axis = None
        points_method()
    
    def _points_fixed_longitude(self):
        '''
        Find nearest points to required longitude.
        
        '''
        
        self.xsect_axis = 'y'
        
        # Get coordinates
        lon = self.items.get_ndarray('longitude', None, float)
        lat = self.items.get_ndarray('latitude', None, float)
       
        # Check keyword sizes
        if (lon is None) or (len(lon) != 1):
            raise CrossSectError('A longitude must be specified in items.ini')
        if (lat is not None) and (len(lat) != 2):
            err = 'If specified, the latitude range must have length 2 (length is %s)'
            raise CrossSectError(err % len(lat))
        
        # String representation of points for titling
        self.points_repr = '%s%s' %(abs(lon[0]), ['W' if lon[0] < 0 else 'E'][0])
        
        # Wrap longitudes
        lon = utils.wrap_longitude(lon)
            
        # Set tolerance depending on grid resolution 
        # (set to half a grid point to avoid duplicate points)
        toldeg = (self.nav_lon[0, 1] - self.nav_lon[0, 0]) / 2.
        
        # Set latitude limits if necessary
        if lat is None:
            xmin = self.nav_lat.min()
            xmax = self.nav_lat.max()
        else:
            xmin = max(lat[0], self.nav_lat.min())
            xmax = min(lat[1], self.nav_lat.max())

#???: Can have issues towards the north fold where the grid starts to coarsen. Better to simply search for nearest points to target coord?
        # Find indices (j,i) of points that sit within longitude tolerance
        self.points = (np.where(np.where(self.nav_lon > lon[0] - toldeg, 1, 0) * 
                                np.where(self.nav_lon <= lon[0] + toldeg, 1, 0) * 
                                np.where(self.nav_lat >= xmin, 1, 0) * 
                                np.where(self.nav_lat <= xmax, 1, 0)))
        
    def _points_fixed_latitude(self):
        '''
        Find nearest points to required latitude.
        
        '''
        
        self.xsect_axis = 'x'
        
        # Get coordinates
        lon = self.items.get_ndarray('longitude', None, float)
        lat = self.items.get_ndarray('latitude', None, float)
        
        # Check keyword sizes
        if (lat is None) or (len(lat) != 1):
            raise CrossSectError('A latitude must be specified in items.ini')
        if (lon is not None) and (len(lon) != 2):
            err = 'If specified, the longitude range must have length 2 (length is %s)'
            raise CrossSectError(err % len(lon))
        
        # String representation of points for titling
        self.points_repr = '%s%s' %(abs(lat[0]), ['S' if lat[0] < 0 else 'N'][0])
        
        # Wrap longitudes
        if lon is not None:
            lon = utils.wrap_longitude(lon)
            
        # Set tolerance depending on grid resolution 
        # (set to half a grid point to avoid duplicate points).
        # Resolution in latitude varies so test near the required latitude
        ind = np.argmin(abs(self.nav_lat[:, 0] - lat[0]))
        toldeg = (self.nav_lat[ind, 0] - self.nav_lat[ind - 1, 0]) / 2.
        
        # Set longitude limits if necessary
        if lon is None:
            xmin = self.nav_lon.min()
            xmax = self.nav_lon.max()
        else:
#TODO: This won't work for a section crossing Greenwich Meridian
#NOTE: This is because -180 to 0 becomes 180 to 360 using the default 0 base; 0 is the left hand boundary
#      This could be fixed by setting "base = xmin - toldeg"
            xmin = max(lon[0], self.nav_lon.min())
            xmax = min(lon[1], self.nav_lon.max())
            
        # Catch points crossing 0E
        if xmin > xmax:
            raise CrossSectError('Section longitude limits traverse the 0 meridian; not currently supported')

#???: Can have issues towards the north fold where the grid starts to coarsen. Better to simply search for nearest points to target coord?
        # Find indices (j,i) of points that sit within latitude tolerance
        self.points = (np.where(np.where(self.nav_lat > lat[0] - toldeg, 1, 0) * 
                                np.where(self.nav_lat <= lat[0] + toldeg, 1, 0) * 
                                np.where(self.nav_lon >= xmin, 1, 0) * 
                                np.where(self.nav_lon <= xmax, 1, 0)))
        
    def _points_meridional_grid_line(self):
        '''
        Find points along a given j grid line.
        
        '''
        
        self.xsect_axis = 'y'
        
        # Get coordinates
        ind_i = self.items.get_ndarray('i-index', None, int)
        ind_j = self.items.get_ndarray('j-index', None, int)
        lat = self.items.get_ndarray('latitude', None, float)

        # Check keyword sizes
        if (ind_i is None) or (len(ind_i) != 1):
            raise CrossSectError('An i-index must be specified in items.ini')
        for j_coord in (ind_j, lat):
            if (j_coord is not None) and (len(j_coord) != 2):
                err = 'If specified, the "latitude" or "j-index" keyword must have length 2 (length is %s)'
                raise CrossSectError(err % len(j_coord))
        
        # String representation of points for titling
        self.points_repr = '%si' % ind_i[0]
        
        # Check indices are within global domain
        self._check_indices(ind_i, ind_j)

        # Set cross section limits
        if ind_j is not None:
            xlims = np.arange(ind_j[0], ind_j[1] + 1)
        elif lat is not None:
            points = np.where(self.nav_lat >= lat[0], 1, 0) * \
                     np.where(self.nav_lat <= lat[1], 1, 0)
            points = np.where(points[:, ind_i[0]])[0]
            xlims = np.arange(points.min(), points.max() + 1)
        else:
            xlims = np.arange(self.nav_lat.shape[0])
        
        # Find indices (j,i) of points along the grid line
        self.points = (xlims, np.tile(ind_i[0], xlims.size))
    
    def _points_zonal_grid_line(self):
        '''
        Find points along a given i grid line.
        
        '''
        
        self.xsect_axis = 'x'
        
        # Get coordinates
        ind_i = self.items.get_ndarray('i-index', None, int)
        ind_j = self.items.get_ndarray('j-index', None, int)
        lon = self.items.get_ndarray('longitude', None, float)
        
        # Check keyword sizes
        if (ind_j is None) or (len(ind_j) != 1):
            raise CrossSectError('An j-index must be specified in items.ini')
        for i_coord in (ind_i, lon):
            if (i_coord is not None) and (len(i_coord) != 2):
                err = 'If specified, the "longitude" or "i-index" keyword must have length 2 (length is %s)'
                raise CrossSectError(err % len(i_coord))

        # String representation of points for titling
        self.points_repr = '%sj' % ind_j[0]
        
        # Wrap longitudes
        if lon is not None:
            lon = utils.wrap_longitude(lon)

        # Check indices are within global domain
        self._check_indices(ind_i, ind_j)

        # Set cross section limits
        if ind_i is not None:
            xlims = np.arange(ind_i[0], ind_i[1] + 1)
        elif lon is not None:
            points = np.where(self.nav_lon >= lon[0], 1, 0) * \
                     np.where(self.nav_lon <= lon[1], 1, 0)
            points = np.where(points[ind_j[0], :])[0]
            xlims = np.arange(points.min(), points.max() + 1)
        else:
            xlims = np.arange(self.nav_lat.shape[1])
        
        # Find indices (j,i) of points along the grid line
        self.points = (np.tile(ind_j[0], xlims.size), xlims)
        
    def _points_true_path(self):
        '''
        Find evenly-spaced, interpolated points between two end-points in latitude-longitude space.
        '''
#IDEA: The fixed_* methods could be generalised in a similar way to this
        
        raise CrossSectError('Section method "true_zonal_line" still to be implemented')
        
    def _points_true_zonal_mean(self):
        '''
        '''
        
        raise RuntimeError('Section method "true_zonal_mean" still to be implemented')
    
    def _points_grid_zonal_mean(self):
        '''
        Find points within a box for which a zonal (over i) mean will be performed.
        
        '''
        
        self.xsect_axis = 'y'
        self.mean_axis = 'x'
        self.points = self._box_points()
        
    def _points_true_meridional_mean(self):
        '''
        '''
        
        raise RuntimeError('Section method "true_meridional_mean" still to be implemented')
    
    def _points_grid_meridional_mean(self):
        '''
        Find points within a box for which a meridional (over j) mean will be performed.
        
        '''
        
        self.xsect_axis = 'x'
        self.mean_axis = 'y'
        self.points = self._box_points()

    def _points_custom(self):
        '''
        '''
        
        raise RuntimeError('Section method "custom" still to be implemented')
    
    def _box_points(self):
        '''
        Find the indices contained within a given box.
        The box bounds are defined using pairs of given indices or lat-lon coordinates.
        
        '''
#IDEA: This could form the basis of the conventional point finding methods, with the method-specific functions containing error constraints
#      If we always use mask_and_crop for indexing, we can control / retain the dimensionality of the field throughout the processing and collapse only
#      at the end when needed (broadcasting for path-type sections, averaging for mean sections). 
        
        # Get coordinates
        ind_i = self.items.get_ndarray('i-index', None, int)
        ind_j = self.items.get_ndarray('j-index', None, int)
        lon = self.items.get_ndarray('longitude', None, float)
        lat = self.items.get_ndarray('latitude', None, float)
        
        # Check keyword sizes
        for coord in (ind_i, ind_j, lon, lat):
            if (coord is not None) and (len(coord) != 2):
                err = 'If specified, coordinate keywords must have length 2 (length is %s)'
                raise CrossSectError(err % len(coord))

        # Check indices are within global domain
        self._check_indices(ind_i, ind_j)
            
        # String representation of index limits for titling
        lims = None
        if self.xsect_axis == 'y':
            if ind_i is not None:
                lims = ind_i
                coord = ['i'] * 2
            elif lon is not None:
                lims = lon
                coord = ['W' if i < 0 else 'E' for i in lon]
        elif self.xsect_axis == 'x':
            if ind_j is not None:
                lims = ind_j
                coord = ['j'] * 2
            elif lat is not None:
                lims = lat
                coord = ['S' if i < 0 else 'N' for i in lat]
        
        # Wrap longitude input
        if lon is not None:
            lon = utils.wrap_longitude(lon)

        # Catch points crossing 0E
        if (lon is not None) and (lon[0] > lon[1]):
            raise CrossSectError('Section longitude limits traverse the 0 meridian; not currently supported')

        # Index array starts off with area limited by ind_* 
        if (ind_i is None) and (ind_j is None):
            index = np.ones(self.nav_lat.shape)
        else:
            index = np.zeros(self.nav_lat.shape)

#NOTE: We're assuming [y, x] ordering here
            # Box limits: indices
            ind = [slice(None)] * 2
            if ind_i is not None:
                ind[1] = slice(ind_i[0], ind_i[1] + 1)
            if ind_j is not None:
                ind[0] = slice(ind_j[0], ind_j[1] + 1)
                
            index[ind] = 1
         
        # Box limits: coordinates
#TODO: Coordinate tolerance (2D field) will be needed if this function is to be used for path-type sections also
        if lon is not None:
            if ind_i is None:
                index *= np.where(self.nav_lon >  lon[0], 1, 0) * \
                         np.where(self.nav_lon <= lon[1], 1, 0)
            else:
                io_utils.warn('Both longitude and index-i are specified in the items file; longitude is ignored')
        if lat is not None:
            if ind_j is None:
                index *= np.where(self.nav_lat >  lat[0], 1, 0) * \
                         np.where(self.nav_lat <= lat[1], 1, 0)
            else:
                io_utils.warn('Both latitude and index-j are specified in the items file; latitude is ignored')
                     
        points = np.where(index)

        # Check the box indices are finite
        if points[0].size == 0:  
            raise CrossSectError('Coordinates specified do not form a finite box')
        
        # String representation of points for titling
        if lims is not None:
            self.points_repr = '%s%s - %s%s average' % (lims[0], coord[0], lims[1], coord[1])
        else:
            self.points_repr = 'Globally averaged'
        
        return points
    
#TODO: Duplicated code here with latlon      
    def _make_mean(self):
        '''
        Check if meanfile exists (and check if it's readable?). If it doesn't then create it.
        Should this be put in a separate module? - leave here for now and refactor later if required
        '''
        
        # File does not exist (or needed to be remade and has been deleted)
        if not self._check_mean_file():

            # Get list of input filedates
            fdates = self.expt.filedates[self.items['grid']][decode_period(self.period)]

#TODO: Should there be a full metric failure because one month or season file is missing?
            if decode_period(self.period) == '1m':
                fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end, month = self.period)
            elif decode_period(self.period) == '1s':
                fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end, season = self.period)
            elif decode_period(self.period) == '1y':
                fdates = fdates.in_range(self.expt.mean_start, self.expt.mean_end)
            else:
                err = 'Period "%s" is not implemented'
                raise CrossSectError(err % self.period)
        
            grid = self.items['grid'].lower()
            meaning.make_mean(self.meandir+'/'+self.meanfile, fdates, self.items['variable'], self.expt.mesh, grid)

#TODO: Duplicated code with latlon
    def _check_mean_file(self):
        '''
        Check if mean file exists and is readable (return True if it is) or False if not
        If file exists and is not readable attempt to delete it so a new file can be made
        '''
        
        try:
            exists = meaning.check_file(self.meandir + '/' + self.meanfile)
            return exists
        except meaning.MeaningError as err:
            raise CrossSectError(err) 
        
    def _check_indices(self, ind_i, ind_j):
        '''
        Check if i and j indices are within the global domain.
        '''

        err = '%s-index %s lies outside the global range (0 - %s)'

#NOTE: We're assuming [y, x] ordering here
        for ij in xrange(2):
            ind_x = (ind_j, ind_i)[ij]
            bounds = self.nav_lat.shape[ij] - 1

            if ind_x is None:
                continue

            if not ((0 <= ind_x) & (ind_x <= bounds)).all():
                raise CrossSectError(err % (('j', 'i')[ij], ind_x, bounds))
            
    def _check_cntl_grid(self):
        '''
        Check that both models are on the same grid. 
        '''
        
        if self.cntl is not None:
            if not np.array_equal(self.xcoord, self.cntl.xcoord):
                raise CrossSectError('Both models must be on the same horizontal grid')
            if not np.array_equal(self.depth, self.cntl.depth):
                raise CrossSectError('Both models must be on the same vertical grid')

    def _make_xcoord(self):
        '''
        Create x coordinate depending on the type of section.
        In most cases the coordinate will be latitude or longitude.
        In the case of custom sections (i.e. not following grid lines or lat/lon lines) it will be distance from the end of the section
        '''
        
        if self.section_type == 'fixed_longitude':
            self.xcoord = self.nav_lat[self.points]
        elif self.section_type == 'fixed_latitude':
            self.xcoord = self.nav_lon[self.points]
        elif self.section_type == 'zonal_grid_line':
            self.xcoord = self.nav_lon[self.points]
        elif self.section_type == 'meridional_grid_line':
            self.xcoord = self.nav_lat[self.points]
        elif self.section_type == 'grid_zonal_mean':
            xcoord = utils.mask_and_crop(self.nav_lat, list(self.points))[0]
            self.xcoord = self.mean_over_axis(xcoord, 1)
        elif self.section_type == 'grid_meridional_mean':
            xcoord = utils.mask_and_crop(self.nav_lon, list(self.points))[0]
            xcoord[np.where(xcoord > 180)] -= 360.
            self.xcoord = self.mean_over_axis(xcoord, 0)            
        elif self.section_type == 'true_zonal_mean':
            raise RuntimeError('type of section still to be implemented')
        elif self.section_type == 'true_meridional_mean':
            raise RuntimeError('type of section still to be implemented')
        elif self.section_type == 'custom':
            raise RuntimeError('type of section still to be implemented')

#TODO: This deals with 2D section plots. This module must also deal with 1D sections (i.e. those without a depth coordinate)
    def plot_sections(self, fig=None, domain='global'):
        '''
        Create multipanel plot of field and field minus obs. 
        If cntl model is present cntl-expt will be plotted and cntl - obs
        '''

        if fig is None:
            fig = plt.gcf()
        
        use_cntl = self.cntl is not None
        use_obs = self.obs is not None
        
        # Plot expts
        if use_cntl:
            self.plot_panel('self', 221, domain)
            self.plot_panel('diff', 222, domain)
            if use_obs:
                self.plot_panel('obs', 224, domain)
                self.cntl.plot_panel('obs', 223, domain)
        else:
            self.plot_panel('self', 211, domain)
            if use_obs:
                self.plot_panel('obs', 212, domain)
        
        # Main figure title
        main_title = '%s (%s)'
        plt.suptitle(main_title % (self.items['title'], domain), fontsize=14)

#TODO: Put code not specific to the CrossSection class into a function in plot_tools
    def plot_panel(self, plot_type, subplot=111, domain='global', **kwargs):
        '''
        Plot a single panel of a cross section metric figure.
        
        Arguments:
            plot_type-
                One of 'self', 'diff', 'obs'
        
        Keywords:
            subplot-
                matplotlib.figure.Figure.add_subplot() position number
            **kwargs-
                Other keyword arguments into matplotlib.pyplot.contourf / pcolormesh
        
        '''
        
        # Whether a plot should cause a metric failure
        _plot_fail = {'self':True, 'diff':False, 'obs':True}
        
        # Plotting information
        _plot_items_levs = {'self':'contour_levels_fill', 'diff':'contour_levels_diff', 'obs':'contour_levels_obs'}
        _plot_items_cmap = {'self':'cmap_file', 'diff':'cmap_file_diff', 'obs':'cmap_file_obs'}
        _plot_cmap_default = {'self':plt.cm.jet, 'diff':plt.cm.RdBu_r, 'obs':plt.cm.RdBu_r}
        _plot_set_mid = {'self':False, 'diff':True, 'obs':True}
        
        # Check the plot type
        if plot_type not in _plot_items_levs:
            raise CrossSectError('Plot type "%s" is not implemented' % plot_type)
        
        # Get colourmap from items
        cmap_file = self.items.get(_plot_items_cmap[plot_type], None)
        if cmap_file is not None: 
            cmap = plot_tools.read_cmap(cmap_file)
        else: 
            cmap = _plot_cmap_default[plot_type]
        
        # Get levels from items
        levs = self.items.get(_plot_items_levs[plot_type], None)
        if levs is not None:
            exec "levs=" + levs
            
        # Plotting method
        method = self.items.get('contour_method', 'mesh')
            
        # Sort by xcoord
        sort_ind = np.argsort(self.xcoord)
        
        # Attempt to create a plot
        try:

            # Field to plot and the title
            if plot_type == 'self':
                plot_field = self.field[domain]
                plot_title = self.expt.runid
            elif plot_type == 'diff':
                self._check_cntl_grid()
                plot_field = self.field[domain] - self.cntl.field[domain]
                plot_title = self.expt.runid + ' - ' + self.cntl.expt.runid
            elif plot_type == 'obs':
                plot_field = self.field[domain] - self.obs_field[domain]
                plot_title = self.expt.runid + ' - ' + self.obs_class.long_name

            # Plot a single panel
            plot_tools.plot_xz_fill(self.xcoord[sort_ind], self.depth, plot_field[:,sort_ind], 
                                    levels=levs, cmap=cmap, subplot=subplot, method=method, 
                                    set_middle=_plot_set_mid[plot_type], **kwargs)
        
        # If the plot fails, how to handle it? 
        except Exception as err:
            if _plot_fail[plot_type]:
                raise
            else:
                warn = '"%s" type plot panel could not be created; %s'
                io_utils.warn(warn % (plot_type, err.message))
                return
            
        # Set the title
        plt.gca().set_title(plot_title)
        
        # Custom plt.axis options
        plot_tools.item_ax_options(plt.gca(), self.items)
        
#TODO: Some replicated code that could be tidied up in the following functions
    def mean(self, domain='global'):
        '''
        Calculate the area-weighted mean of field over specified domain
        '''

        err = 'cannot calculate model mean'
        if (domain is not 'global') and (domain not in self.expt.domains):
            io_utils.warn('Undefined domain specified; ' + err)
            return np.nan
        
        area = self.face_areas[domain]

        return (self.field[domain] * area).sum() / area.sum()
    
    def obs_mean(self, domain='global'):
        '''
        Calculate the area-weighted mean of obs field over specified domain
        '''
        if self.obs is None:
            return np.nan
        else:
            err = 'cannot calculate observation mean'
            if (domain is not 'global') and (domain not in self.obs_field):
                io_utils.warn('Undefined domain specified; ' + err)
                return np.nan
            if not hasattr(self, 'obs_field'):
                io_utils.warn('No observations specified; ' + err)
                return np.nan
        
            area = self.face_areas[domain]
            return (self.obs_field[domain] * area).sum() / area.sum()
 
    def obs_rms(self, domain='global'):
        '''
        Calculate the area-weighted RMS difference between field and observations for specified domain
        '''
        if self.obs is None:
            return np.nan
        else:
            err = 'cannot calculate model RMS error'
            if (domain is not 'global') and (domain not in self.expt.domains) and (domain not in self.obs_field):
                io_utils.warn('Undefined domain specified; ' + err)
                return np.nan
            if not hasattr(self, 'obs_field'):
                io_utils.warn('No observations specified; ' + err)
                return np.nan
            if self.field[domain].shape != self.obs_field[domain].shape:
                io_utils.warn('Grids do not match; ' + err)
                return np.nan
        
            area = self.face_areas[domain]
        
            return ((area * (self.field[domain] - self.obs_field[domain]) ** 2.).sum() / area.sum()) ** 0.5

    def mod_rms(self, domain='global'):
        '''
        Calculate the area-weighted RMS difference between the model and the control experiment
        '''
        
        err = 'cannot calculate model - control RMS difference'
        if (domain is not 'global') and (domain not in self.expt.domains) and (domain not in self.cntl.domains):
            io_utils.warn('Undefined domain specified; ' + err)
            return np.nan
        if self.cntl is None:
            io_utils.warn('No control model specified; ' + err)
            return np.nan
        
        try:
            self._check_cntl_grid()
            area = self.face_areas[domain]            
            return ((area * (self.field[domain] - self.cntl.field[domain]) ** 2.).sum() / area.sum()) ** 0.5
        except CrossSectError as exc:
            io_utils.warn('{}; {}'.format(exc.message, err))
            return np.nan
    
    def mean_over_axis(self, field, axis, mdtol=1):
        '''
        Perform an area-weighted average over the specified axis.
        
        A mask is applied where the proportion of missing data over 
        the axis exceeds mdtol or is equal to 1.
        '''
        
        # Check mdtol value
        if not 0 <= mdtol <= 1:
            err = 'mdtol must have a value 0 <= mdtol <= 1: found "%s"'
            raise CrossSectError(err % mdtol)
        
        # Weights are the cell areas derived from horizontal scale factors
        wgt = self.get_e123('x') * self.get_e123('y')
         
        # Force area weights to conform to shape of field
        if (wgt.shape != field.shape) and (wgt.ndim < field.ndim):
            try:
                if type(field) is np.ma.core.MaskedArray:
                    mask = field.mask
                else:
                    mask = None
                 
                wgt, field = np.broadcast_arrays(wgt, field)
                field = np.ma.array(field)
                 
                # Re-mask field
                if mask is not None:
                    field.mask = mask
                 
            except ValueError:
                raise CrossSectError('Area weights with shape %s could not be broadcast to field with shape %s' 
                                     % (wgt.shape, field.shape))
        
        # An extra error check in case either wgt.ndim > field.ndim or broadcasting isn't correct
        if (wgt.shape != field.shape):
            raise CrossSectError('Area weights with shape %s do not match field with shape %s' 
                                     % (wgt.shape, field.shape))
            
        # Average over axis (masked equivalent to mdtol = 1)
        field_mean = np.ma.average(field, axis, wgt)
        
        # Mask where the proportion of missing data > mdtol
        if mdtol < 1:
            frac_mask = np.sum(field.mask, axis, np.float) / field.shape[axis]        
            field_mean = np.ma.masked_where(frac_mask > mdtol, field_mean)
        
        return field_mean
    
    def get_face_area(self):
        '''
        Calculate the areas of the cell faces for the cross section given by self.field.
        
        Method:
            Read in horizontal and vertical scale factors from the mesh file, the horizontal 
            scale factor being parallel to the cross section axis, then multiply them.
            If doing a mean cross section, first create mean cross sections for the scale 
            factors before multiplying them.
        
        Returns:
             area- The area of the cell faces of the self.field cross section
        
        '''
            
        grid = self.items['grid'].lower()
         
        # Get e3 field
        e3 = self.get_e123('z')
        
        # Get the e1 or e2 field parallel to the cross-section axis
        e12 = self.get_e123(self.xsect_axis)
        
        # Broadcast prior to masking
        e12, e3 = np.broadcast_arrays(e12, e3)
        e12 = np.ma.array(e12)
        e3 = np.ma.array(e3)
        
        # Initialize area
        area = {}

        # Get mean axis
        if self.mean_axis is not None:
            axis = self.expt.mesh.get_e3(grid, return_dims=True)[self.mean_axis]
            
        # Loop over subdomains
        for domain in self.domains:
            
#NOTE: get_domain_mask is also called for each subdomain in read_field: This is duplication
#      and therefore more compute time, but the alternative is carrying 3D masks around 
            # Apply subdomain mask
            mask = self.get_domain_mask(domain)
            e3.mask = mask
            e12.mask = mask
            
            # If doing a mean cross section, do this for the scale factors before multiplying
            if self.mean_axis is not None:
                e3_dom = self.mean_over_axis(e3, axis)
                e12_dom = self.mean_over_axis(e12, axis)
                
                area[domain] = e3_dom * e12_dom
            else:
                area[domain] = e3 * e12
                
        return area

    def get_e123(self, axis):
        '''
        Read in a scale factor from the mesh file, then index by the points used to produce the cross section.
        
        Method:
            The full 2D or 3D scale factor field is read in, cropped to the domain of interest 
            and masked for points not in the cross section. 

        Arguments:
            axis-
                The axis parallel to the scale factor; one of ('x', 'y', 'z')

        Returns:
            e123-
                The scale factor field indexed by self.points, with dimensions of (z, x) if
                a path-type section or (z, y, x) if a mean section
        
        '''
        
        grid = self.items['grid'].lower()
        levels = np.arange(self.minlev, self.maxlev + 1)
        
        # Choice of mesh function and dimension indices
        if axis in ('x', 'y'):
            e123 = self.expt.mesh.get_e12(axis, grid)
            e123_dims = self.expt.mesh.get_e12(axis, grid, return_dims=True)
        elif axis in ('z'):
            e123 = self.expt.mesh.get_e3(grid, levels)
            e123_dims = self.expt.mesh.get_e3(grid, return_dims=True)
        
        index = [slice(None)] * e123.ndim
        index[e123_dims['y']] = self.points[0]
        index[e123_dims['x']] = self.points[1]
        
        # Mask and crop scale factor (does not mask land yet)
        e123, index = utils.mask_and_crop(e123, index)
        
        # If a path-type section, index so that broadcasting returns a (z, x) array 
        if self.mean_axis is None:
            e123 = e123[index]

        return e123

    def get_domain_mask(self, domain):
        '''
        Get subdomain mask for the cross section.
        
        Method:
            The subdomain mask is read from domain_masks.nc and then multiplied by the 
            global mask to give a 3D domain mask. The resulting field is cropped to the
            cross-section domain of interest.
            
        Arguments:
            domain-
                Name of the subdomain, as specified in domains.ini
        
        Returns:
            mask-
                A numpy.ndarray containing a boolean mask.
        '''
        
        grid = self.items['grid'].lower()
        levels = np.arange(self.minlev, self.maxlev + 1)
        
        # Get subdomain mask
        submask = self.expt.get_domain_mask(domain)

#TODO: Yes, this is indeed duplicate code! Remove once/if self.expt.get_domain_mask is moved to mesh_tools (ncid will remain open) 
        ncid = nc4.Dataset(self.expt.domain_masks)
        submask_dims = utils.query_dims(ncid.variables[domain])
        ncid.close()

        # Mask and crop subdomain mask        
        index = [slice(None)] * submask.ndim
        index[submask_dims['y']] = self.points[0]
        index[submask_dims['x']] = self.points[1]
        submask = utils.mask_and_crop(submask, index)[0]

#NOTE: I added this code for the global mesh because I don't think it should be assumed that the self.field data is masked.
#      Again, masking is something that need be done only once at the I/O level.
        # Get global 3D mask and its dimensions
        mask = self.expt.mesh.get_mask(grid, levels)
        mask_dims = self.expt.mesh.get_mask(grid, return_dims=True)
        
        # Mask and crop global mask
        index = [slice(None)] * mask.ndim
        index[mask_dims['y']] = self.points[0]
        index[mask_dims['x']] = self.points[1]
        mask, index = utils.mask_and_crop(mask, index)
        
        # Multiply together masks (and broadcast to 3D) and mask zero land values.
#FIXME: See above; might cause a MemoryError
        mask = np.ma.masked_equal(mask * submask, 0.)
        
        # If a path-type section, index so that broadcasting returns a (z, x) array 
        if self.mean_axis is None:
            mask = mask[index]

        return mask.mask
        
        
#TODO: Duplicated code with latlon
def decode_period(period):
    '''
    Parse period to work out what needs to be done
    ''' 
    
    # If individual months or seasons are specified then need to do monthly or seasonal averaging
    if period in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
        mean_period = '1m'
    elif period in ['djf', 'mam', 'jja', 'son']:
        mean_period = '1s'
    elif period == '1y':
        mean_period = '1y'
    else: raise CrossSectError('Period specified ('+period+') is not valid')

    return mean_period

