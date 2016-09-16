'''
Created on Nov 6, 2013

@author: hadtd
'''
import model as model_class
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from netCDF4 import netcdftime as nctime
from datetime import date as dtdate
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
import numpy as np
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator
import os
import io_utils

class PlotToolsError(Exception): pass


def time_series_plot(ts, dates, ax = None, linestyle = '-', marker=None, hold=True, **kwargs):
    '''
    Function time_series_plot
        Produce a time series plot
    Inputs:
        ts = a 1D numpy array of data
        dates = a list of datetime objects (these can be netcdf datetime objects)
    Method:
        use matplotlib.pyplot.plot_date and matplotlib.dates to make the plot
        If the time series is from a 360 day calendar and contains 30 February (and 29th February in non leap years) 
        these dates will be removed
    '''
    
    if isinstance(dates[0], nctime.datetime):
        #Need to loop through dates and convert to standard datetime objects so that they will work with mdates  
        new_dates=[]
        new_ts=[]
        for i,date in enumerate(dates):
            try:
                new_dates.append(dtdate(date.year, date.month, date.day))
                new_ts.append(ts[i])
            except ValueError as err:
                #This could be because date is 29th/30th Feb
                if (err.message == 'day is out of range for month') and (date.month == 2) and (date.day in [29, 30]):
                    raise UserWarning("Time series plots don't support 360 day calendar yet")
                    raise UserWarning("Will skip 30th Feb and 29th Feb in non-leap years")
                else:
                    raise err
    elif isinstance(dates[0], datetime.datetime):
        new_dates = dates 
        new_ts = ts
    else:
        raise TypeError("dates must be a list or array of netcdftime.datetime objects")
    
    plotdates=mdates.date2num(new_dates)
    
    if ax is None:
        ax = plt.gca()
        if hold and not ax.ishold():
            ax.hold()
    
    hndl = ax.plot_date(plotdates, new_ts, linestyle=linestyle, marker=marker, **kwargs)
    
    return ax, hndl


def cm_fig(width_cm,height_cm):
    '''
    Create a figure using input arguments of width and height in cm.
    Returns the figure handle
    '''
    inches_per_cm = 1 / 2.58               # Convert cm to inches
    fig_width = width_cm * inches_per_cm # width in inches
    fig_height = height_cm * inches_per_cm       # height in inches
    fig_size = [fig_width, fig_height]
    f=plt.figure(figsize=fig_size)
    return f


def figure_setup(fig_type):
    
    #Could have a figures.ini file to do this?
    #Shouldn't really need to change it much?
    if fig_type == 'ts_two_panel': #Two time series subplots
        fig = cm_fig(30, 20)
        fig.subplots_adjust(hspace = 0.15, left = 0.1, right = 0.95, top = 0.9)
    elif fig_type == 'ts_one_panel': #A single time series plot (use a wide but short window)
        fig = cm_fig(30, 12)
    elif fig_type == 'four_panel': #Four panel plot (validation note style plots, MOC s.f.)
        fig = cm_fig(30,20)
        fig.subplots_adjust(hspace = 0.15, wspace = 0.1, left = 0.1, right = 0.95, top = 0.9)
    elif fig_type == 'two_panel': #Two panel plot (e.g. Single model field and model bias)
        fig = cm_fig(17, 20)
        fig.subplots_adjust(hspace = 0.15, left = 0.1, right = 0.95, top = 0.9)
    elif fig_type == 'one_panel': #One panel plot (e.g. MOC streamfunction)
        fig = cm_fig(17, 12)
        fig.subplots_adjust(top = 0.85)
    else: 
        raise PlotToolsError('fig_type '+fig_type+' not recognised')
    
    return fig


def figure_filename(model, metric_dict, period, domain, out_dir):
    '''
    Create a filename for the figure.

    '''

    if type(model) is not model_class.Model:
        raise PlotToolsError('model must be an instance of the model.Model class')

    if metric_dict['obsname'] == 'None':
        obsname = None
    else:
        obsname = 'vs_%s' % metric_dict['obsname']

    if model.cntl is None:
        cntl_runid = None
    else:
        cntl_runid = model.cntl.runid

    fname = '_'.join(i for i in [model.runid, cntl_runid, period,  
                      '%4i%02i%02i' %(model.mean_start.year, model.mean_start.month, model.mean_start.day),
                      '%4i%02i%02i' %(model.mean_end.year,   model.mean_end.month,   model.mean_end.day),
                      metric_dict['type'], metric_dict['metric'], obsname, domain] if i is not None ) + '.png'

    return '%s/%s' % (out_dir, fname)


def item_ax_options(ax, item):
    '''
    Process the plot_options keyword if set by the metric item.
    
    '''
    
    plot_options = item.get_list('plot_options', None)
    warn = 'plot_options entry "%s" is not valid'

    if plot_options is not None:
        opts = [i.strip() for i in plot_options]
    else:
        opts = []
        
    for opt in opts:
        try:
            exec "ax." + opt
        except:
            io_utils.warn(warn % opt)


def read_cmap(cmap_file):
    '''
    Read an ascii file of RGB colours and create a color map
    Inputs: cmap_file:  Either full path to a cmap_file or a cmap_file in cmaps directory.
                        The file should contain a list of rgb colours 
    '''
    
    if os.path.exists(cmap_file): 
        fname = cmap_file
    else:
        fname = os.path.abspath(os.path.dirname(__file__))+'/../cmaps/'+cmap_file
    
    try:
        with open(fname) as fid:
            rgb = [np.array( line.strip().lstrip('[').rstrip(']').strip().split(), dtype=float) for line in fid]
    except IOError as err:
        #Not sure that this error reporting is quite right?
        raise PlotToolsError("Could not find cmap_file "+cmap_file,err)
    
    return mcolors.ListedColormap(rgb)


def _get_default_levels(field, nlevs=25, mnfld=None, mxfld=None, **kwargs):
    '''
    Get default contour levels.
    '''
    
    # Get max and min from data if not provided
    if mxfld is None: 
        mxfld = np.ma.max(field)
    if mnfld is None: 
        mnfld = np.ma.min(field)
    
    locator = MaxNLocator(nlevs + 1)
    locator.create_dummy_axis()
    locator.set_bounds(mnfld, mxfld)
    levs = locator()

    return levs


def plot_field(x, y, field, method='mesh', levels=None,
               set_middle=False, cmap=plt.cm.jet, transform=None, **kwargs):
    '''
    Plot a 2D field.
        
    Arguments:
        x-
            Array of x coordinate values.
        y-
            Array of y coordinate values.
        field-
            Array of data.
        
    Keywords:
        method-
            The method to use for plotting: 'mesh' or 'filled'.
        levels-
            The contours / boundaries to plot. 
        set_middle-
            Force the use of colours in the middle of the colour map.
        nlevs-
            The number of contours / boundaries to plot (overridden by levels).
        mnfld-
            The lower bound of contours to draw when using nlevs.
        mxfld-
            The upper bound of contours to draw when using nlevs.
        **kwargs-
            Any other keywords into contourf or pcolormesh.
    '''

    # Contour levels
    if levels is None:
        levels = _get_default_levels(field, **kwargs)
    
    # Default transform
    if transform is None:
        transform = plt.gca().transData
        
    # Set colours
    norm = ExBoundaryNorm(levels, ncolors=cmap.N, set_middle=set_middle)
        
    # Do the plotting
    if method == 'filled':
        clev = plt.contourf(x, y, field,
                            levels=levels, extend='both',
                            norm=norm, transform=transform, cmap=cmap, **kwargs)
    elif method == 'mesh':
        clev = plt.pcolormesh(x, y, field, 
                              norm=norm, transform=transform, cmap=cmap, **kwargs)
    else:
        raise PlotToolsError('Plot method "{}" not recognised'.format(method))
    
    return clev


def plot_xz_fill(x, z, field, subplot=111, hold=False, 
                 colorbar='horizontal', **kwargs):
    '''
    Plot a field described by a depth and horizontal coordinate.
    
    Arguments:
        x-
            Array of horizontal coordinate values.
        z-
            Array of depth coordinate values.
        field-
            Array of data.
        
    Keywords:
        subplot-
            A matplotlib subplot argument.
        hold-
            Hold the axes (for overplotting).
        colorbar-
            An 'orientation' argument for the colorbar.
        **kwargs-
            Any other keywords into plot_field, contourf or pcolormesh.
    '''
    
    # Hold the axes if required
    if hold:
        ax = plt.gca()
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot)
        
    # Do the plotting
    clev = plot_field(x, z, field, **kwargs)
    
    # Further adjustments
    ax.set_axis_bgcolor((0.77, 0.74, 0.77))
    
    # Colour bar
    if colorbar:
        plt.colorbar(clev, orientation=colorbar, extend='both')
    
    
def plot_xy_fill(lon, lat, field, projection=ccrs.PlateCarree(), 
                 subplot=111, hold=False, colorbar='horizontal', 
	             **kwargs):
    '''
    Plot a field described by latitude-longitude coordinates.
    
    Arguments:
        lon-
            Array of longitude values.
        lat-
            Array of latitude values.
        field-
            Array of data.
        
    Keywords:
        projection-
            The projection to use.
        subplot-
            A matplotlib subplot argument.
        hold-
            Hold the axes (for overplotting).
        colorbar-
            An 'orientation' argument for the colorbar.
        **kwargs-
            Any other keywords into plot_field, contourf or pcolormesh.
    '''
 
    if type(field) is np.ndarray:
        field = np.ma.array(field)
 
    # Plot data either side of the E-W fold to deal with contour wrap-around.
    wh1 = (lon >= 0) | (field.mask == True)
    wh2 = (lon < 0) | (field.mask == True)
    fld1 = field.copy() 
    fld1.mask = wh1
    fld2 = field.copy() 
    fld2.mask = wh2
    
    # Some additional treatment is needed for pcolormesh
    lon2 = lon.copy()
    lon2[lon < -10] = 180.
    
    # Hold the axes if required
    if hold:
        ax = plt.gca()
    else:
        fig = plt.gcf()
        ax = fig.add_subplot(subplot, projection=projection)

    # Default levels must be determined for the global field
    if kwargs.get('levels', None) is None:
        kwargs['levels'] = _get_default_levels(field, **kwargs) 
        
    # Do the plotting   
    plot_field(lon, lat, fld1,
               transform=ccrs.PlateCarree(), **kwargs)
    clev = plot_field(lon2, lat, fld2,
                      transform=ccrs.PlateCarree(), **kwargs)
    
    # Set coastlines and land
    if not _is_jasmin_host():
        ax.coastlines()
        ax.stock_img()
    
    # Axis formatting
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.xlines = gl.ylines = gl.xlabels_top = gl.ylabels_right = False
    gl.xlabel_style = gl.ylabel_style = {'size':9, 'color':'gray'}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    # Colour bar
    if colorbar:
        plt.colorbar(clev, orientation=colorbar, extend='both')


#HACK: Temporary host query for Cartopy issue
def _is_jasmin_host():
    import socket
    _JASMIN_hosts = ['ceda.ac.uk', 'rl.ac.uk']
    
    for host in _JASMIN_hosts:
        if socket.gethostname().endswith(host):
            return True 
    
    return False


class ExBoundaryNorm(mcolors.BoundaryNorm):
    '''
    Subclass of BoundaryNorm that reserves some end colours for extrema.
    '''
    
    def __init__(self, boundaries, ncolors, clip=False, set_middle=False):
        '''
        Pick evenly-spaced colours for the given boundaries, reserving space at
        either end of the colour map for extrema.
        
        Arguments:
            boundaries-
                The boundaries for which to set colours.
            ncolors-
                The number of colours to use.
                
        Keywords:
            clip-
                Not used.
            set_middle-
                Set the middle boundaries to the central colour value.
        '''
         
        self.set_middle = set_middle
        self.ncolors = ncolors
        self.space = ncolors / len(boundaries)
        
        super(ExBoundaryNorm, self).__init__(boundaries, ncolors - 2 * self.space)
        
        # Determine mid points for odd and even numbers of boundaries
        if self.N % 2 != 0:
            self.mid = [self.boundaries[self.N / 2 + i] for i in (-1, 1)]
        else:
            self.mid = [self.boundaries[self.N / 2 + i] for i in (-1, 0)]
        
    def __call__(self, x, clip=None):
        '''
        Return colours for the requested boundaries.
        
        '''
        
        # Do not modify colours returned for masked data
        if np.ma.is_masked(x):
            not_masked = ~x.mask
        else:
            not_masked = slice(None)
        
        col = super(ExBoundaryNorm, self).__call__(x) 
        col_valid = col[not_masked]
        x_valid = x[not_masked]
        
        # Offset non-extrema colours
        col_valid += self.space
        col_valid[x_valid < self.vmin] = -1
        col_valid[x_valid >= self.vmax] = self.ncolors
        
        # Set middle colours
        if self.set_middle:
            in_mid = np.where((self.mid[0] <= x_valid) & 
                              (x_valid < self.mid[-1]))
            col_valid[in_mid] = self.ncolors / 2
        
        col[not_masked] = col_valid
        
        return col
    
        
