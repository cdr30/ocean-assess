# Calculate MOC on a NEMO grid
# Use [xy] min and max to define a region (e.g. Weddel Sea)
#
# Tim Graham Jan 2012
#

#Need to make this work with multiple fields per file

import numpy as np
import netCDF4 as nc4
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import subprocess
from plot_tools import time_series_plot as tsplot
from plot_tools import figure_setup
import plot_tools
from time import time
from FileDates import File_Dates
import utils
import io_utils

class MocError(Exception): pass

#TODO: Would be good to restructure this module so it better resembles others

def assess_moc(model, metric, items, out_dir):
    '''
    Carry out assessment of the MOC
    '''
    if metric['period'] not in [['1y'],['1m']]:
        raise MocError("For MOC period must be '1m' or '1y'")
    
    #Find out which mask this domain uses
    #Do we want to accept multiple domains in one call?
    plot_domain = metric['plot_domain'][0]
    mask = model.domains[plot_domain.lower()].mask
    
    #Initiate MOC classes
    runs = [moc(model, metric['period'][0], mask, model.f90dir + '/cdfmoc')]
    if model.cntl is not None: 
        runs.append(moc(model.cntl, metric['period'][0], mask, model.f90dir + '/cdfmoc') )
        
        do_model_cntl = check_cntl_grid(*runs)
    
#TODO: Split plot_type into separate top level functions
    # Make plots depending on what we want to do
    plot_type = metric['type']

    plot_tuple = utils.plot_tuple()
    table_tuple = utils.table_tuple()

    if plot_type == 'mean':
        plotfilename = plot_tools.figure_filename(model, metric, metric['period'][0], plot_domain, out_dir)

        if model.cntl is not None:
            if do_model_cntl:
                fig = figure_setup('four_panel')
                fig.subplots_adjust(hspace=0.2, wspace = 0.15)
                nx, ny = [2, 2]
            else:
                fig = figure_setup('two_panel')
                nx, ny = [2, 1]
        else:
            fig = figure_setup('one_panel')
            nx, ny = [1, 1]

        cmap = items.get('cmap_file', None)
        if cmap is not None: 
            cmap = plot_tools.read_cmap(cmap)
        else:
            cmap = plt.cm.jet
        cmap_diff = items.get('cmap_file_diff', None)
        if cmap_diff is not None: 
            cmap_diff = plot_tools.read_cmap(cmap_diff)
        else:
            cmap_diff = plt.cm.RdBu_r

        fill_levs = items.get('contour_levels_fill')
        if fill_levs is not None:
            exec "fill_levs=" + fill_levs
        line_levs = items.get('contour_levels_lines')
        if line_levs is not None:
            exec "line_levs=" + line_levs
        diff_levs = items.get('contour_levels_diff')
        if diff_levs is not None:
            exec "diff_levs=" + diff_levs
            
        # Plotting method
        method = items.get('contour_method', 'mesh')

        for i,run in enumerate(runs):
            plt.subplot(nx, ny, i + 1)
            run.plot_mean(date_range=[run.model.mean_start, run.model.mean_end], 
                          cfill_levs=fill_levs, cline_levs = line_levs, 
                          cmap=cmap, method=method)
            plot_tools.item_ax_options(plt.gca(), items)
            
        if (model.cntl is not None) and (do_model_cntl):
            plt.subplot(nx, ny, 3)
            runs[0].plot_diff(runs[1], date_range=[model.mean_start, model.mean_end],
                              date_range_ref=[model.cntl.mean_start, model.cntl.mean_end], 
                              cfill_levs=diff_levs, cline_levs=diff_levs, 
                              cmap=cmap_diff, method=method)
            plot_tools.item_ax_options(plt.gca(), items)
            
        main_title = '%s (%s)' 
        plt.suptitle(main_title % (items['title'], plot_domain), fontsize = 14)
        
        plt.savefig(plotfilename)
        plt.close(fig)
        #??? Not sure plot_tuple.fnames has to be a list? See run_metrics.py:69
        plotlist = [plot_tuple(metric['period'][0], [plotfilename])]
        tablelist = None
        
    elif plot_type == 'ts':
        # Make a time series plot
       
        plotfilename = plot_tools.figure_filename(model, metric, metric['period'][0], plot_domain, out_dir)
        
        # Plot a time series
        if 'latitude' in items.keys(): 
            lat = float(items['latitude'])
        else:
            lat=None
        fig = figure_setup('ts_one_panel')
        mean = []
        stdev = []
        ts_func = items.get('ts_func','max')
        for i, run in enumerate(runs):
            time_series = run.plot_ts(ts_func, date_range = None,
                                      lat_range=lat, color=['k','r'][i])
            mean.append(np.array(time_series).mean())
            stdev.append(np.array(time_series).std())
        
        main_title = '%s (%s)'
        plt.suptitle(main_title % (items['title'], plot_domain), fontsize = 14)
        plt.legend([run.model.runid for run in runs], fancybox=True, loc=0, fontsize=11)
        
        plt.gca().set_ylabel(items['ylabel'])
        plot_tools.item_ax_options(plt.gca(), items)
        plt.savefig(plotfilename)
        plt.close(fig)
        #???: Obs to be done?
        obs = None
        if model.cntl is not None:
            table = [[ "Metric" , model.runid+" mean", model.runid+" std_dev", model.cntl.runid+" mean", model.cntl.runid+" std_dev", "Obs" ],
                     [items["title"], mean[0], stdev[0], mean[1], stdev[1], obs ]]
        else:
            table = [[ "Metric" , model.runid+" mean", model.runid+" std_dev", "Obs" ],
                     [items["title"], mean[0], stdev[0], obs ]]
        #??? Not sure plot_tuple.fnames has to be a list? See run_metrics.py:69
        plotlist = [plot_tuple(metric['period'][0], [plotfilename])]
        tablelist = [table_tuple(metric['period'][0], table)]
            
    else:
        raise MocError("Metric type "+type+" not recognised")
    
    return plotlist, tablelist, None


def check_cntl_grid(run_1, run_2):
    '''
    Check that both models are on the same grid. 
    '''
    
    err = 'cannot create model - control plot'
               
    if not np.array_equal(run_1.lat, run_2.lat):
        io_utils.warn('Both models must be on the same horizontal grid; ' + err)
        return False
    if not np.array_equal(run_1.depth, run_2.depth):
        io_utils.warn('Both models must be on the same vertical grid; ' + err)
        return False
    
    return True


class moc:
    '''
    Class to calculate or read in MOC fields and extract or plot timeseries
    '''
    def __init__(self,model, period, mask, cdfmoc):
        try:
            self.name = model.description
            self.Vdates = model.filedates['V'][period]
            self.mocdir = model.datadirs['assess']+'/'+period+'/moc/'
            if not os.path.isdir(self.mocdir):
                os.makedirs(self.mocdir)
            mocfiles = glob.glob(self.mocdir+'/' + model.runid + '*moc.nc')
            self.mocfiledates = File_Dates(mocfiles)
            self.model = model
            self.mesh = model.meshmask
            self.maskfile = model.subbasins 
            self.get_lat()
            self.get_depth()
            self.mask = mask
            self.cdfmoc = cdfmoc
            self.mask_to_varname()
        except NameError:
            raise
    
    def __str__(self):
        return self.name+" MOC class"
   
    def fld_mask(self): 
        '''
        Integrate Vmask and basin mask to give a mask to use as missing data in MOC fields
        '''
        ncid_mesh = nc4.Dataset(self.mesh)
        vmask = ncid_mesh.variables['vmask']
        nz = len(ncid_mesh.dimensions['z'])
        ny = len(ncid_mesh.dimensions['y'])
        nx = len(ncid_mesh.dimensions['x'])
        
        #Use a slice so we don't need to worry about whether the mask has a time dimension
        ndims = len(vmask.dimensions)
        index = [slice(None)]*ndims
        zdim = vmask.dimensions.index('z')
        
        if self.mask is not None:
            basin_mask = nc4.Dataset(self.maskfile).variables[self.mask][:].squeeze()
        else:
            basin_mask = np.ones([ny,nx])
            
        fld_mask = np.zeros([nz, ny])
        for k in range(nz):
            index[zdim] = k
            fld_mask[k] = (vmask[tuple(index)].squeeze() * basin_mask).max(axis = -1)
        
        #Invert the mask
        fld_mask -= 1
        fld_mask *= -1
        
        return fld_mask
        
    def get_field(self,date):
        '''
        Read in MOC streamfunction from cdfmoc output file if it exists already
        Else call cdfmoc to calculate MOC and read in the data
        Inputs:
            date - a datetime object
        '''
        mocdates = self.mocfiledates.dates()
        vdates = self.Vdates.dates()

        if date in mocdates:
            ind = mocdates.index(date)
            (fname, r_index) = (self.mocfiledates[ind].filename, self.mocfiledates[ind].record)
        elif date in vdates:
            ind = vdates.index(date)
            (vfile, r_index) = (self.Vdates[ind].filename, self.Vdates[ind].record)
            fname = self.call_cdfmoc(vfile)
            
            #Append the file and date to file list (order doesn't matter here)
            filedates = File_Dates([fname])
            self.mocfiledates.extend(filedates)
        else:
            raise MocError('No MOC or Vfile for date '+date+'. Cannot calculate MOC')
        
        try:
            with nc4.Dataset(fname) as ncid:
                moc = ncid.variables[self.varname][r_index]
        except KeyError:
            raise MocError('Cannot read in MOC: Variable '+self.varname+' not in file '+fname)
        
        return moc
    
    def call_cdfmoc(self,vfile):
        '''
        Interface to cdfmoc code.
        '''
        fname = self.mocdir + '/' +os.path.basename(vfile).replace('.nc' , '_moc.nc') 
        cmd = [self.cdfmoc, vfile, self.mesh, self.mesh, self.maskfile, self.mesh, fname]
        try:
            t1=time()
            p1 = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE )
            (stdout, stderr) = p1.communicate()
            if p1.wait() != 0:
                if stderr:
                    raise MocError("cdfmoc failed with return code "+str(p1.wait()) +" and error message "+stdout)
                else:
                    raise MocError("cdfmoc failed with return code "+str(p1.wait()) )

#TODO: This more often than not is because the CDF tools aren't compiled. Need a specific error for this.
        except OSError as err:
            raise MocError('Call to CDFMOC failed with OSError: ',err)
        
        return fname
 
     
    def get_lat(self):
        try:
            with nc4.Dataset(self.mesh) as fileid:
                #Find which axis is x axis
                var = fileid.variables['gphiv']
                xdim = var.dimensions.index('x')
                self.lat = var[:].mean(axis = xdim).squeeze()
        except KeyError:
            raise MocError("Variable gphiv not found in "+self.mesh) 
        except IOError:
            raise MocError("Cannot open meshfile ("+self.mesh+") to find latitude")
        
    def get_depth(self):
        '''
        Read depth coordinate from mesh mask.
        '''
        
        # Get depth variable
        self.depth = self.model.mesh.get_depth('w')

    def plot_ts(self,ts_func, date_range=None, lat_range=None,depth_range=None,linestyle='-',verbose=False,**kwargs):
        '''
        Extract and plot a time series of MOC
        Options: 
            lat_range: single value or 2 element tuple/list restricting the 
                        latitude range from which value is taken
            depth_reange: single value or 2 element tuple/list restricting 
                            the depth range from which the value is taken 
        '''
        time_series=[]
        lat_max_vals=[]
        depth_max_vals=[]
        depth_zero_vals=[]
        lat=self.lat
        depth=self.depth
        self.Vdates.sort() #Make sure they are in order
        
        if date_range is not None:
            dates = self.Vdates.in_range(date_range[0], date_range[1]).dates()
            
            if len(dates) == 1:
                err = 'Date range [%s - %s] results in only one data point'
                raise MocError(err % (date_range[0], date_range[1]))
        else:
            dates = self.Vdates.dates()
            
        for date in dates:
            field = self.get_field(date)
            if ts_func == 'depth_zero':
                depth_zero_vals.append(moc_depth_zero(field,lat,depth,lat_range))
            else:
                (max_val,lat_max1,depth_max1)=moc_max(field,lat,depth,lat_range,depth_range)
                time_series.append(max_val)
                lat_max_vals.append(lat_max1)
                depth_max_vals.append(depth_max1)
        
        if ts_func == 'lat_max':
            ts = lat_max_vals
        elif ts_func ==  'depth_max':
            ts = depth_max_vals
        elif ts_func == 'depth_zero':
            ts = depth_zero_vals
        else:
            ts = time_series

        tsplot(ts, dates, linestyle=linestyle, **kwargs)
        
        return ts
        
    def plot_mean(self, date_range = None , cline_colors = 'k', cfill_levs=None, cline_levs=None, method='mesh', **kwargs):
        '''
        Plot Mean of MOC fields
        '''
        
        if date_range is None:
            dates = self.Vdates
        elif isinstance(date_range,(list, tuple) ) and len(date_range) == 2:
            dates = self.Vdates.in_range(date_range[0], date_range[1])
        else:
            raise ValueError("In moc.plot_mean: date_range must be a 2 element list or tuple")
        
        #Chamge file_dates tuple to give list of dates
        dates = dates.dates()
        
        plt_field=self.get_field(dates[0])
        
        #???: Why not meaning.make_mean?
        for date in dates[1:]:
            plt_field += self.get_field(date)
        plt_field /= len(dates)
        
        plt_field = np.ma.array(plt_field.squeeze(), mask = self.fld_mask() )
        
        plot_tools.plot_xz_fill(self.lat, self.depth, plt_field, levels=cfill_levs, 
                                hold=True, method=method, **kwargs)
        
        kwargs['cmap'] = None
        plt.contour(self.lat, self.depth, plt_field, colors = cline_colors, levels=cline_levs, **kwargs)
        plt.title(self.model.runid)
                

    def plot_diff(self, moc_ref, date_range=None, date_range_ref=None , cfill_levs=None, cline_levs=None, cline_colors='k', method='mesh',**kwargs):
        '''
        Make a plot of MOC streamfunction difference between this moc instance and moc_ref
        Inputs:
            moc_ref - An MOC class to mean and subtract from this one
            date_range - 2 element list or tuple of start and end date for meaning period of this MOC
            date_range_ref - 2 element list or tuple of start and end date for meaning period of reference model MOC
        '''
        
        if date_range is None:
            dates = self.Vdates
        elif isinstance(date_range,(list, tuple) ) and len(date_range) == 2:
            dates = self.Vdates.in_range(date_range[0], date_range[1])
        else:
            raise ValueError("In moc.plot_mean: date_range must be a 2 element list or tuple")
        if date_range_ref is None:
            dates_ref = moc_ref.Vdates
        elif isinstance(date_range_ref,(list, tuple) ) and len(date_range) == 2:
            dates_ref = moc_ref.Vdates.in_range(date_range_ref[0], date_range_ref[1])
        else:
            raise ValueError("In moc.plot_mean: date_range_ref must be a 2 element list or tuple")
        
        #Change file_dates tuple to give list of dates
        dates = dates.dates()
        plt_field=self.get_field(dates[0])
        for date in dates[1:]:
            plt_field += self.get_field(date)
        plt_field /= len(dates)
        plt_field = np.ma.array(plt_field.squeeze(), mask = self.fld_mask() )
 
        #Change file_dates tuple to give list of dates
        dates_ref = dates_ref.dates()
        plt_field_ref=moc_ref.get_field(dates_ref[0])
        for date in dates_ref[1:]:
            plt_field_ref += moc_ref.get_field(date)
        plt_field_ref /= len(dates_ref)
        plt_field_ref = np.ma.array(plt_field_ref.squeeze(), mask = moc_ref.fld_mask() )
        
        plt_field -= plt_field_ref
        
        plot_tools.plot_xz_fill(self.lat, self.depth, plt_field, levels=cfill_levs, 
                                hold=True, method=method, set_middle=True, **kwargs)
        
        kwargs['cmap'] = None
        plt.contour(self.lat, self.depth, plt_field, colors = cline_colors, levels=cline_levs, **kwargs)
        plt.title('%s - %s' % (self.model.runid, moc_ref.model.runid))
        
    def mask_to_varname(self):
        #???: zomsfatl is used if Atlantic domain, otherwise global streamfunction used. Why?
        self.varname = {'tmaskatl' : 'zomsfatl'}.get(self.mask, 'zomsfglo')
    
    
def moc_max(field,lat=None,depth=None,lat_range=None,depth_range=None):
    '''
    Extract a point MOC value in an MOC field (must be an np.MaskedArray) with coordinates lat and depth.
    If lat_range and/or depth_range are specified as tuples (or 2 element lists) then the search 
    will be restricted to those range(s).
    If lat_range and/or depth_range are single values then value will be at nearest point.
    ''' 
    
    #Check if field is a Masked Array
    if not (np.ma.isMA(field) or isinstance(field, np.ndarray)): 
        raise TypeError("In nemo_moc.moc_max: field must be a masked array")
    
    field = field.squeeze()
    ny=field.shape[-1]
    nz=field.shape[-2]
    
    #If lat_range is provided limit search to this range 
    if lat_range is not None:
        if isinstance(lat_range,(list,tuple)):
            lat_ind=[abs(lat-latval).argmin() for latval in lat_range]
            if len(lat_ind)==1 : lat_ind.append(lat_ind[0]+1)
        elif isinstance(lat_range,(float,int)):
            lat_ind=abs(lat-lat_range).argmin()
            lat_ind=[lat_ind,lat_ind+1]
        else:
            raise TypeError("In nemo_moc.moc_max: lat_range must be list,tuple,float or integer")
            #Need error checking here
    else:
        lat_ind=[0,ny]
    
    #If depth_range is provided limit search to this range 
    if depth_range is not None:
        if isinstance(depth_range,(list,tuple)):
            depth_ind=[abs(depth-depthval).argmin() for depthval in depth_range]
            if len(depth_ind)==1 : depth_ind.append(depth_ind[0]+1)
        elif isinstance(depth_range,(float,int)):
            depth_ind=abs(depth-depth_range).argmin()
            depth_ind=[depth_ind,depth_ind+1]
        else:
            raise TypeError("In nemo_moc.moc_max: lat_range must be list,tuple,float or integer")
            #Need error checking here
    else:
        depth_ind=[0,nz]
    
    #This will fail if lat or depth are not provided
    #Should add error handling  
    #???: A slice object might look cleaner here (i.e. work from a field_max)
    lat_max=lat[field[... , depth_ind[0]:depth_ind[1],lat_ind[0]:lat_ind[1]].max(axis=-2).argmax()+lat_ind[0]]
    depth_max=depth[field[... , depth_ind[0]:depth_ind[1],lat_ind[0]:lat_ind[1]].max(axis=-1).argmax()+depth_ind[0]]
        
    return field[depth_ind[0]:depth_ind[1],lat_ind[0]:lat_ind[1]].max(),lat_max,depth_max


def moc_depth_zero(field,fld_lat,fld_depth,lat):
    
    #Check if field is a Masked Array
    if not (np.ma.isMA(field) or isinstance(field, np.ndarray)): 
        raise TypeError("In nemo_moc_tools.moc_depth_zero: field must be a numpy array or masked array")
    
    #If lat is provided limit search to this range 
    if isinstance(lat,(float,int)):
        lat_ind=abs(fld_lat-lat).argmin()
    else:
        raise TypeError("In nemo_moc.moc_max: lat must be float or integer")
    
    #May have to think more carefully about upper ocean cells if this is to be used in tropics
    ind=min(np.where(field[:,lat_ind] < 0)[0])
    
    #Linearly interpolate between adjacent values
    #TODO: put this on a separate line
    return fld_depth[ind-1]+(fld_depth[ind]-fld_depth[ind-1])*field[ind-1,lat_ind]/(field[ind-1,lat_ind]-field[ind,lat_ind])
    
