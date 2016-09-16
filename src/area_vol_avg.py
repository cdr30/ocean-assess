'''
Create time series of area or volume weighted averages of variables

Created on Nov 18, 2013

@author: hadtd
'''
import subprocess
from FileDates import File_Dates
import os
import netCDF4 as nc4
import numpy as np
from plot_tools import time_series_plot, figure_setup
import plot_tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils

class AreaVolAvgError(Exception): 
    pass

def area_avg_ts(model, metric, items, out_dir):
    '''
    Simply calls vol_avg_ts but with a more user friendly name
    '''
    return vol_avg_ts(model, metric, items, out_dir)

def vol_avg_ts(model, metric, items, out_dir):
    '''
    Plot time series plots of area average
    '''
    #Check levels in items
    lev_range = [ int(items['minlev']), int(items['maxlev'])]
    
    #Check domains
    if metric['plot_domain'] == ['all']:
        plot_domains = model.domains.keys()
    else: 
        plot_domains = metric['plot_domain']

    # If a domain mask has no valid points, cdfmeanvar fails with an unintuitive error.
    # We exclude these domains, and if no domains are left then we return a metric failure.
    with nc4.Dataset(model.domain_masks) as ncid:
        for domain in plot_domains:
            if ncid.variables[domain][:].sum() == 0:
                plot_domains.remove(domain)

    if len(plot_domains) == 0:
        err = 'No subdomains found with unmasked points'
        raise AreaVolAvgError(err)
    
    colors = ['k','r']
    
    plotlist = []
    plot_tuple = utils.plot_tuple()
    #Loop through periods
    for period in metric['period']:
        thisplotlist = []
        runs = [Avg_TS(model, metric, period, items['variable'], items['grid'], lev_range, plot_domains) ]
        if model.cntl is not None:
            runs.append(Avg_TS(model.cntl, metric, period, items['variable'], items['grid'], lev_range, plot_domains))
        
        for plot_domain in plot_domains:
            fig = figure_setup('ts_one_panel')
            for i, run in enumerate(runs):
                run.plot_ts(plot_domain, color = colors[i])

            plt.legend([run.model.runid for run in runs], fancybox=True, loc=0, fontsize=11)
            
            plotfilename = plot_tools.figure_filename(model, metric, period, plot_domain, out_dir)

            plt.gca().set_title(' '.join([plot_domain, items['title']]))
            y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            plt.gca().yaxis.set_major_formatter(y_formatter)
            plot_tools.item_ax_options(plt.gca(), items)
            fig.savefig(plotfilename)
            plt.close(fig)
            thisplotlist.append(plotfilename)
        plotlist.append(plot_tuple(period, thisplotlist))
        
    return plotlist, None, None

class Avg_TS(object):
    
    def __init__(self, model, metric, period, variable, grid, lev_range, plot_domains ):
        self.model = model
        self.period = period 
        self.variable = variable
        self.metric = metric['metric']
        self.grid = grid
        self.lev_range = lev_range
        self.plot_domains = plot_domains
        self.mesh = model.meshmask
        self.domain_file = model.domain_masks
        self.plot_domains = plot_domains
        self.f90dir = model.f90dir

        self.assess_dir = '/'.join([model.datadirs['assess'], period, 'area_vol_avg' ]) 
        if not os.path.isdir(self.assess_dir): 
            os.makedirs(self.assess_dir)

        self.meanfile = self.assess_dir + '/' + \
            '_'.join([model.runid + 'o', period, "%s", self.metric ]) + '.nc'

        self.in_file_dates = model.filedates[grid][period]

        self._get_data()
            
    def _call_cdfmeanvar(self, infile, domains):
        '''
        Call cdfmeanvar to calculate mean of field contained in input file
        '''
        cmd = self.f90dir+'/cdfmeanvar'
        cmd = [cmd, infile, self.variable, self.grid, self.mesh, self.mesh, self.mesh, str(self.lev_range[0]+1), 
               str(self.lev_range[1]+1), self.domain_file ]
        cmd.extend(domains)
        
        try:
            p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (results, stderr) = p1.communicate()
            #Parse stdout (renamed results) to read means
            code = p1.wait()
            if code != 0:
                raise AreaVolAvgError('Call to CDFMEANVAR failed with error "' + stderr+'"')
            lines = results.strip().split('\n')
            means = {}
            for domain in domains: 
                means[domain] = []
            for line in lines:
                if line.split()[0] in domains:
                    means[line.split()[0]].append(line.split()[1])
             
            return means  
        except OSError as err:
            raise AreaVolAvgError('Call to CDFMEANVAR failed with OSError: ', err)
        except Exception as err:
            raise err
    
    def _get_data(self):
        '''
        Loop through dates in input files
        If metric exists in output file for that date then use it.
        If not calculate it (by calling call_cdfmeanvar)
        Output files are named by metric and domain name
        '''
        
        domain_file_dates = {}
        domain_files = {}
        domain_dates = {}
        for domain in self.plot_domains: 
            domain_file = self.meanfile % domain
            domain_files[domain] = domain_file
            if os.path.isfile(domain_file):
                domain_file_dates[domain] = File_Dates([domain_file])
                domain_dates[domain] = domain_file_dates[domain].dates()
            else:
                self._create_ts_file(domain_file)
                domain_file_dates[domain] = []
                domain_dates[domain] = []
        
        data = {}
        for domain in self.plot_domains: 
            data[domain] = []
        
        for file_date in self.in_file_dates:
            date = file_date.date
            domains_to_calc = []
            for domain in self.plot_domains:
                if date in domain_dates[domain]:
                    ind = domain_dates[domain].index(date)
                    data[domain].append(self._read_ts_file(domain_file_dates[domain][ind]))
                else:
                    domains_to_calc.append(domain)
            
            #Call _call_cdfmeanvar if required
            if len(domains_to_calc) !=0:
                tmp_data = self._call_cdfmeanvar(file_date.filename, domains_to_calc)
                this_file_dates = File_Dates( file_date.filename )
                
                #Append new data to netcdf file so other dates in the file can be read later
                for domain in domains_to_calc:
                    new_fdates = self._append_ts_file(domain_files[domain], 
                                                     tmp_data[domain], 
                                                     this_file_dates.dates())
                    domain_file_dates[domain] = new_fdates
                    domain_dates[domain] = new_fdates.dates()
                    ind = domain_dates[domain].index(date)
                    data[domain].append(self._read_ts_file(domain_file_dates[domain][ind]))
        
        self.data = data
    
    def _create_ts_file(self,fname):
        '''
        Create a new netcdf file in which to store time series output
        Set up dimensions and variable
        '''
        
        try:
            with nc4.Dataset(fname, 'w') as ncid:
                #Create time counter dimension
                ncid.createDimension('time_counter')
                time_var = ncid.createVariable('time_counter', 'd', dimensions=('time_counter'))
                #Should add some attributes here
                ncid.createVariable(self.metric, 'f', dimensions=('time_counter'))
                
                #Open the first netcdf model file and use time attributes from its time_counter variable
                #This should be the earliest time in the model run
#IDEA: Could also use meaning.copy_atts
                with nc4.Dataset(self.in_file_dates[0].filename) as ncid_in:
                    time_in = ncid_in.variables['time_counter']
                    ncattrs = {}
                    for attr in time_in.ncattrs():
                        ncattrs[attr] = time_in.getncattr(attr)
                    time_var.setncatts(ncattrs)
            
                
        except Exception as err:
            raise AreaVolAvgError("Could not create NetCDF file", err)
    
    def _read_ts_file(self, file_date):
        '''
        Read data from a netcdf file at time described by file_date tuple
        '''
        with nc4.Dataset(file_date.filename) as ncid:
            return ncid.variables[self.metric][file_date.record]                    
    
    
    def _append_ts_file(self, fname, data, dates): 
        '''
        Append data to netcdf file
        Inputs: 
            fname - An existing netcdf file to write to
            data -  List of data to write to the file
            dates - List of netcdftime objects corresponding to the data points 
        Returns:
            new_dates - A list of file_date tuples corresponding to the updated file
        '''
        
        try:
            with nc4.Dataset(fname, 'r+') as ncid:
                old_data = ncid.variables[self.metric][:].copy() #Not sure if .copy() is needed but safer to use it
                time_var = ncid.variables['time_counter']
                old_times = time_var[:].copy()
                
                #Convert the new dates to numbers using the units and calendar in the file
                new_times = nc4.date2num(dates, time_var.units, time_var.calendar)
                
                # append new data to old data
                old_data = np.concatenate([old_data,data])
                old_times = np.concatenate([old_times,new_times])
                
                #Sort to put in order of time
                order = np.argsort(old_times)
                old_times = old_times[order]
                old_data = old_data[order]
                
                #Write data to file
                time_var[:] = old_times
                ncid.variables[self.metric][:] = old_data
       
        except Exception as err:
            raise AreaVolAvgError("Writing data to file "+fname+" failed",err)
        
        # Update file_dates for file
        new_dates = File_Dates(fname)
        
        return new_dates
    
    def plot_ts(self, domain, **kwargs):
        '''
        Make a time series plot on the current axis
        Inputs:
            domain - name of the domain to plot
        Option inputs - Any keyword arguments accepted by pyplot.plot
        '''
        
        #Check that domain exists in self.data
        if domain not in self.data.keys():
            raise AreaVolAvgError("In plot_ts: Invalid domain specified")
        
        dates = [fdate.date for fdate in self.in_file_dates]
        
        (ax,line) = time_series_plot(self.data[domain], dates, **kwargs)
        
        return (ax,line)
        
