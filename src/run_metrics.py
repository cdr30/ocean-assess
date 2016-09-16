'''
Created on Nov 1, 2013


@author: hadtd
'''

from nemo_moc_tools import assess_moc
from cross_section_plot import cross_section
from area_vol_avg import area_avg_ts, vol_avg_ts
import latlon
import webpage
import os
import io_utils
try:
    import configparser as ConfigParser #Python version 3 and later
except ImportError:
    import ConfigParser #Python version 2.7

#Error handling
class run_metricsFatalError(Exception): pass
class run_metricsError(Exception): pass

def run(expt):
    '''
    Carry out each of the metrics specified in metrics 
    '''
    #Make output directory (if it doesn't already exist)
    out_dir = make_out_dir(expt)
    
    # Set up the logger
    logger = io_utils.Logger(expt, out_dir, 'ocean_assess_errors.log')
    
    # Set up the webpage
    html_out = webpage.Webpage(expt)

    # Loop over metrics
    for metric in expt.metrics:
        
        try:
            #Find out what to do for this metric
            items = read_items(metric['metric'], expt.item_file)
            
            #Find out the process associated with this metric type (from items)
            proc = items[metric['type']]

#IDEA: I think this ought to be in a separate function. Since these must be top-level functions
#      of a metric module, the checking can be reduced to a single line.
            if proc == 'lat_lon_mean':
                (plots,table,csv_dat) = latlon.lat_lon_mean(expt, metric, items, out_dir)
            elif proc == 'lat_lon_max':
                (plots,table,csv_dat) = latlon.lat_lon_max(expt, metric, items, out_dir)
            elif proc == 'lat_lon_min':
                (plots,table,csv_dat) = latlon.lat_lon_min(expt, metric, items, out_dir)
            elif proc == 'lat_lon_argmax':
                (plots,table,csv_dat) = latlon.lat_lon_argmax(expt, metric, items, out_dir)
            elif proc == 'lat_lon_argmin':
                (plots,table,csv_dat) = latlon.lat_lon_argmin(expt, metric, items, out_dir)
            elif proc == 'area_avg':
                (plots,table,csv_dat) = area_avg_ts(expt, metric, items, out_dir)
            elif proc == 'vol_avg':
                (plots,table,csv_dat) = vol_avg_ts(expt, metric, items, out_dir)
            elif proc == 'assess_moc':
                (plots,table,csv_dat) = assess_moc(expt, metric, items, out_dir)
            elif proc == 'cross_section':
                (plots,table,csv_dat) = cross_section(expt, metric, items, out_dir)
            else:
                raise run_metricsError("Process "+proc+" not known")
            
            #Change filenames to relative filenames
            # TODO - should this check the filenames and just remove the output directory from the start
            if isinstance(plots,list): 
                for i, plot in enumerate(plots):
                    basenames = [os.path.basename(fname) for fname in plot.fnames]
                    plots[i] = plot._replace(fnames = basenames)
            else: 
                basenames = [os.path.basename(fname) for fname in plots.fnames]
                plots = plots._replace(fnames = basenames)

#IDEA: Values of metric['type'] are not standardized except in a few older parts of the code
#      (e.g. moc). There isn't really a reason to do so either, as it is 
#      items[metric['type']] that must be valid and it is otherwise unnecessarily restrictive. 
#      I propose that the type info below be passed back from the top-level functions instead.
            type_title = {'ts':'timeseries', 'mean':'mean'}.get(metric['type'],'plot')
            title = '%s: %s' % (type_title, items['title'])

            html_out.metric(title, plots, table) 
        
        except run_metricsFatalError as err:
            #For this error code should stop as it affects all metrics
            raise err
        except Exception as err:
            #Any errors only affecting this metric can be dealt with by error handler
            logger.write_error(metric['metric'], err)
            
        # Write any warnings to logger
        logger.write_warnings(metric['metric'])
        
#TODO: We need an option to append metrics to a webpage. This requires inheriting existing plots
#      and table metrics. There are two ways to do this:
#        1. Create a web page parser to extract this information
#        2. Store this information in a netCDF file
    # Finish off by writing webpage to file
    html_out.write(out_dir + "/assess.html")


def read_items(metric, items_file):
    '''
    Read specifications for current metric from the items file
    Inputs:
        metric  - the string name of the metric
        items_file - the .ini file specifying actions for the metric
    Output:
        item - dictionary of specifications from the output file
    '''
    
    conf = io_utils.OAConfigParser()
    
    try:
        with open(items_file) as fid:
            conf.readfp(fid)
            items = conf.items(metric)
        return items
    except IOError as err:
        # If there is an IOError then all metrics will crash so 
        # fail now with meaningful error message
        raise run_metricsFatalError("Cannot open items file ("+items_file+") ",err)
    except ConfigParser.NoSectionError as err:
        raise run_metricsError("No action defined for metric "+metric, err)
    
    
def make_out_dir(expt):
    '''
    Check if out_dir exists and is writable.
    If not writable then raise a fatal exception
    If it doesn't exist then create it
    '''
    try:
        out_dir = expt.paths['output_dir'] + '/' + expt.runid
        if expt.cntl is not None:
            out_dir = out_dir + '_vs_' + expt.cntl.runid
        if os.path.isdir(out_dir):
            if not os.access(out_dir, os.W_OK) and os.access(out_dir, os.R_OK):
                raise run_metricsFatalError("output directory must be writable and Readable")
        else:
            os.makedirs(out_dir) 
    except KeyError:
        raise run_metricsFatalError('output (output_dir) directory must be specified in paths.ini file')
    except OSError as err:
        raise run_metricsFatalError('Could not create output directory '+out_dir, err)
    
    return out_dir
    
