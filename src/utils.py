'''
Created on Nov 8, 2013

@author: hadtd
'''

from netCDF4 import date2num 
from netCDF4 import Variable as nc4_variable 
import numpy as np
import collections
import biggus

def sort_dates(dates, calendar):
    '''
    Utility to sort a list of netcdftime.datetime objects into date order
    For some reason a simple sort doesn't work so convert them to numbers first and then use argsort()
    '''
    
    nums = np.array([date2num(date, 'seconds since 0000-00-00 00:00:00', calendar) for date in dates])
    index = nums.argsort()
    
    return [dates[i] for i in index]
   

def wrap_longitude(longitudes,base=0.0):
    '''
    Convert points in longitude to range from base (default zero) to base+360 
    ''' 
    if isinstance(longitudes,list): 
        longitudes=np.array(longitudes)
        intype='list'
    elif isinstance(longitudes,tuple): 
        longitudes=np.array(longitudes)
        intype='tuple'
    else: intype=None
    wrapped_longs=((longitudes-base+720.) % 360.)+base
    
    if intype=='list': wrapped_longs=wrapped_longs.tolist()
    if intype=='tuple': wrapped_longs=tuple(wrapped_longs.tolist())
    
    return wrapped_longs


def mask_gen(lon,lat,region):
    '''
    Generate a numpy array consisting of 1 within region and 0 outside of the region
    Inputs arguments:
    lon = 1D or 2D numpy array of longitude
    lat = 1D or 2D numpy array of latitude
    region = tuple or list of format [North, South, West, East] specifying the limits of the region
    '''
    
    #Check input types
    if not isinstance(lat,np.ndarray) or (lat.ndim not in [1,2]):
        raise RuntimeError("lat must be a 1 dimensional or 2dimensional numpy array")
    if not isinstance(lon,np.ndarray) or (lon.ndim not in [1,2]):
        raise RuntimeError("lon must be a 1 dimensional or 2 dimensional numpy array")
    if not isinstance(region,(tuple,list)) or (len(region) != 4):
        raise RuntimeError("Region must be 4 element tuple or list")
    if lon.shape != lat.shape:
        raise RuntimeError("lat and lon must have same shape")
    
    #Wrap longitude to be in range region[2]:region[2]+360
    lon=wrap_longitude(lon,base=region[2])
    
    #Search for lat and lon points within range and set to 1
    latmask=np.where(lat <= region[0],1,0)*np.where(lat >= region[1],1,0)
    lonmask=np.where(lon <= region[3],1,0)*np.where(lon >= region[2],1,0)
    
    #Combine latmask and lonmask
    if lon.ndim == 1:
        #TODO: ???
        pass
    else:
        mask=np.where(latmask*lonmask == 1, 1, 0)

    return mask


def mask_and_crop(field, index, crop=True, force_slices=False):
    '''
    Create a masked copy of a field, with unmasked points determined by the given indices.
    The number of field dimensions is unaltered, regardless of the index types provided. 
    
    Method:
        The full field is first sliced by the maximum extent of the given indices (for disk
        access efficiency). The mask of this sliced field is then set to False for the given
        indices and True everywhere else.
        This cropped field is then placed back into a masked full global field if required.
    
    Arguments:
        field- 
            numpy.ndarray type or netCDF4.Variable object.
        index- 
            Sequence of indices into field to return as valid data; 
            other points will be masked. Must be given as a sequence of indices
            into each dimension of field.
            May each be one of (int, list, tuple, numpy.ndarray).
        
    Keywords:
        crop- 
            If True, then crop the field to the extent of the indices.
            If False, instead mask the field to the extent of the indices.
        force_slices -
            If True, then the indices are treated as slices between the minimum 
            and maximum of each index array. 
            If False, then the indices are treated normally.
            
    Returns:
        field_out-
            A copy of field, masked except where specified by index.
        index_out-
            A copy of index, valid for the cropped domain. 
    '''
    
    # Input control
    if type(field) not in (nc4_variable, np.ndarray, np.ma.core.MaskedArray):
        raise TypeError('Input data type not supported: %s' % type(field))
    if len(index) != field.ndim:
        err = 'Indices must be a {}-d array, matching the number of dimensions of field'
        raise ValueError(err.format(len(index)))
    
    # Index treatment. A numpy array version of index is first calculated,
    # then the index objects are redefined relative to the sliced domain.
    index_out = index[:]
    index_np = []
    slices = []
    for i in xrange(len(index_out)):
        
        # List / tuple
        if type(index_out[i]) in (list, tuple):
            index_np += [np.array(index_out[i])]
            if type(index_out[i]) is list: 
                index_out[i] = [j - min(index_out[i]) for j in index_out[i]]
            else:
                index_out[i] = tuple([j - min(index_out[i]) for j in index_out[i]])
                
        # Integers become a list so the dimension isn't collapsed
        elif type(index_out[i]) is int:
            index_np += [np.array( [index_out[i]] )]
            index_out[i] = [0]
        
        # Slice limits are converted to explicit integers
        elif type(index_out[i]) is slice:
            start = [0 if index_out[i].start is None else index_out[i].start][0]
            stop = [field.shape[i] if index_out[i].stop is None else index_out[i].stop][0]
            index_np += [np.arange(start, stop)]
            index_out[i] = slice(0, stop - start) 
        
        # Numpy array
        elif type(index_out[i]) is np.ndarray:
            index_np += [index_out[i]]
            index_out[i] = index_out[i] - index_out[i].min()

        else:
            raise TypeError('Index type not supported: %s' % type(index_out[i]))
        
        # Infer domain slices
        slices += [slice(index_np[i].min(), index_np[i].max() + 1)]

    # Save full field dimensions 
    if not crop: 
        global_shape = field.shape

    # First access the data orthogonally (more efficient than directly indexing)
    field = field[slices]
    if type(field) is np.ndarray:
        field = np.ma.array(field)
    
#TODO: This copy statement needs to go
    # We still need field.mask
    field_out = field.copy()

    # Do indexing unless we want indices to be interpreted as slices
    if not force_slices:
    
        # Start with a masked copy of the field then unmask for the given indices
        # (subject to numpy broadcasting failures)
        field_out.mask = True
        field_out.mask[index_out] = False
        
        # Restore the land mask
        field_out.mask = field_out.mask | field.mask

    # If not cropping the output, restore the cropped points as masked data
    if not crop:
        field = np.ma.masked_all(global_shape)
        field[slices] = field_out
        field_out = field
        index_out = index[:]
    
    return field_out, index_out

def replicate_ndarray(var, rep_shape, axis=0):
    '''
    '''

    if type(var) not in (nc4_variable, np.ndarray):
        raise TypeError('Type %s is not supported' % type(var))

    if not hasattr(rep_shape, '__iter__'):
        rep_shape = [rep_shape]
        
    stack = var
    
    for ind in xrange(len(rep_shape)):
        n_rep = rep_shape[::-1][ind]
        
        stack = biggus.ArrayStack([biggus.OrthoArrayAdapter(stack)] * n_rep)
        try:
            if axis < 0:
                raise ValueError
            
            stack = np.rollaxis(stack, 0, axis + 1)
        except ValueError:
            err = 'Axis value must be a position within or ' + \
            'adjacent to the input variable shape (0 <= axis <= %s)'
            raise ValueError(err % var.ndim)
    
    return stack.ndarray()

def query_dims(var, ignore_single=False):
    '''
    Parse netCDF4.Variable dimension names and return a dictionary of indices.
    
    '''

    dims = {}
    var_dims = np.array(var.dimensions)
    
    _dim_checks = {'t': lambda s: s.lower().startswith('time') or s == 't',
                   'z': lambda s: s.lower().startswith('depth') or s == 'z',
                   'y': lambda s: s.lower().startswith('latitude') or s == 'y',
                   'x': lambda s: s.lower().startswith('longitude') or s == 'x'
                   }
    
    # Ignore single dimensions as if the output were squeezed
    if ignore_single:
        ignore_dims = np.where(np.array(var.shape) > 1)
        var_dims = var_dims[ignore_dims]

    # Check if the dimension name meets a criterion in _dim_checks
    for i, dim in enumerate(var_dims):
        for dim_id, check in _dim_checks.items():
            if check(dim):
                dims[dim_id] = i
                break
            
    return dims

def plot_tuple():
    '''
    Create a named tuple to describe plots
    '''
    return collections.namedtuple('plotlist', ['period', 'fnames'])

def table_tuple():
    '''
    Create a named tuple to describe plots
    '''
    return collections.namedtuple('tablelist', ['period', 'table'])

def list_unique(seq):
    '''
    Return a list of the unique elements in list input whilst preserving order
    '''
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]
