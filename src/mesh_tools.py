import netCDF4 as nc4
import numpy as np
import utils

'''
A set of tools to contain functions for getting variables from NEMO mesh files
Placing them in here reduces the amount of duplicated code across different sections.

Created on Oct 22, 2014

@author: hadtd
'''

class mesh(object):
    '''
    Class to carry open meshmask file nc4.Dataset
    Associated functions can be used to access the meshfiles
    '''
    
    def __init__(self, meshname):
        self.ncid = nc4.Dataset(meshname)

    def get_area(self, grid, return_dims=False):
        '''
        Calculate grid cell areas for the specified grid.
        
        Arguments:
            grid-
                't','u','v','f'
                
        Keywords:
            return_dims-
                Return query_dims dictionary instead of data
                
        Returns:
            area, the 2D cell areas.
        '''
        #TODO: Error handling
        if return_dims:
            area = self.get_e12('x', grid, return_dims=True)
        else:
            area = self.get_e12('x', grid) * self.get_e12('y', grid)
        
        return area

#TODO: The user could provide an incorrect mesh mask. There should be a warning or
#      error when it does not match an existing variable mask in an apply_mask function
    def get_mask(self, grid, levels=None, return_dims=False):
        '''
        Return the mask for a specified level and grid.
         
        Arguments:
            grid-
                't','u','v','f'
        
        Keywords:
            levels-
                Level(s) for which to retrieve mask (integer, list or numpy array)
            return_dims-
                Return query_dims dictionary instead of data
                
        Returns:
            xmask, a 2D binary land mask.
        '''
        #TODO: Error handling
        
        # Read in mask for the minimum level and mask area
        var = self.ncid.variables[grid.lower() + 'mask']
         
        if return_dims:
            mask = self.query_dims(var)
        else:
            mask = self.index_by_levels(var, levels).squeeze()
            
        return mask
    
    def get_e12(self, axis, grid, return_dims=False):
        '''
        Return horizontal grid scale factor (e1x or e2x) for the specified grid.
        
        Arguments:
            axis-
                'x', 'y', corresponding to e1 and e2 scale factors
            grid-
                't','u','v','f'
            
        Keywords:
            return_dims-
                Return query_dims dictionary instead of data
                
        Returns:
            e1x or e2x, the 2D horizontal scale factor. 
        '''
        _valid_sf = {'x':'1', 'y':'2'}
        
#TODO: Need class-specific errors here too
        if axis not in _valid_sf:
            raise StandardError('"axis" must be one of ("x", "y")')
        
        sf = _valid_sf[axis]
        var = self.ncid.variables['e' + sf + grid.lower()]
        
        if return_dims:
            e12 = self.query_dims(var)
        else:
            e12 = var[:].squeeze()
        
        return e12
    
    def get_e3(self, grid, levels=None, return_dims=False):
        '''
        Return e3x (old type mesh mask) or e3x_0 (new type mesh mask).
        
        Arguments:
            grid- 
                One of 't', 'u', 'v', 'w'
                
        Keywords:
            levels-
                Level(s) for which to retrieve e3x (integer, list or numpy array)
            return_dims-
                Return query_dims dictionary instead of data
        
        Returns:
            e3x, the 3D vertical scale factor sliced by the requested vertical levels.
        
        '''
        
        var = None
        e3x_0 = 'e3%s_0' % grid.lower()
        e3x = 'e3%s' % grid.lower()
        
        for i_var in (e3x_0, e3x):
            if i_var in self.ncid.variables:
                e3x_shape = len([i for i in self.ncid.variables[i_var].shape if i > 1])
                
                if e3x_shape == 3:
                    var = self.ncid.variables[i_var]
                    break
                
        if var is None:
            raise StandardError('No 3D e3 variable found in mesh mask')
        
        if return_dims:
            e3 = self.query_dims(var)
        else:
            e3 = self.index_by_levels(var, levels).squeeze()
        
        return e3
    
    def get_latlon(self, grid, return_dims=False):
        '''
        Return gphi and glam variables.
        
        Arguments:
            grid-
                One of 't', 'u', 'v', 'f'
                
        Keywords:
            return_dims-
                Return query_dims dictionary instead of data
                
        Returns:
            A tuple containing (latitude, longitude) variables
        '''
        
        var = (self.ncid.variables['gphi' + grid.lower()],
               self.ncid.variables['glam' + grid.lower()])
        
        if return_dims:
            gphi_glam = self.query_dims(var[0])
        else:
            gphi_glam = (var[0][:].squeeze(), var[1][:].squeeze())
        
        return gphi_glam

    def get_depth(self, grid, levels=None, return_dims=False):
        '''
        Return gdepx_0 (old type mesh mask) or gdepx_1d (new type mesh mask).

        Arguments:
            grid-
                One of 't', 'w'

        Keywords:
            levels-
                Level(s) for which to retrieve gdep (integer, list or numpy array)
            return_dims-
                Return query_dims dictionary instead of data

        Returns:
            gdep, the 1D depth coordinate sliced by the requested vertical levels.
        '''
        
        var = None
        gdepx_1d = 'gdep%s_1d' % grid.lower()
        gdepx_0 = 'gdep%s_0' % grid.lower()

        for i_var in (gdepx_1d, gdepx_0):
            if i_var in self.ncid.variables:
                gdepx_shape = len([i for i in self.ncid.variables[i_var].shape if i > 1])
                
                if gdepx_shape == 1:
                    var = self.ncid.variables[i_var]
                    break
                
        if var is None:
            raise StandardError('No 1D depth variable found in mesh mask')
        
        if return_dims:
            gdep = self.query_dims(var)
        else:
            gdep = self.index_by_levels(var, levels).squeeze()

        return gdep
        
    def index_by_levels(self, var, levels):
        '''
        Return input variable sliced by requested vertical levels,
        otherwise return the input variable unsliced. 
        '''
        
        zdim = np.where(np.array(var.dimensions) == 'z')[0]
        index = [slice(None)] * len(var.dimensions)
        #What variable type is levels
        if len(zdim) != 0:
            if isinstance(levels, int):
                index[zdim] = np.array( [levels] )
            elif isinstance(levels, list) and len(levels) != 0:
                index[zdim] = np.array( levels )
            elif isinstance(levels, np.ndarray) and levels.size != 0:
                index[zdim] = levels
        
        return var[index]

    def query_dims(self, var):
        '''
        Return a dictionary of dimension indices for a mesh mask variable.
        '''
        
        return utils.query_dims(var, ignore_single=True)
