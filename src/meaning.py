'''
Module to deal with creation of mean files to avoid duplication of code in other modules
Created on Aug 20, 2014

@author: hadtd
'''

import netCDF4 as nc4
import numpy as np
import biggus
import os
import stat

class MeaningError(Exception):
    pass

#NOTE: This (like the subdomain mask) can be run for multiple notes for the same mean file, 
#      which will cause failures due the file existing but containing no data. There is 
#      probably a clever way of checking for this and waiting for the file to finish being made,
#      but I suspect for now we just have to ensure separate directories are specified, even though
#      this is duplication of mean creation. 
def make_mean(meanfile, fdates, variable, mesh, grid):
    '''
    Make an average of variable and save in meanfile.
    '''

    nc_out = nc4.Dataset(meanfile, 'w')
    
    #Open first file to get dimensions and variable info
    nc_dummy = nc4.Dataset(fdates[0].filename)
    
    # Copy dimensions to new file
    copy_dims(nc_dummy, nc_out)

    # Check that variable is in 1st input file and copy it (and its attributes)
    try:
        varin = nc_dummy.variables[variable]
    except KeyError:
        raise MeaningError("Variable " + variable +" not found in file.")

    # Use mask value from source file (to save converting the data)
    if '_FillValue' in varin.ncattrs():
        mdi = varin._FillValue
    elif 'missing_value' in varin.ncattrs():
        mdi = varin.missing_value
    else:
        mdi = 9.96921e+36

    varout = nc_out.createVariable(variable, varin.dtype, varin.dimensions, fill_value=mdi)
    copy_atts(varin, varout)
    
    shape = list(varin.shape)
    #TODO:Assumes that record dimension is first!
    shape[0] = 1
    
    #TODO:What should we do with coordinate/auxiliary variables?
    #Leave them for now and work out later

    nc_dummy.close() #Close this as it will be reopened in loop later
    
    input_data = []
    for fdate in fdates:
        #TODO: Should I be more careful about axes here - assumption that record dimension is first
        input_data.append(biggus.OrthoArrayAdapter(  \
                          nc4.Dataset(fdate.filename). \
                          variables[variable])[fdate.record:fdate.record+1])
    #input_data = np.array(input_data)
    array = biggus.ArrayStack(input_data)
    the_mean = biggus.mean(array, 0)
    
    #Save the data to disk
    biggus.save([the_mean], [varout])
    nc_out.close()
    
    #Give group and others write access to the mean file so that they could run the code next time.
    st = os.stat(meanfile)
    os.chmod(meanfile, st.st_mode | stat.S_IWOTH)
    
    nc_out = nc4.Dataset(meanfile,'r+')
    varout = nc_out.variables[variable]

#Apply mask to data for cases where land is set to zero
#TODO: Implement biggus in this bit
    index_var = [slice(None)]*varout.ndim
#TODO: Assumption about record dimensiion here
    index_var[0] = np.array([0])
    #How many levels are in variable?
    zdim_var=None
    #Does the variable use the depth dimension?
    for i,dim in enumerate(varout.dimensions):
        if dim[0:5] == 'depth' or dim[0]=='z': zdim_var=i
    if zdim_var is None: 
        nlev = 1
    else: nlev = shape[zdim_var]

#TODO: Try to replace this bit with biggus or something that is fast but uses less memory (for ORCA12)
#TODO: Assumption about dimension order below
    for k in range(nlev):
        masklev = mesh.get_mask(grid.lower(), k)

        try:
            if nlev == 1:
                varout[:,:,:] = np.ma.masked_array(varout[:,:,:], mask = (masklev == 0))
            else:
                varout[:,k,:,:] = np.ma.masked_array(varout[:,k,:,:], mask = (masklev == 0))
                
#TODO: This is another thing that should be in a high level I/O function
        except np.ma.core.MaskError:
            err = 'Mesh mask with shape %s does not conform to source data with shape %s'
            raise MeaningError(err % (masklev.shape, varout.shape))
    nc_out.close()

def copy_dims(ncid_in, ncid_out):
    '''
    Copy alls dimensions from ncid_in to ncid_out
    Both ncid_in and ncid_out are netCDF4.Dataset classes
    '''
    for dimname,dim in ncid_in.dimensions.iteritems():
        if dim.isunlimited():
            ncid_out.createDimension(dimname)
        else:
            ncid_out.createDimension(dimname, len(dim))

def copy_atts(varin, varout):
    '''
    Copy attributes from one netCDF variable to another
    '''
    for att in varin.ncattrs():
        if att != '_FillValue': 
            varout.setncattr(att, varin.getncattr(att) )

def check_file(meanfile):
    '''
    Utility to check whether a file exists and whether it is readable
    '''
#TODO: Need more here: it is possible the file could be mid-write and have partially-valid data (zeros and non-zeros)
#      How to check that a file is already open for writing?
#TODO: This is a general file I/O task, nor specific to meaning files. Should write as such outside this module.
    if os.path.isfile(meanfile):
        try:
            ncid = nc4.Dataset(meanfile)
            for dim in ncid.dimensions.values():
                if dim.isunlimited() and len(dim)==0:
                    ncid.close()
                    raise RuntimeError('Length of unlimited dimension is zero. Incomplete NetCDF file?')
            ncid.close()
            return True
        except RuntimeError:
            #There was a problem reading the file so try to delete it
            if os.access(meanfile, os.W_OK):
                os.remove(meanfile)
                return False
            else:
                raise MeaningError('Meanfile ' + meanfile + ' exists \n ' +
                                   'but there is a problem reading'+ 
    		                      'it and it cannot be deleted/remade. '+
                                  'HINT: Try deleting it manually')
    else: return False



if __name__ == '__main__':
    pass
