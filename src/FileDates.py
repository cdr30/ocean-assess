'''
Created on Nov 21, 2013

@author: hadtd
'''

from netCDF4 import Dataset, num2date, date2num 
import numpy as np
import os
import datetime

class FileDatesError(Exception):
    pass

class FileDate():
    '''
    A class to act as a reference linking dates to file names
    This allows data corresponding to a particular date to be extracted easily
    '''

    def __init__(self, date, calendar, fname, record):
        '''
        Set up FileDate instance
        Inputs:
            date - NetCDF datetime object
            calendar - String of the calendar type associated with data
            fname - Filename containing the data
            record - record index corresponding to this time within the file
        '''
        self.date = date
        self.calendar = calendar
        self.filename = fname
        self.record = record
        
    def __str__(self):
        return ", ".join([str(self.date), self.filename, self.calendar, 
                          str(self.record)])

    def __repr__(self):
        return ", ".join([str(self.date), self.filename, self.calendar,
                          str(self.record)])


def get_file_date(fname):
    '''
    Return a FileDateList of FileDate objects corresponding to all dates in a netcdf file 
    '''
    
    if not os.path.isfile(fname):
        raise RuntimeError('File "%s" does not exist' % fname)

    # Identify the time coordinate
    time_name = get_time_var(fname)

    with Dataset(fname) as ncid:
        time = ncid.variables[time_name]

        # Return an empty list if the file is empty
        if time.size != 0:
            dates = num2date(time[:], time.units, time.calendar)
            return FileDatesList([FileDate(date, time.calendar, fname, i) 
                                  for i, date in enumerate(dates)])
        else:
            return [] 


def get_time_var(fname):
    '''
    Find the time coordinate variable according to CF metadata rules.
    
    A time coordinate **must** have a units attribute with a Udunits value of
    '<units> since <reference_time>'.
    
    Where dimension and auxiliary coordinates are found, the former is preferred.
    '''

    # Attribute check
    _attr_check = lambda var: 'units' in var.ncattrs() and 'since' in var.getncattr('units').strip().split(' ')

    with Dataset(fname) as ncid:
            
        # Time coordinates with the 'units' attribute
        time_var = [name for name, var in ncid.variables.iteritems() if _attr_check(var)] 

        # Use a dimension coordinate variable if possible
        time_dim_coords = [name for name in time_var if name in ncid.variables[name].dimensions]
        if len(time_dim_coords) == 1:
            time_var = time_dim_coords
        
    # Require only one time coordinate 
    if len(time_var) != 1:
        err = 'Found {n_var} time coordinates (require 1) in {file}: {vars!r}'
        raise FileDatesError(err.format(n_var=len(time_var), file=fname, vars=time_var))

    return time_var[0]

def File_Dates(fnames):
    '''
    Return a FileDateList of FileDate objects corresponding to all dates within a list of files 
    '''
    outlist = FileDatesList()
    if isinstance(fnames, (str, unicode)): 
        fnames = [fnames] 
    for fname in fnames:
        outlist.extend(get_file_date(fname))
    if len(outlist) != 0:
        outlist.sort()
    return outlist
    

class FileDatesList(list):
    '''
    All functionality of a list but with extra functions related to FileDate class
    '''

    def __new__(cls, list_of_file_dates=None):
        '''
        Given a class: `list` of file_date objects create a FileDatesList
        '''
        file_date_list = list.__new__(cls, list_of_file_dates)
        
        #Need to check that all items in list are file_dates
        if not all( [isinstance(file_date, FileDate) for file_date in file_date_list] ):
            raise ValueError('All items in list_of_file_dates must be FileDate objects')
        return file_date_list
    
    def __add__(self, other):
        return FileDatesList(list.__add__(self, other))
    
    def __getitem__(self, keys):
        """x.__getitem__(y) <==> x[y]"""
        result = super(FileDatesList, self).__getitem__(keys)
        if isinstance(result, list):
            result = FileDatesList(result)
        return result
    
    def __getslice__(self, start, stop):
        """
        x.__getslice__(i, j) <==> x[i:j]
        """
        result = super(FileDatesList, self).__getslice__(start, stop)
        result = FileDatesList(result)
        return result
    
    def __repr__(self):
        ''' Return repr string for each of the FileDates in list'''
        return "\n".join([repr(filedate) for filedate in self])
    
    def __str__(self):
        ''' Return str string for each of the FileDates in list'''
        return "\n".join([str(filedate) for filedate in self])

    def datetimeref(self):
        ''' Return reference time string for date2num conversions '''
        
        dt = self.dates()[0]
        return 'seconds since %04i-01-01 00:00:00' % dt.year
      

    def sort(self):
        '''
        Sort FileDateList into date order
        '''
  

        calendar = list(set(self.calendars()))
        if len(calendar) != 1:
            raise RuntimeError("In sort_file_dates: all file_date tuples must have same calendar")

        nums = np.array([date2num(date, self.datetimeref(), calendar[0]) for date in self.dates()])
        index = nums.argsort()
    
        return FileDatesList([self[i] for i in index])
    
    def in_range(self, start, end, month = None, season = None): 
        '''
        Return the subset of file_dates that fall between start and end
        Inputs:
            start/end - netcdftime objects giving the start and end date
        '''
        
        calendars = list(set(self.calendars()))
        if len(calendars) != 1:
            raise RuntimeError("In file_dates_in_range: all file_date tuples must have same calendar")
            
    
        
        start_dt = datetime.datetime(start.year, start.month, start.day)
        end_dt = datetime.datetime(end.year, end.month, end.day)

        startnum = date2num(start_dt, self.datetimeref(), self[0].calendar)
        endnum = date2num(end_dt, self.datetimeref(), self[0].calendar)
        dates_num = np.array([date2num(date, self.datetimeref(), self[0].calendar) for date in self.dates()])
        
        ind = np.where( (dates_num >= startnum) & (dates_num <= endnum) )

        if len(ind[0]) != 0:
            new_list = FileDatesList([self[i] for i in ind[0]])
        else:
            raise RuntimeError("No dates found in range") 

        if season is not None:
            new_list = new_list.in_season(season)
        if month is not None:
            new_list = new_list.in_month(month)

        return new_list
    
    def in_season(self, season):
        '''
        Reduce FileDatesList to contain data within specified season only
        '''

        try:
            months_to_select = {'djf' : [12, 1, 2],
                                'mam' : [ 3, 4, 5],
                                'jja' : [ 6, 7, 8],
                                'son' : [ 9,10,11] }[season.lower()]
        except IndexError:
            raise FileDatesError('Unrecognized season '+season)
        months = [date.month for date in self.dates()]
        new_list = []
        #Is this safe - could order get changed here?
        for i, month in enumerate(months):
            if month in months_to_select:
                new_list.append(self[i])
            
        if new_list:
            return FileDatesList(new_list)
        else:
            raise RuntimeError("No dates found in season "+season) 
        
    def in_month(self, month):
        '''
        Reduce FileDatesList to contain data within specified month only
        Inputs:
	       month = either an integer/float corresponding to a month (1 = January)
	           or a string of the 1st 3 characters of the month name
        Returns:
	       A new FileDatesList only containing FileDates for the requested month
	    '''
        if not isinstance(month, (str, unicode, float, int)):
            raise FileDatesError('Month must be a string, float or integer')

        if isinstance(month, (str, unicode)):
            try:
                month_to_select = {'jan' : 1,
	                               'feb' : 2,
       			                   'mar' : 3,
                                   'apr' : 4,
			                       'may' : 5,
			                       'jun' : 6,
			                       'jul' : 7,
			                       'aug' : 8,
			                       'sep' : 9,
			                       'oct' : 10,
			                       'nov' : 11,
			                       'dec' : 12}[month.lower()]
            except IndexError:
                raise FileDatesError('Unrecognized Month abbreviation '+month)
        else:
            if month not in np.arange(12)+1:
                raise FileDatesError('If month is a number it must be a whole number in between 1 and 12')
            else: month_to_select = month

        months = np.array([date.month for date in self.dates()])
        ind = np.where(months == month_to_select)
 
        if ind[0].size:
            return FileDatesList([self[i] for i in ind[0]])
        else:
            raise RuntimeError("No dates found in range for month '%s'" % month.title()) 

    
    def dates(self):
        '''
        Return a list of all dates
        '''
        return [file_date.date for file_date in self]
    
    def filenames(self):
        '''
        Return a list of all filenames
        '''
        return [file_date.filename for file_date in self]

    def calendars(self):
        '''
        Return a list of all calendars
        '''
        return [file_date.calendar for file_date in self]
    
    def records(self):
        '''
        Return a list of all records
        '''
        return [file_date.record for file_date in self]
