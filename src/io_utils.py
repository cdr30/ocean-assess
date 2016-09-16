try:
    import configparser as ConfigParser #Python version 3 and later
except ImportError:
    import ConfigParser #Python version 2.7
import logging, logging.handlers
import os
import numpy as np
import warnings
import traceback


class OAWarning(Warning):
    pass


def warn(msg):
    '''
    warnings.warn wrapper for the OAWarning warning type. 
    '''

    warnings.warn(msg, OAWarning)


class OAConfigParser(ConfigParser.SafeConfigParser, object):
    '''
    Sub-class of ConfigParser to return items as an OAConfigDict.
    '''
    
    def items(self, *args, **kwargs):
        '''
        Return items as an OAConfigDict with path expansion.
        '''
        
        items = super(OAConfigParser, self).items(*args, **kwargs)
        items = OAConfigDict(items)
        
        # Expand filenames
        for key, value in items.iteritems():
            items[key] = expand_path(value)
                
        return items
    
    
class OAConfigDict(dict):
    '''
    Sub-class of dictionary object to implement further getter methods.
    '''
    
    def get_list(self, *args):
        '''
        Convert a list in purely string format into a conventional list. 
        '''
                
        value = super(OAConfigDict, self).get(*args)
        
        if value is not None:
            value = [i.strip().strip('[').strip(']') for i in value.split(',')]
        
        return value
    
    def get_ndarray(self, key, default = None, dtype = float):
        '''
        Convert a list in purely string format into a numpy array.
        '''
        
        value = self.get_list(key, default)
        
        if value is not None:
            value = np.array(value, dtype = dtype)
        
        return value


class Logger(object):
    '''
    Class related to logging activities.
    '''
    
    def __init__(self, expt, out_dir, logfile='ocean_assess.log'):
        '''
        Set up the logger and its handlers.
        
        '''
        
        # Clear log file
        self.logfile = out_dir + '/' + logfile
        if os.path.exists(self.logfile):
            os.remove(self.logfile)
            
        # Set up loggers and handlers
        self.loggers = {key:val for (key,val) in zip(('warn', 'main'), self._setup_loggers())}
        self.handlers = {key:val for (key,val) in zip(('warn', 'main'), self._setup_handlers())}
        
        # Set up formatting
        self._set_format()
        
        # Add handlers to loggers
        self.loggers['main'].addHandler(self.handlers['main'])
        self.loggers['warn'].addHandler(self.handlers['warn'])
        
        # Capture raised warnings
        logging.captureWarnings(True)

    def _setup_loggers(self):
        '''
        Set up the loggers. 
        '''
        
        # Set up main logger
        main_logger = logging.getLogger('main')
        main_logger.setLevel(logging.DEBUG)
        
        # Set up warning logger to perform deferred logging
        warn_logger = logging.getLogger('py.warnings')
        
        return warn_logger, main_logger
    
    def _setup_handlers(self):
        '''
        Set up the handlers.
        '''
        
        # Set up main file handler
        main_handler = logging.FileHandler(self.logfile)
        
        # Set up warning buffer handler; 
        # this flushes to the main file handler
        warn_handler = logging.handlers.MemoryHandler(10000, target=main_handler)
    
        return warn_handler, main_handler
    
    def _set_format(self):
        '''
        Set the format for logging entries.
        '''
        
        # Logger message format
        fmt = LoggerFormatter('[%(levelname)s] %(message)s \n')
        self.handlers['main'].setFormatter(fmt)
        
        # Warnings message format
        warnings.formatwarning = LoggerFormatter.formatwarning

    def write_warnings(self, metric):
        '''
        Write all warnings raised for a metric to the logging file.
        '''
        
        msg = 'Metric %s returned the following warning(s):\n'
        msg += '-' * len(msg % metric)
        
        if len(self.handlers['warn'].buffer) > 0:
            self.loggers['main'].info(msg % metric)
            self.handlers['warn'].flush()
            self.loggers['main'].info('=' * 100)

    def write_error(self, metric, err):
        '''
        Write a metric error to the logging file.
        '''
        
        msg = 'Metric %s failed with the following error message:\n'
        msg += '-' * len(msg % metric)
        
        self.loggers['main'].info(msg % metric)
        self.loggers['main'].exception(err)
        self.loggers['main'].info('=' * 100)


class LoggerFormatter(logging.Formatter):
    '''
    '''
    
    def format(self, record):
        '''
        Control the format of entries via the logging module.
        
        Remove [INFO] tags.
        '''
        
        fmt = super(LoggerFormatter, self).format(record)
        fmt = fmt.replace('[INFO] ', '')
        
        return fmt
    
    def formatException(self, exc):
        '''
        Control the format of exception tracebacks.
        
        Add indentation to the traceback.
        '''
        
        fmt = [' ' * 4 + i for i in traceback.format_exception(*exc)]
        
        return ''.join(fmt)

    @staticmethod
    def formatwarning(warn, *args):
        '''
        Control the format of entries via the warnings module.
        
        Print the warning and indented traceback.
        '''
        
        fmt = [warn.message + '\n']
        fmt += ['    Traceback (most recent call last):\n']
        fmt += [' ' * 4 + i for i in traceback.format_stack()[:-2]]
                
        return ''.join(fmt).rstrip('\n')
        
def expand_path(path):
    '''
    Expand a path string's environment variables and tilde prefix.
    '''
    
    return os.path.expanduser(os.path.expandvars(path))
