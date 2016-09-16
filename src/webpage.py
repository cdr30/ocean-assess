import markup
import HTML
import string

class Webpage(markup.page):
    """Class for assessment webpage"""
    def __init__(self,model,area=None):
        markup.page.__init__(self)

        runid = model.runid 
        title = 'Assessment of:<br>%s (%s - %s)' % \
            (runid, model.mean_start.strftime('%Y/%m/%d'), model.mean_end.strftime('%Y/%m/%d'))
        description = '%s: %s' % (runid, model.description)
        
        if model.cntl is not None:
            title += '<br>vs.<br>%s (%s - %s)' % \
                (model.cntl.runid, model.cntl.mean_start.strftime('%Y/%m/%d'), model.cntl.mean_end.strftime('%Y/%m/%d'))
            description += '<br>%s: %s' % (model.cntl.runid, model.cntl.description)

        self.init(title=title.replace('<br>', ' ').replace(':', ''))
        self.addheader('<h1 align="center">' + title +'</h1><hr>')
        self.addheader('<h2 align="center">' + description +'</h2>' )
        
        self.toc = ['<h3> Index </h3>']
        self.nitems = 0
        
        self.add('<br><center>')

    def write(self,filename):
        if len(self.toc) != 1: self.addheader(string.join(self.toc))
        fid = open( filename, 'w' )
        print >>fid, self
        fid.close()
        self.filename = filename


    def link_to_image(self, image, target, alt_text="Image missing", write = True):
        img_tag = markup.oneliner.img(src=image, alt=alt_text)
        img_tag = markup.oneliner.a(img_tag, href=target)
        if write: self.a(img_tag)
        
        return img_tag

    def table(self, tab_data, header = None, write = True, **kwargs):
        '''
        Wrapper to HTML.table to add a table to the HTML output page
        If header is 1 then take top row as header
        '''
        if header == 1:
            content = HTML.table(tab_data[1:], header_row = tab_data[0], **kwargs)
        else:
            content = HTML.table( tab_data, header_row = header , **kwargs)
        
        if write: self.add(content)
        
        return content
    
    def metric(self, title, plots, tables):
        '''
        Add a whole metric to the web page with a title (level h3),
        and a table (with no borders) containing the plot(s) and table of data
        '''
        
        if not isinstance(plots,list): plots=[plots]
        if tables is not None:
            table_periods = [table.period for table in tables]
        else: 
            table_periods = []
        
        for period,fnames in plots:
            self.contents_item(nice_period(period) + ' ' + title)
            content = [[self.link_to_image(fname, fname, write = False)] for fname in fnames]
            
            #Is there a table for this period? 
            if period in table_periods:
                metric_table = round_table_entries(tables[table_periods.index(period)].table)
                
                content.append([self.table(metric_table, 
                                           header = 1, write = False,
                                           attribs = {'align' : 'center'})]) 
        
            self.table(content, border = '0', attribs = {'align' : 'center'} )
            self.add('<br>')
            self.a('Return to top', href='#top')
            self.add('<br><br>')
        
    def contents_item(self, title): 
        '''
        Create a title with an anchor and add a link to the table of contents
        A number is added to the title (but not printed) so that all links are unique
        '''
        self.nitems = self.nitems + 1
        title_no_space = title.replace(' ','_')
        
        self.add('<a name=' + str(self.nitems) + title_no_space +'></a><h3>' + title + '</h3>')
        self.toc.append('<a href="#'+str(self.nitems) + title_no_space +'"> '+ title + '</a><br>') 
        
def nice_period(period):
    '''
    Convert likely abbreviations to nicer human readable formats
    '''
    
    out_period = {'1y':'Annual', '1s':'Seasonal', '1m':'Monthly'}.get(period,period)
    out_period = out_period.title()
    return out_period
    
    
def round_table_entries(table, prec=3):
    '''
    Round off table entries to 3 dp. Values < 0.001 use scientific notation.
    '''
    
    f_prec = '%.{}f'.format(prec)
    e_prec = '%.{}e'.format(prec)
    
    for row in xrange(len(table)):
        for col in xrange(len(table[row])):
            try:
                entry = float(table[row][col])
                
                if (f_prec % entry) != '0.000':
                    table[row][col] = f_prec % entry
                else:
                    table[row][col] = e_prec % entry
            except:
                pass

    return table
