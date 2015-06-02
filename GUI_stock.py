from datetime import datetime
from urllib.request import urlopen
import  tkinter as tk
import  tkinter.ttk as ttk

import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib import style
mpl.style.use('ggplot')

import numpy as np
import pandas as pd
import pandas.io.data as web
from sklearn.decomposition import KernelPCA

class YStock:
    """
    Credit: professor Massimo DePierro.
    Class that downloads and stores data from Yahoo Finance
    Examples:
    >>> google = YStock('GOOG')
    >>> current = google.current()
    >>> price = current['price']
    >>> market_cap = current['market_cap']
    >>> h = google.historical()
    >>> last_adjusted_close = h[-1]['adjusted_close']
    >>> last_log_return = h[-1]['log_return']
    """
    URL_CURRENT = 'http://finance.yahoo.com/d/quotes.csv?s=%(symbol)s&f=%(columns)s'
    URL_HISTORICAL = 'http://ichart.yahoo.com/table.csv?s=%(s)s&a=%(a)s&b=%(b)s&c=%(c)s&d=%(d)s&e=%(e)s&f=%(f)s'
    def __init__(self,symbol):
        self.symbol = symbol.upper()

    def current(self):
        import urllib
        FIELDS = (('price', 'l1'),
                  ('change', 'c1'),
                  ('volume', 'v'),
                  ('average_daily_volume', 'a2'),
                  ('stock_exchange', 'x'),
                  ('market_cap', 'j1'),
                  ('book_value', 'b4'),
                  ('ebitda', 'j4'),
                  ('dividend_per_share', 'd'),
                  ('dividend_yield', 'y'),
                  ('earnings_per_share', 'e'),
                  ('52_week_high', 'k'),
                  ('52_week_low', 'j'),
                  ('50_days_moving_average', 'm3'),
                  ('200_days_moving_average', 'm4'),
                  ('price_earnings_ratio', 'r'),
                  ('price_earnings_growth_ratio', 'r5'),
                  ('price_sales_ratio', 'p5'),
                  ('price_book_ratio', 'p6'),
                  ('short_ratio', 's7'))
        columns = ''.join([row[1] for row in FIELDS])
        url = self.URL_CURRENT % dict(symbol=self.symbol, columns=columns)
        raw_data = urllib.urlopen(url).read().strip().strip('"').split(',')
        current = dict()
        for i,row in enumerate(FIELDS):
            try:
                current[row[0]] = float(raw_data[i])
            except:
                current[row[0]] = raw_data[i]
        return current

    def historical(self,start=None, stop=None):
        import datetime, time, urllib, math,io
        start =  start or datetime.date(1900,1,1)
        stop = stop or datetime.date.today()
        url = self.URL_HISTORICAL % dict(
            s=self.symbol,
            a=start.month-1,b=start.day,c=start.year,
            d=stop.month-1,e=stop.day,f=stop.year)
        # Date,Open,High,Low,Close,Volume,Adj Close
        data = urlopen(url)
        f = io.TextIOWrapper(data,encoding='utf-8')
        lines = f.readlines()
        raw_data = [row.split(',') for row in lines[1:] if row.count(',')==6]
        previous_adjusted_close = 0
        series = []
        raw_data.reverse()
        for row in raw_data:
            open, high, low = float(row[1]), float(row[2]), float(row[3])
            close, vol = float(row[4]), float(row[5])
            adjusted_close = float(row[6])
            adjustment = adjusted_close/close
            if previous_adjusted_close:
                arithmetic_return = adjusted_close/previous_adjusted_close-1.0

                log_return = math.log(adjusted_close/previous_adjusted_close)
            else:
                arithmetic_return = log_return = None
            previous_adjusted_close = adjusted_close
            series.append(dict(
               date = datetime.datetime.strptime(row[0],'%Y-%m-%d'),
               open = open,
               high = high,
               low = low,
               close = close,
               volume = vol,
               adjusted_close = adjusted_close,
               adjusted_open = open*adjustment,
               adjusted_high = high*adjustment,
               adjusted_low = low*adjustment,
               adjusted_vol = vol/adjustment,
               arithmetic_return = arithmetic_return,
               log_return = log_return))
        return series

    @staticmethod
    def download(symbol='goog',what='adjusted_close',start=None,stop=None):
        return [d[what] for d in YStock(symbol).historical(start,stop)]

import os
import uuid
import sqlite3
import pickle

class PersistentDictionary(object):
    """ 
    credit: professor Massimo DiPierro
    A sqlite based key,value storage.
    The value can be any pickleable object.
    Similar interface to Python dict
    Supports the GLOB syntax in methods keys(),items(), __delitem__()

    Usage Example:
    >>> p = PersistentDictionary(path='test.sqlite')
    >>> key = 'test/' + p.uuid()
    >>> p[key] = {'a': 1, 'b': 2}
    >>> print p[key]
    {'a': 1, 'b': 2}
    >>> print len(p.keys('test/*'))
    1
    >>> del p[key]
    """

    CREATE_TABLE = "CREATE TABLE persistence (pkey, pvalue)"
    SELECT_KEYS = "SELECT pkey FROM persistence WHERE pkey GLOB ?"
    SELECT_VALUE = "SELECT pvalue FROM persistence WHERE pkey GLOB ?"
    INSERT_KEY_VALUE = "INSERT INTO persistence(pkey, pvalue) VALUES (?,?)"
    DELETE_KEY_VALUE = "DELETE FROM persistence WHERE pkey LIKE ?"
    SELECT_KEY_VALUE = "SELECT pkey,pvalue FROM persistence WHERE pkey GLOB ?"

    def __init__(self,
                 path='persistence.sqlite',
                 autocommit=True):
        self.path = path
        self.autocommit = autocommit
        create_table = not os.path.exists(path)
        self.connection  = sqlite3.connect(path)
        self.connection.text_factory = str # do not use unicode
        self.cursor = self.connection.cursor()
        if create_table:
            self.cursor.execute(self.CREATE_TABLE)
            self.connection.commit()

    def uuid(self):
        return str(uuid.uuid4())

    def keys(self,pattern='*'):
        "returns a list of keys filtered by a pattern, * is the wildcard"
        self.cursor.execute(self.SELECT_KEYS,(pattern,))
        return [row[0] for row in self.cursor.fetchall()]

    def __contains__(self,key):
        return True if self[key] else False

    def __iter__(self):
        for key in self:
            yield key

    def __setitem__(self,key,value):
        if value is None:
            del self[key]
            return
        self.cursor.execute(self.INSERT_KEY_VALUE,
                            (key, pickle.dumps(value)))
        if self.autocommit: self.connection.commit()

    def __getitem__(self,key):
        self.cursor.execute(self.SELECT_VALUE, (key,))
        row = self.cursor.fetchone()
        return pickle.loads(row[0]) if row else None

    def __delitem__(self,pattern):
        self.cursor.execute(self.DELETE_KEY_VALUE, (pattern,))
        if self.autocommit: self.connection.commit()

    def items(self,pattern='*'):
        self.cursor.execute(self.SELECT_KEY_VALUE, (pattern,))
        return [(row[0],pickle.loads(row[1])) \
                    for row in self.cursor.fetchall()]


class WelcomePage():
	def __init__(self, master):
		self.master = master
		self.master.geometry("500x500+100+100")
		self.master.title("ATIVO_CAPITAL")
		self.label = ttk.Label(self.master,text="WELCOME TO ATIVO CAPITAL PORTFOLIO ANALYTICS PLATFORM")
		self.label.grid(row = 0, column = 0, padx = 50, pady = 50)
	
		self.buttonFrame = ttk.Frame(self.master)
		self.buttonFrame.grid(row = 4, column = 0, columnspan = 2)
	
		self.button1 = ttk.Button(self.buttonFrame, text="Construct your porfolio",command = self.goConstructPage)
		self.button1.grid(row = 0, column = 0)
			
		self.button2 = ttk.Button(self.buttonFrame, text="ENTER PLATFORM",command = self.goPlatform)
		self.button2.grid(row = 1, column =0)
			
		self.button3 = ttk.Button(self.buttonFrame,text="QUIT",command = self.finish)
		self.button3.grid(row = 4, column = 0)
		ttk.Style().configure("TButton", padding=6, relief="groove",background="#dadae3",padx = 2, pady = 2, height = 5, width = 20,foreground='#000000',highlightthickness='20',font=('Helvetica', 12, 'bold') )		
	def goPlatform(self):
		root2 = tk.Toplevel(self.master)
		root2.geometry("1280x800")
		root2.title('ANALYTICS PLATFORM')
		root2.protocol('WM_DELETE_WINDOW',self.master.quit)
		myPlatform = noteBook(root2,self.master)
	
	def goConstructPage(self):
		root3 = tk.Toplevel(self.master)
		myEntry = ttk.Frame(root3)	

	def finish(self):
		self.master.destroy()

class noteBook():
	def __init__(self,master,parent):
		parent.withdraw() # hide the root window
	#	to make oarent window visible
	# 	parent.deiconify()	
		self.master = master
		self.master.title("ANALYTICS PLATFORM")

		main = ttk.Frame(master, name="master")
		main.pack(fill = tk.BOTH)
		root.protocol('WM_DELETE_WINDOW',main.quit) # force the entire slave frame to close

		n = ttk.Notebook(main, name='notebook')
		n.pack(fill = tk.BOTH, padx = 2, pady = 2)

		f1 = Linear_PCA_vs_SPY(n)
		f2 = PCA_graph(n)
		f3 = WindowGraph(n)

		n.add(f1,text="Linear Regression")
		n.add(f2,text="PCA_One component")
		n.add(f3,text="Graph")
		
#		if not HAVE_DB:
#			demo()

class  WindowGraph(ttk.Frame):
	def __init__(self,master):
		super(WindowGraph,self).__init__(master)

		fig = Figure(figsize=(10,10), dpi = 100)
		ax= fig.add_subplot(1,1,1, axisbg='#cccccc')
		
		x = [i for i in range(0,10)]
		y = [i*i for i in range(-5,5)]
		ax.plot(x,y,'c',linewidth = 2)

		canvas = FigureCanvasTkAgg(fig, self)
		canvas.show()
		canvas.get_tk_widget().pack(side = tk.TOP,fill= tk.BOTH,expand = True)

		toolbar = NavigationToolbar2TkAgg(canvas, self)
		toolbar.update()
		canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)


class  PCA_graph(ttk.Frame):
	def __init__(self,master):
		super(PCA_graph,self).__init__(master)

		fig = Figure(figsize=(10,10), dpi = 100)
		ax= fig.add_subplot(1,1,1, axisbg='#cccccc')
		demo_port = ['SPY','BA', 'WFC', 'PEP', 'AMGN', 'BAX', 'BK', 'FB', 'COST', 'DIS', 'LOW', 'FDX',  'TWX', 'AIG', 'MSFT', 'IBM', 'SBUX', 'FCX', 'PG', 'BMY', 'MDT', 'SPG', 'VZ', 'OXY', 'CL', 'GILD', 'CVS', 'AMZN', 'GE', 'ABT', 'JNJ', 'UTX', 'WMT', 'ALL', 'PFE', 'FOXA', 'MO', 'MCD', 'MMM', 'SO', 'MON', 'APC', 'NOV', 'APA', 'CMCSA', 'DVN', 'ACN', 'CAT', 'EXC', 'TXN', 'UNP', 'HPQ', 'V', 'LMT', 'RTN', 'CSCO', 'DOW', 'LLY', 'NSC', 'JPM', 'C', 'HAL', 'INTC', 'ABBV', 'UNH', 'MA', 'GM', 'XOM', 'KO', 'EBAY', 'MET', 'GS', 'CVX', 'HON', 'MRK', 'AXP', 'USB', 'EMC', 'DD', 'HD', 'AAPL', 'PM', 'F', 'T', 'UPS', 'SLB', 'AEP', 'EMR', 'COF', 'MDLZ', 'GOOG', 'NKE', 'COP', 'QCOM', 'TGT', 'ORCL', 'GD', 'MS', 'BAC']
	
	
		data = pd.DataFrame()
		for symbol in demo_port:
			data[symbol] = web.DataReader(symbol, data_source='yahoo')['Close']
		data = data.dropna()
	
		spy = pd.DataFrame(data.pop('SPY'))	
	
		#normalize data
		scale_func = lambda x: ( x-x.mean())/x.std()
		#apply PCA
		pca = KernelPCA().fit(data.apply(scale_func))

		get_we = lambda x: x/x.sum()
		#print (get_we(pca.lambdas_)[:20])

		pca_one = KernelPCA(n_components = 1).fit(data.apply(scale_func))
		spy['PCA_1'] = pca_one.transform(data)

		# Plotting
		ax= fig.add_subplot(1,1,1, axisbg='#cccccc')
		ax.plot(spy.apply(scale_func))
		spy_patch = mpatches.Patch(color='red', label = 'SPY 500')
		PCA_1_patch = mpatches.Patch(color ='blue', label = 'Portfolio PCA 1 component')
		ax.legend(handles =[spy_patch, PCA_1_patch], loc = 2)
		
			
		canvas = FigureCanvasTkAgg(fig, self)
		canvas.show()
		canvas.get_tk_widget().pack(side = tk.TOP,fill= tk.BOTH,expand = True)

		toolbar = NavigationToolbar2TkAgg(canvas, self)
		toolbar.update()
		canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)
		#plt.show()


class  Linear_PCA_vs_SPY(ttk.Frame):
	def __init__(self,master):
		super (Linear_PCA_vs_SPY ,self).__init__(master)

		fig = Figure(figsize=(10,10), dpi = 100)
		ax= fig.add_subplot(1,1,1, axisbg='#cccccc')
		demo_port = ['SPY','BA', 'WFC', 'PEP', 'AMGN', 'BAX', 'BK', 'FB', 'COST', 'DIS', 'LOW', 'FDX',  'TWX', 'AIG', 'MSFT', 'IBM', 'SBUX', 'FCX', 'PG', 'BMY', 'MDT', 'SPG', 'VZ', 'OXY', 'CL', 'GILD', 'CVS', 'AMZN', 'GE', 'ABT', 'JNJ', 'UTX', 'WMT', 'ALL', 'PFE', 'FOXA', 'MO', 'MCD', 'MMM', 'SO', 'MON', 'APC', 'NOV', 'APA', 'CMCSA', 'DVN', 'ACN', 'CAT', 'EXC', 'TXN', 'UNP', 'HPQ', 'V', 'LMT', 'RTN', 'CSCO', 'DOW', 'LLY', 'NSC', 'JPM', 'C', 'HAL', 'INTC', 'ABBV', 'UNH', 'MA', 'GM', 'XOM', 'KO', 'EBAY', 'MET', 'GS', 'CVX', 'HON', 'MRK', 'AXP', 'USB', 'EMC', 'DD', 'HD', 'AAPL', 'PM', 'F', 'T', 'UPS', 'SLB', 'AEP', 'EMR', 'COF', 'MDLZ', 'GOOG', 'NKE', 'COP', 'QCOM', 'TGT', 'ORCL', 'GD', 'MS', 'BAC']
	
	
		data = pd.DataFrame()
		for symbol in demo_port:
			data[symbol] = web.DataReader(symbol, data_source='yahoo')['Close']
		data = data.dropna()
	
		spy = pd.DataFrame(data.pop('SPY'))	
	
		#normalize data
		scale_func = lambda x: ( x-x.mean())/x.std()
		#apply PCA
		pca = KernelPCA().fit(data.apply(scale_func))

		get_we = lambda x: x/x.sum()
		#print (get_we(pca.lambdas_)[:20])

		pca_one = KernelPCA(n_components = 1).fit(data.apply(scale_func))
		spy['PCA_1'] = pca_one.transform(data)
		
		
		# Plotting
		spy.apply(scale_func)
		ax= fig.add_subplot(1,1,1, axisbg='#cccccc')
		lin_reg = np.polyval(np.polyfit(spy['PCA_1'],spy['SPY'],1) , spy['PCA_1'])
		
		ax.scatter(spy['PCA_1'], spy['SPY'], c = data.index)
		ax.plot(spy['PCA_1'], lin_reg, 'r', lw = 2)
		ax.set_xlabel('PCA_1')
		ax.set_ylabel('SPY')
		canvas = FigureCanvasTkAgg(fig, self)
		canvas.show()
		canvas.get_tk_widget().pack(side = tk.TOP,fill= tk.BOTH,expand = True)

		toolbar = NavigationToolbar2TkAgg(canvas, self)
		toolbar.update()
		canvas._tkcanvas.pack(side = tk.TOP, fill = tk.BOTH, expand = True)



db = None
HAVE_DB = True

def create_db():
	global db
	global HAVE_DB
	if HAVE_DB:
		db = PersistentDictionary("Historical.db")
		HAVE_DB = False
	else:
		HAVE_DB = True
	
	
def save_db(symbol):
	global db
	data = YStock(symbol)
	days = data.historical()
	if symbol not in db:
		db[symbol] = days

def load_portfolio():
	pass

def demo():
	demo_port = ['SPY','BA', 'WFC', 'PEP', 'AMGN', 'BAX', 'BK', 'FB', 'COST', 'DIS', 'LOW', 'FDX',  'TWX', 'AIG', 'MSFT', 'IBM', 'SBUX', 'FCX', 'PG', 'BMY', 'MDT', 'SPG', 'VZ', 'OXY', 'CL', 'GILD', 'CVS', 'AMZN', 'GE', 'ABT', 'JNJ', 'UTX', 'WMT', 'ALL', 'PFE', 'FOXA', 'MO', 'MCD', 'MMM', 'SO', 'MON', 'APC', 'NOV', 'APA', 'CMCSA', 'DVN', 'ACN', 'CAT', 'EXC', 'TXN', 'UNP', 'HPQ', 'V', 'LMT', 'RTN', 'CSCO', 'DOW', 'LLY', 'NSC', 'JPM', 'C', 'HAL', 'INTC', 'ABBV', 'UNH', 'MA', 'GM', 'XOM', 'KO', 'EBAY', 'MET', 'GS', 'CVX', 'HON', 'MRK', 'AXP', 'USB', 'EMC', 'DD', 'HD', 'AAPL', 'PM', 'F', 'T', 'UPS', 'SLB', 'AEP', 'EMR', 'COF', 'MDLZ', 'GOOG', 'NKE', 'COP', 'QCOM', 'TGT', 'ORCL', 'GD', 'MS', 'BAC']
	
	create_db()
	for symbol in demo_port:
		save_db(symbol)


root = tk.Tk()
myApp = WelcomePage(root)
'''
master = Frame(root, name="master")
master.pack(fill=BOTH)

n = Notebook(master, name='notebook')
n.pack(fill=BOTH, padx = 2, pady = 2)

f1=ttk.Frame(n)

f2=ttk.Frame(n)

f3 = WindowGraph(n)

n.add(f1,text="One")
n.add(f2,text="Two")
n.add(f3,text="Graph")
'''
root.mainloop()
