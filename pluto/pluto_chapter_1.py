
# create an object
# First, importing the basic library
import torch
import pandas
import numpy
import matplotlib
import pathlib
import PIL
import datetime
import sys
import psutil
# create class/object 
class PacktDataAug(object):
  #
  # initialize the object
  def __init__(self, name="Pluto", is_verbose=True,*args, **kwargs):
    super(PacktDataAug, self).__init__(*args, **kwargs)
    self.author = "Duc Haba"
    self.version = 1.0
    self.name = name
    if (is_verbose):
      self._ph()
      self._pp("Hello from class", f"{self.__class__} Class: {self.__class__.__name__}")
      self._pp("Code name", self.name)
      self._pp("Author is", self.author)
      self._ph()
    #
    return
  #
  # pretty print output name-value line
  def _pp(self, a, b):
    print("%28s : %s" % (str(a), str(b)))
    return
  #
  # pretty print the header or footer lines
  def _ph(self):
    print("-" * 28, ":", "-" * 28)
    return
# ---end of class
#
# Hack it! Add new decorator
# add_method() is inspired Michael Garod's blog, 
# AND correction by: Филя Усков
#
import functools
def add_method(x):
  def dec(z):
    @functools.wraps(z) 
    def y(*args, **kwargs): 
      return z(*args, **kwargs)
    setattr(x, z.__name__, y)
    return z 
  return dec
#

pluto = PacktDataAug("Pluto")

@add_method(PacktDataAug)
def say_sys_info(self):
  self._ph()
  now = datetime.datetime.now()
  self._pp("System time", now.strftime("%Y/%m/%d %H:%M"))
  self._pp("Platform", sys.platform)
  self._pp("Pluto Version (Chapter)", self.version)
  v = sys.version.replace('\n', '')
  self._pp("Python (3.7.10)", f'actual: {v}')
  self._pp("PyTorch (1.11.0)", f'actual: {torch.__version__}')
  self._pp("Pandas (1.3.5)", f'actual: {pandas.__version__}')
  self._pp("PIL (9.0.0)", f'actual: {PIL.__version__}')
  self._pp("Matplotlib (3.2.2)", f'actual: {matplotlib.__version__}')
  #
  try:
    val = psutil.cpu_count()
    self._pp("CPU count", val)
    val = psutil.cpu_freq()
    if (None != val):
      val = val._asdict()
      self._pp("CPU speed",  f'{val["current"]/1000:.2f} GHz')
      self._pp("CPU max speed", f'{val["max"]/1000:.2f} GHz') 
    else:
      self._pp("*CPU speed", "NOT available")
  except:
    pass
  self._ph()
  return
