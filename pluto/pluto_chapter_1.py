
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
class PackTDataAug(object):
  #
  # initialize the object
  def __init__(self, name="Wallaby", is_verbose=True,*args, **kwargs):
    super(PackTDataAug, self).__init__(*args, **kwargs)
    self.author = "Duc Haba"
    self.name = name
    if (is_verbose):
      self._ph()
      self._pp("Hello from class", str(self.__class__) + " Class: " + str(self.__class__.__name__))
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
# Hack it! Add new method as needed.
# add_method() is copy from Michael Garod's blog, 
# https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6
# AND correction by: Филя Усков
#
import functools
def add_method(cls):
  def decorator(func):
    @functools.wraps(func) 
    def wrapper(*args, **kwargs): 
      return func(*args, **kwargs)
    setattr(cls, func.__name__, wrapper)
    return func 
  return decorator
#
pluto = PackTDataAug("Pluto")
@add_method(PackTDataAug)
def say_sys_info(self):
  self._ph()
  now = datetime.datetime.now()
  self._pp("System time", now.strftime("%Y/%m/%d %H:%M"))
  self._pp("Platform", sys.platform)
  self._pp("Python version (3.7+)", sys.version)
  self._pp("PyTorch version (1.11+)", torch.__version__)
  self._pp("Pandas version (1.3.5+)", pandas.__version__)
  self._pp("PIL version (9.0.0+)", PIL.__version__)
  self._pp("Matplotlib version (3.2.2+)", matplotlib.__version__)
  #
  try:
    val = psutil.cpu_count()
    self._pp("CPU count", val)
    val = psutil.cpu_freq()
    if (None != val):
      val = val._asdict()
      self._pp("CPU speed", (str(round((val["current"] / 1000), 2)) + " GHz"))
      self._pp("CPU max speed", (str(round((val["max"] / 1000), 2)) + " GHz"))
    else:
      self._pp("*CPU speed", "NOT available")
  except:
    pass
  self._ph()
  return