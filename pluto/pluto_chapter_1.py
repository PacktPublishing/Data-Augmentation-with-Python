
# AI function documentation
#
# prompt: write documentation for the following function: add_method()
# Note: B-grade, cycle the above to one function at a time. Give multiple
# functions will confused it.

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
  """
    The PacktDataAug class is the based class for the
    "Data Augmentation with Python" book.
  """
  #
  # initialize the object
  def __init__(self, name="Pluto", is_verbose=True,*args, **kwargs):
    """

    This is the constructor function.

    Args:

     name (str): It requires a name for the object. The default is 'Pluto'
     verbose (bool):  The default value of `verbose` is True. This function prints out the
        name of the object if `is_verbose == True`. This is used to debug
        code. When you are ready to deploy the model, then you should set
        `is_verbose == False` in order to avoid printing out diagnostic
        messages.

      Additionally, this function takes any number of other
      parameters. These parameters are stored in `**kwargs` and are
      accessed via the function `get_kwargs()`. See the documentation
      for `get_kwargs()` for more details.
      Note that `__init__()` is
      automatically called when you create a new object.

    Returns:
      None.
    """
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
    """

      pretty print output name-value line

      Args:
          a (str): Name of key
          b (any): value of key

      Returns:
          None
    """
    print("%28s : %s" % (str(a), str(b)))
    return
  #
  # pretty print the header or footer lines
  def _ph(self):
    """
      pretty print the header or footer lines

      Args:
          None

      Returns:
          None
      """
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
  """

    Decorator creates a new method to class
    `x` with the same name and parameters as function `z`
    Args:
        x: class to add function
        z: function to add to class `x`
    Returns:
        a decorator
  """
  def dec(z):
    @functools.wraps(z)
    def y(*args, **kwargs):
      return z(*args, **kwargs)
    setattr(x, z.__name__, y)
    return z
  return dec
#

# create pluto (or any name you choose)
pluto = PacktDataAug("Pluto")

@add_method(PacktDataAug)
def say_sys_info(self):
  """

    Print out system information. Useful for
    debugging purposes. Prints out information such as
    the system time, platform, Python version, PyTorch
    version, Pandas version, PIL version, and
    Matplotlib version. Also prints the number of CPU
    cores and the CPU speed.

    Note that this function is added to the class `PacktDataAug` via
    the decorator `@add_method()`. This means that you can
    call this function as `p.say_system_info()`,
    where `p` is an instance of `PacktDAtaAug`.

    Args:
      None

    Returns:
      None
  """
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
