
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

# prompt: write python documentation for the method: pluto.remember_kaggle_access_keys()

pluto.version = 2.0
import opendatasets
#
@add_method(PacktDataAug)
def remember_kaggle_access_keys(self,username,key):
  """
    This method takes a username and a Kaggle API key as arguments and stores
    them in the class object.

    Args:
     username (str): The Kaggle username.
     key (str): The Kaggle API key.

    Returns:
     None
  """

  self.kaggle_username = username
  self.kaggle_key = key
  return

# write python documentation for the following method: pluto.remember_kaggle_access_keys()
@add_method(PacktDataAug)
def _write_kaggle_credit(self):
  """
  This method writes the Kaggle credentials to the json file.
  The Kaggle credentials include the username and the API key.
  The value is stored in the class variables. It starts with
  underscore, and so it meant for internal/private use.
  The json file is written in the following format:

  {
    "username":"<username>",
    "key":"<API Key>"
  }

  The credentials are written to the following two locations:

  - ~/.kaggle/kaggle.json
  - ./kaggle.json

  Args:
    None

  Returns:
    None
  """
  creds = '{"username":"'+self.kaggle_username+'","key":"'+self.kaggle_key+'"}'
  kdirs = ["~/.kaggle/kaggle.json", "./kaggle.json"]
  #
  for k in kdirs:
    cred_path = pathlib.Path(k).expanduser()
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)
  import kaggle
  #
  return
#
@add_method(PacktDataAug)
def fetch_kaggle_comp_data(self,cname):
  """
  This method downloads and unzip the data from the Competition on Kaggle.
  You need to join the competition before downloading the data.

  Args:
    cname (str): The name of the competition on Kaggle.

  Returns:
    None
  """
  path = pathlib.Path(cname)
  kaggle.api.competition_download_cli(str(path))
  zipfile.ZipFile(f'{path}.zip').extractall(path)
  return
#
#
@add_method(PacktDataAug)
def fetch_kaggle_dataset(self,url,dest="kaggle"):
  """
  This method downloads the data from the Kaggle's dataset.
  You need NOT to join the competition before downloading the data.

  Args:
    url (str): The url of the dataset on Kaggle.
    dest (str): the destination path, default is "kaggle" directory.

  Returns:
    None
  """
  opendatasets.download(url,data_dir=dest)
  return

import zipfile
import os

# write python documentation for the following method: fetch_df
@add_method(PacktDataAug)
def fetch_df(self, csv,sep=','):
  """
  This method reads and loads a CSV file into a Pandas DataFrame.

  Args:
    csv (str): The path to a CSV file.
    sep (str): The column separator character, default is comma ",".

  Returns:
    DataFrame: A pandas DataFrame.
  """
  df = pandas.read_csv(csv, encoding='latin-1', sep=sep)
  return df
#
@add_method(PacktDataAug)
def _fetch_larger_font(self):
  """
  This method fetches CSS styles for larger font.

  Args:
    None

  Returns:
    dfstyle (dict): The dictionary containing CSS styles for larger font.
  """
  heading_properties = [('font-size', '20px')]
  cell_properties = [('font-size', '18px')]
  dfstyle = [dict(selector="th", props=heading_properties),
    dict(selector="td", props=cell_properties)]
  return dfstyle

# prompt: write the python inline documentation for the following function: build_sf_fname
@add_method(PacktDataAug)
def build_sf_fname(self, df):
  """
  This method builds the file name for a given row in the State Farm DataFrame.

  Args:
    df (Pandas DataFrame): The State Farm DataFrame.

  Returns:
    None
  """
  root = 'state-farm-distracted-driver-detection/imgs/train/'
  df["fname"] = root + df.classname+'/'+df.img
  return

# prompt: write the python inline documentation for the following function: draw_batch
pluto.fname_id = 0
#
@add_method(PacktDataAug)
def _drop_image(self,canvas,
  fname=None,
  format=".jpg",
  dname="Data-Augmentation-with-Python/pluto_img"):
  """
  This method saves the output image to a file.

  Args:
    canvas (matplotlib.pyplot.Figure): The output image from plt.savefig.
    fname (str): The file name. If None, the file name will be generated.
    format (str): The file format, e.g. jpg or png. Default is ".jpg"
    dname (str): The directory where the file needs to be saved. Default is
    "Data-Augmentation-with-Python/pluto_img"

  Returns:
    None
  """
  if (fname is None):
    self.fname_id += 1
    if not os.path.exists(dname):
      os.makedirs(dname)
    fn = f'{dname}/pluto{self.fname_id}{format}'
  else:
    fn = fname
  canvas.savefig(fn, bbox_inches="tight", pad_inches=0.25)
  return
#
@add_method(PacktDataAug)
def draw_batch(self,df_filenames,
  disp_max=10,
  is_shuffle=False,
  figsize=(16,8)):

  """
  This method draws the images specified in the DataFrame.

  Args:
    df_filenames (Pandas DataFrame): The DataFrame containing the file names.
    disp_max (int): The maximum number of images to be drawn. Default is 10.
    is_shuffle (bool): Whether to shuffle the images. Default is False.
    figsize (tuple): The figure size. Default is (16,8).

  Returns:
    None
  """

  disp_col = 5
  disp_row = int(numpy.round((disp_max/disp_col)+0.4, 0))
  _fns = list(df_filenames)
  if (is_shuffle):
    numpy.random.shuffle(_fns)
  k = 0
  clean_fns = []
  if (len(_fns) >= disp_max):
    canvas, pic = matplotlib.pyplot.subplots(disp_row,disp_col, figsize=figsize)
    for i in range(disp_row):
      for j in range(disp_col):
        try:
          im = PIL.Image.open(_fns[k])
          pic[i][j].imshow(im)
          pic[i][j].set_title(pathlib.Path(_fns[k]).name)
          clean_fns.append(_fns[k])
        except:
          pic[i][j].set_title(pathlib.Path(_fns[k]).name)
        k += 1
    canvas.tight_layout()
    self._drop_image(canvas)
    canvas.show()
  else:
    print("**Warning: the length should be more then ", disp_max, ". The given length: ", len(_fns))
  return clean_fns

# prompt: write the python inline documentation for the following function: build_shoe_fname
@add_method(PacktDataAug)
def build_shoe_fname(self, start_path):
  """
  This method builds the file name for a given directory.

  Args:
    start_path (str): The starting directory.

  Returns:
    DataFrame: The DataFrame containing the file names.
  """
  df = pandas.DataFrame()
  for root, dirs, files in os.walk(start_path, topdown=False):
   for name in files:
      f = os.path.join(root, name)
      p = pathlib.Path(f).parent.name
      d = pandas.DataFrame({'fname': [f], 'label': [p]})
      df = pandas.concat([df, d], ignore_index=True)
      #df = df.append(d, ignore_index=True)
  #
  # clean it up
  df = df.reset_index(drop=True)
  return df
#
# create the same with a generic function name
@add_method(PacktDataAug)
def make_dir_dataframe(self, start_path):
  """
  This method builds the file name for a given directory.

  Args:
    start_path (str): The starting directory.

  Returns:
    DataFrame: The DataFrame containing the file names.
  """
  return self.build_shoe_fname(start_path)

# prompt: write the python inline documentation for the following function: print_batch_text
@add_method(PacktDataAug)
def print_batch_text(self,df_orig, 
  disp_max=6, 
  cols=["title", "description"],
  is_larger_font=True):

  """
  This method shows a batch of text data.

  Args:
    df_orig (DataFrame): The input DataFrame.
    disp_max (int): The maximum number of rows to be displayed. Default is 6.
    cols (list): The list of columns to display. Default is ["title", "description"].
    is_larger_font (bool): Whether to use larger font. Default is True.

  Returns:
    None
  """
  df = df_orig[cols]
  with pandas.option_context("display.max_colwidth", None):
    if (is_larger_font):
      display(df.sample(disp_max).style.set_table_styles(self._fetch_larger_font()))
    else:
      display(df.sample(disp_max))
  return

# prompt: write the python inline documentation for the following function: count_word
@add_method(PacktDataAug)
def count_word(self, df, col_dest="description"):
  """
  This method counts the number of words in a column named "wordc"

  Args:
    df (DataFrame): The input DataFrame.
    col_dest (str): The column name to be counted, default "description"

  Returns:
    None
  """
  df['wordc'] = df[col_dest].apply(lambda x: len(x.split()))
  return

# prompt: write the python inline documentation for the following function: draw_word_count
@add_method(PacktDataAug)
def draw_word_count(self,df, wc='wordc',is_stack_verticle=True):
  """
  This method creates two plots:
    1. a boxplot of word count
    2. a histogram of word count

  Args:
    df (DataFrame): The input DataFrame.
    wc (str): The column name of word count, default "wordc".
    is_stack_verticle (bool): Whether to stack the two plots vertically. Default is True.

  Returns:
    None
  """
  if (is_stack_verticle):
    canvas, pic = matplotlib.pyplot.subplots(2,1, figsize=(8,10))
  else:
    canvas, pic = matplotlib.pyplot.subplots(1,2, figsize=(16,5))
  df.boxplot(ax=pic[0],column=[wc],vert=False,color="black")
  df[wc].hist(ax=pic[1], color="cornflowerblue", alpha=0.9)
  #
  title=["Description BoxPlot", "Description Histogram"]
  yaxis=["Description", "Stack"]
  x1 = f'Word Count: Mean: {df[wc].mean():0.2f}, Min: {df[wc].min()}, Max: {df[wc].max()}'
  xaxis=[x1, "Word Count"]
  #
  pic[0].set_title(title[0], fontweight ="bold")
  pic[1].set_title(title[1], fontweight ="bold")
  pic[0].set_ylabel(yaxis[0])
  pic[1].set_ylabel(yaxis[1])
  pic[0].set_xlabel(xaxis[0])
  pic[1].set_xlabel(xaxis[1])
  #
  canvas.tight_layout()
  self._drop_image(canvas)
  #
  canvas.show()
  return

# prompt: write the python inline documentation for the following function: _strip_punc
import re
import spellchecker
@add_method(PacktDataAug)
def _strip_punc(self,s):
  """
  This method removes all punctuation from a string.

  Args:
    s (str): The input string.

  Returns:
    str: The string without punctuation.
  """
  p = re.sub(r'[^\w\s]','',s)
  return(p)
#
@add_method(PacktDataAug)
def check_spelling(self,df, col_dest='description'):
  """
  This method checks the spelling in a column and returns a new column 
  "misspelled" and "misspelled_count".

  Args:
    df (DataFrame): The input DataFrame.
    col_dest (str): The column name to be checked, default "description"

  Returns:
    None
  """
  spell = spellchecker.SpellChecker()
  df["misspelled"] = df[col_dest].apply(lambda x: spell.unknown(self._strip_punc(x).split()))
  df["misspelled_count"] = df["misspelled"].apply(lambda x: len(x))
  return
