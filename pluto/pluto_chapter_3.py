
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

# STOP: if failed the import below, you need to run:
# !pip install -Uqq fastai
#
pluto.version = 3.0
from fastcore.all import *
import fastai
import fastai.vision
import fastai.vision.core
import fastai.vision.augment
import numpy
print("\nfastai version (should be 2.6.3 or higher): ", fastai.__version__)

# prompt: write detail documentation for the following function: draw_image_flip_pil
import PIL
import PIL.ImageOps
@add_method(PacktDataAug)
def draw_image_flip_pil(self,fname):

  """
  Draw an image and its horizontal flipped version.

  Args:
    fname (str): path to the input image.

  Returns:
    None.
  """

  img = PIL.Image.open(fname)
  mirror_img = PIL.ImageOps.mirror(img)
  display(img, mirror_img)
  return

# prompt: write detail documentation for the following function: _make_data_loader
@add_method(PacktDataAug)
def _make_data_loader(self,df, tfms, i_tfms=None):

  """
  Make a data loader from input dataframe using fastai. It is private function.

  Args:
    df (pandas.DataFrame): input dataframe.
    tfms (list): transformations to be applied on the dataloader.
    i_tfms (list, Optional): item transformations to be applied on the dataloader.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """

  dls = fastai.vision.data.ImageDataLoaders.from_df(df,
  fn_col="fname",label_col="label",
  item_tfms=i_tfms,
  batch_tfms=tfms,
  valid_pct=0.2,
  bs=32)
  return dls
#
# fastai.vision.augment.RandomCrop
# fastai.vision.augment.CropPad(480)
# import fastai.vision.augment. (to see all other option like flip, hue, etc)
@add_method(PacktDataAug)
def draw_image_flip(self,df,bsize=15):

  """
  Draw an image and its horizontally flipped version. It also augments the image dataset using horizontal flip and resize.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 15.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """

  aug = [fastai.vision.augment.Flip(p=0.8)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: draw_image_flip_both
@add_method(PacktDataAug)
def draw_image_flip_both(self,df,bsize=15,pad_mode='zeros'):

  """
  Draw an image and its both horizontally and vertically flipped versions.
  It also augments the image dataset using both horizontal and vertical flip and resize.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 15.
    pad_mode (str, optional): padding mode. Defaults to 'zeros'.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """
  aug = fastai.vision.augment.Dihedral(p=0.8,pad_mode=pad_mode)
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: draw_image_crop
@add_method(PacktDataAug)
def draw_image_crop(self,df,bsize=15,pad_mode="zeros",isize=480):

  """
  Draw an image and its cropped versions. It also augments the image dataset using crop and resize.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 15.
    pad_mode (str, optional): padding mode. Defaults to 'zeros'.
    isize (int, optional): image size. Defaults to 480.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """

  aug = fastai.vision.augment.CropPad(isize,pad_mode=pad_mode)
  itfms = fastai.vision.augment.CropPad(isize, pad_mode=pad_mode)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: draw_image_rotate
@add_method(PacktDataAug)
def draw_image_rotate(self,df,bsize=15,max_rotate=45.0,pad_mode='zeros'):

  """
  Draw an image and its rotated versions. It also augments the image dataset using rotation and resize.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 15.
    max_rotate (float, optional): maximum rotation angle. Defaults to 45.0.
    pad_mode (str, optional): padding mode. Defaults to 'zeros'.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """
  aug = [fastai.vision.augment.Rotate(max_rotate,p=0.75,pad_mode=pad_mode)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: draw_image_warp
@add_method(PacktDataAug)
def draw_image_warp(self,df,bsize=15,magnitude=0.2,pad_mode='zeros'):

  """
  Draw an image and its warped versions. It also augments the image dataset using warp and resize.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 15.
    magnitude (float, optional): magnitude of warp. Defaults to 0.2.
    pad_mode (str, optional): padding mode. Defaults to 'zeros'.

  Returns:
    fastai.vision.data.ImageDataLoaders: dataloader object
  """

  aug = [fastai.vision.augment.Warp(magnitude=magnitude, pad_mode=pad_mode,p=0.75)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: draw_image_shift_pil
@add_method(PacktDataAug)
def draw_image_shift_pil(self,fname, x_axis, y_axis=0):

  """
  Draw an image and its shifted versions using PIL library.

  Args:
    fname (str): filepath of the input image.
    x_axis (int): horizontal axis.
    y_axis (int, optional): vertical axis. Defaults to 0.

  Returns:
    None
  """
  img = PIL.Image.open(fname)
  shift_img = PIL.ImageChops.offset(img,x_axis,y_axis)
  display(img, shift_img)
  return

try:
  pluto._ph()
  import albumentations
  pluto._pp("albumentations 1.2.1", "actual " + albumentations.__version__)
# import cat2
except ImportError as e:
  pluto._ph()
  pluto._pp("**Error", e)
pluto._ph()

# prompt: write detail documentation for the following function: _draw_image_album
@add_method(PacktDataAug)
def _draw_image_album(self,df,aug_album,bsize=5):

  """
  Draw an image and its augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    aug_album (albumentations.Compose): albumentations transformation function.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  if (bsize == 2):
    ncol = 2
    nrow = 1
    w = 16
    h = 8
  else:
    ncol = 5
    nrow = int(numpy.ceil(bsize/ncol))
    w = 14
    h = int(4 * nrow)
  #
  canvas, pic = matplotlib.pyplot.subplots(nrow, ncol, figsize=(w, h))
  pics = pic.flatten()
  # select random images
  samp = df.sample(int(ncol * nrow))
  samp.reset_index(drop=True, inplace=True)
  for i, ax in enumerate(pics):
    # convert to an array
    img_numpy = numpy.array(PIL.Image.open(samp.fname[i]))
    label = df.label[i]
    # perform the transformation using albumentations
    img = aug_album(image=img_numpy)['image']
    # display the image in batch modde
    ax.imshow(img)
    ax.set_title(label)
  canvas.tight_layout()
  self._drop_image(canvas)
  canvas.show()
  return

# prompt: write detail documentation for the following function: draw_image_brightness
@add_method(PacktDataAug)
def draw_image_brightness(self,df,brightness=0.2,bsize=5):

  """
  Draw an image and its brightness augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    brightness (float, optional): brightness multiplier. Defaults to 0.2.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=brightness,
    contrast=0.0, saturation=0.0,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_grayscale
@add_method(PacktDataAug)
def draw_image_grayscale(self,df,bsize=5):

  """
  Draw an image and its grayscale versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ToGray(p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_contrast
@add_method(PacktDataAug)
def draw_image_contrast(self,df,contrast=0.2,bsize=5):

  """
  Draw an image and its contrast augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    contrast (float, optional): contrast multiplier. Defaults to 0.2.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=contrast, saturation=0.0,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_saturation
@add_method(PacktDataAug)
def draw_image_saturation(self,df,saturation=0.2,bsize=5):

  """
  Draw an image and its saturation augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    saturation (float, optional): saturation multiplier. Defaults to 0.2.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=0.0, saturation=saturation,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_hue
@add_method(PacktDataAug)
def draw_image_hue(self,df,hue=0.2,bsize=5):

  """
  Draw an image and its hue augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    hue (float, optional): hue multiplier. Defaults to 0.2.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=0.0, saturation=0.0,hue=hue,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_noise
@add_method(PacktDataAug)
def draw_image_noise(self,df,var_limit=(10.0, 50.0),bsize=5):

  """
  Draw an image and its noise augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    var_limit (tuple, optional): noise variance range. Defaults to (10.0, 50.0).
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.GaussNoise(var_limit=var_limit,
    always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_sunflare
@add_method(PacktDataAug)
def draw_image_sunflare(self,df,flare_roi=(0, 0, 1, 0.5),src_radius=400,bsize=2):

  """
  Draw an image and its sunflare augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    flare_roi (tuple, optional): sunflare region of interest. Defaults to (0, 0, 1, 0.5).
    src_radius (int, optional): sunflare source radius. Defaults to 400.
    bsize (int, optional): batch size. Defaults to 2.

  Returns:
    None
  """

  aug_album = albumentations.RandomSunFlare(flare_roi=flare_roi,
    src_radius=src_radius, always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_rain()
@add_method(PacktDataAug)
def draw_image_rain(self,df,drop_length=20, drop_width=1,blur_value=1,bsize=2):

  """
  Draw an image and its rain augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    drop_length (int, optional): drop length. Defaults to 20.
    drop_width (int, optional): drop width. Defaults to 1.
    blur_value (int, optional): blur value. Defaults to 1.
    bsize (int, optional): batch size. Defaults to 2.

  Returns:
    None
  """

  aug_album = albumentations.RandomRain(drop_length=drop_length, drop_width=drop_width,
    blur_value=blur_value,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_sepia
@add_method(PacktDataAug)
def draw_image_sepia(self,df,bsize=5):

  """
  Draw an image and its sepia augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.ToSepia(always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_fancyPCA
@add_method(PacktDataAug)
def draw_image_fancyPCA(self,df,alpha=0.1,bsize=5):

  """
  Draw an image and its FancyPCA augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    alpha (float, optional): alpha value. Defaults to 0.1.
    bsize (int, optional): batch size. Defaults to 5.

  Returns:
    None
  """

  aug_album = albumentations.FancyPCA(alpha=alpha, always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

# prompt: write detail documentation for the following function: draw_image_erasing
@add_method(PacktDataAug)
def draw_image_erasing(self,df,bsize=8,max_count=5):

  """
  Draw an image and its erasing augmented versions using albumentations to do image transformation
  and display it in batch.

  Args:
    df (pandas.DataFrame): input dataframe.
    bsize (int, optional): batch size. Defaults to 8.
    max_count (int, optional): maximum number of times to erase an image. Defaults to 5.

  Returns:
    None
  """

  aug = [fastai.vision.augment.RandomErasing(p=1.0,max_count=max_count)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def print_safe_parameters(self):

  """
  Prints the safe parameters for the given data type.
  The safe parameters are the parameters that are likely to produce good results and do not cause overfitting.
  The function default dataset are following:

    * `Covid-19`: This is the data type for the COVID-19 dataset.
    * `People`: This is the data type for the People dataset.
    * `Fungi`: This is the data type for the Fungi dataset.
    * `Sea Animal`: This is the data type for the Sea Animal dataset.
    * `Food`: This is the data type for the Food dataset.
    * `Mall Crowd`: This is the data type for the Mall Crowd dataset.

  The table has two columns: `Filter` and `Parameter`.
  The `Filter` column lists the different filters that can be used,
  and the `Parameter` column lists the safe parameters for each filter.

  For example, the following is a sample table that is printed by the function for the `Covid-19` data type:

    | Filter              | Parameter |
    |---------------------|-----------|
    | Horizontal Flip     | Yes |
    | Vertical Flip       | Yes |
    | Croping and Padding | pad=border |
    | Rotation            | max_rotate=25.0 |
    | Warping             | magnitude=0.3 |
    | Lighting            | brightness=0.2 |
    | Grayscale           | Yes |
    | Contrast            | contrast=0.1 |
    | Saturation          | saturation=3.5 |
    | Hue Shifting        | hue=0.15 |
    | Noise Injection     | limit=(100.0, 300.0) |
    | Sun Flare           | NA |
    | Rain                | NA |
    | Sepia               | Yes |
    | FancyPCA            | alpha=0.5 |
    | Random Erasing      | max_count=3 |

  """
  data = [['Horizontal Flip','NA','Yes','Yes','Yes','Yes','Yes',],
    ['Vertical Flip','NA','NA','NA','Yes','Yes','NA',],
    ['Croping and Padding','NA','pad=border','pad=border','pad=reflection','pad=reflection','pad=zeros',],
    ['Rotation','NA','max_rotate=25.0','max_rotate=25.0','max_rotate=180.0','max_rotate=180.0','max_rotate=16.0',],
    ['Warping','NA','magnitude=0.3','magnitude=0.3','magnitude=0.4','magnitude=0.4','magnitude=0.3',],
    ['Lighting','brightness=0.2','brightness=0.3','brightness=0.3','brightness=0.4','brightness=0.4','brightness=0.3',],
    ['Grayscale','NA','NA','NA','NA','NA','Yes',],
    ['Contrast','contrast=0.1','contrast=0.3','contrast=0.3','contrast=0.3','contrast=0.4','contrast=0.4',],
    ['Saturation','NA','saturation=3.5','saturation=2.0','saturation=3.0','saturation=3.0','saturation=2.5',],
    ['Hue Shifting','NA','NA','NA','hue=0.15','hue=0.2','hue=0.2',],
    ['Noise Injection','limit=(100.0, 300.0)','limit=(300.0, 500.0)','limit=(200.0, 400.0)','limit=(200.0, 400.0)','limit=(300.0, 400.0)','limit=(300.0, 500.0)',],
    ['Sun Flare','NA','NA','radius=200','NA','NA','NA',],
    ['Rain','NA','NA','length=20','NA','NA','NA',],
    ['Sepia','NA','Yes','NA','NA','NA','NA',],
    ['FancyPCA','NA','alpha=0.5','alpha=0.5','alpha=0.5','alpha=0.5','NA',],
    ['Random Erasing','NA','max_count=3','max_count=3','max_count=4','max_count=4','NA',]]
  # Create the pandas DataFrame
  df = pandas.DataFrame(data, columns=['Filter','Covid-19', 'People', 'Fungi', 'Sea Animal', 'Food', 'Mall Crowd'])
  #
  display(df[['Filter','Covid-19', 'People', 'Fungi']].style.set_table_styles(self._fetch_larger_font()))
  display(df[['Filter','Sea Animal', 'Food', 'Mall Crowd']].style.set_table_styles(self._fetch_larger_font()))
  return

# prompt: write detail documentation for the following class: AlbumentationsTransform
class AlbumentationsTransform(DisplayedTransform):
  """
  This class is used to apply albumentations to the images.

  Args:
    train_aug (object): This is an object of albumentations that contains the image augmentations.

  Attributes:
    split_idx (int): This is the index of the dataset split.
    order (int): This is the order of the transform in the data augmentation pipeline.

  """

  split_idx, order = 0, 2

  def __init__(self, train_aug):
    """
    Initialize the class.

    Args:
      train_aug (object): This is an object of albumentations that contains the image augmentations.

    Returns:
      None.

    """

    store_attr()
    return

  def encodes(self, img: fastai.vision.core.PILImage):
    """
    Encodes the image.

    Args:
      img (object): This is the input image.

    Returns:
      The encoded image.

    """

    aug_img = self.train_aug(image=numpy.array(img))['image']
    return fastai.vision.core.PILImage.create(aug_img)

# prompt: write detail documentation for the following function: _fetch_alumn_covid19()
@add_method(PacktDataAug)
def _fetch_album_covid19(self):

  """
  This function is used to fetch the albumentation for the COVID-19 dataset.

  Args:
    None

  Returns:
    This function returns an albumentation object for the COVID-19 dataset.

  """

  return albumentations.Compose([
  albumentations.GaussNoise(var_limit=(100.0, 300.0), p=0.5)
  ])
#

# prompt: write detail documentation for the following function: draw_augment_covid19()
@add_method(PacktDataAug)
def draw_augment_covid19(self,df,bsize=15):

  """
  This function is used to draw the data loader for the COVID-19 dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the COVID-19 dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the COVID-19 dataset.
  """

  aug = [
    fastai.vision.augment.Brightness(max_lighting=0.3,p=0.5),
    fastai.vision.augment.Contrast(max_lighting=0.4, p=0.5),
    AlbumentationsTransform(self._fetch_album_covid19())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: _fetch_album_people()
@add_method(PacktDataAug)
def _fetch_album_people(self):

  """
  This function is used to fetch the albumentation for the People dataset.

  Args:
    None

  Returns:
    This function returns an albumentation objects for the People dataset.

  """

  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.3, contrast=0.4, saturation=3.5,hue=0.0, p=0.5),
  albumentations.ToSepia(p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(300.0, 500.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_people(self,df,bsize=15):

  """
  This function is used to draw the data loader for the People dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the People dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the People dataset.
  """

  aug = [
    fastai.vision.augment.Flip(p=0.5),
    fastai.vision.augment.Rotate(25.0,p=0.5,pad_mode='border'),
    fastai.vision.augment.Warp(magnitude=0.3, pad_mode='border',p=0.5),
    fastai.vision.augment.RandomErasing(p=0.5,max_count=2),
    AlbumentationsTransform(self._fetch_album_people())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def _fetch_album_fungi(self):

  """
  This function is used to fetch the albumentation for the Fungi dataset.

  Args:
    None

  Returns:
    This function returns an albumentation objects for the Fungi dataset.
  """

  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.3, contrast=0.4, saturation=2.0,hue=0.0, p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.RandomSunFlare(flare_roi=(0, 0, 1, 0.5),src_radius=200, always_apply=True, p=0.5),
  albumentations.RandomRain(drop_length=20, drop_width=1.1,blur_value=1.1,always_apply=True, p=0.5),
  albumentations.GaussNoise(var_limit=(200.0, 400.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_fungi(self,df,bsize=15):

  """
  This function is used to draw the data loader for the Fungi dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the Fungi dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the Fungi dataset.
  """

  aug = [
    fastai.vision.augment.Flip(p=0.5),
    fastai.vision.augment.Rotate(25.0,p=0.5,pad_mode='border'),
    fastai.vision.augment.Warp(magnitude=0.3, pad_mode='border',p=0.5),
    fastai.vision.augment.RandomErasing(p=0.5,max_count=2),
    AlbumentationsTransform(self._fetch_album_fungi())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def _fetch_album_sea_animal(self):

  """
  This function is used to fetch the albumentation for the Sea Animal dataset.

  Args:
    None

  Returns:
    This function returns an albumentation objects for the Sea Animal dataset.
  """

  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=2.0,hue=1.5, p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(200.0, 400.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_sea_animal(self,df,bsize=15):

  """
  This function is used to draw the data loader for the Sea Animal dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the Sea Animal dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the Sea Animal dataset.
  """
  aug = [
    fastai.vision.augment.Dihedral(p=0.5,pad_mode='reflection'),
    fastai.vision.augment.Rotate(180.0,p=0.5,pad_mode='reflection'),
    fastai.vision.augment.Warp(magnitude=0.3, pad_mode='reflection',p=0.5),
    fastai.vision.augment.RandomErasing(p=0.5,max_count=2),
    AlbumentationsTransform(self._fetch_album_sea_animal())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def _fetch_album_food(self):

  """
  This function is used to fetch the albumentation for the Vietnam food dataset.

  Args:
    None

  Returns:
    This function returns an albumentation objects for the Vietnam good dataset.
  """

  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=2.0,hue=1.5, p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(200.0, 400.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_food(self,df,bsize=15):

  """
  This function is used to draw the data loader for the Vietnam food dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the Vietname food dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the Vietnam food dataset.
  """

  aug = [
    fastai.vision.augment.Dihedral(p=0.5,pad_mode='reflection'),
    fastai.vision.augment.Rotate(180.0,p=0.5,pad_mode='reflection'),
    fastai.vision.augment.Warp(magnitude=0.3, pad_mode='reflection',p=0.5),
    fastai.vision.augment.RandomErasing(p=0.5,max_count=2),
    AlbumentationsTransform(self._fetch_album_food())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def _fetch_album_crowd(self):

  """
  This function is used to fetch the albumentation for the Mall Crowd dataset.

  Args:
    None

  Returns:
    This function returns an albumentation objects for the Mall Crowd good dataset.
  """

  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.3, contrast=0.4, saturation=3.5,hue=0.0, p=0.5),
  albumentations.ToSepia(p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(300.0, 500.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_crowd(self,df,bsize=15):

  """
  This function is used to draw the data loader for the Mall Crowd dataset.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the Mall Crowd dataset.
    bsize (int): This is the batch size for the data loader. Default is 15.

  Returns:
    fastai.data.DataLoader: returns a data loader for the Mall Crowd dataset.
  """

  aug = [
    fastai.vision.augment.Flip(p=0.5),
    fastai.vision.augment.Rotate(25.0,p=0.5,pad_mode='zeros'),
    fastai.vision.augment.Warp(magnitude=0.3, pad_mode='zeros',p=0.5),
    fastai.vision.augment.RandomErasing(p=0.5,max_count=2),
    AlbumentationsTransform(self._fetch_album_crowd())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

# prompt: write detail documentation for the following function: _draw_image_teaser
@add_method(PacktDataAug)
def _draw_image_teaser(self,df,aug_album,label='augment image'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    aug_album (albumentations.Compose): This is an albumentation object that will be used to
      perform the augmentation on the image.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "augment image"

  Returns:
    None
  """

  canvas, pic = matplotlib.pyplot.subplots(1, 2, figsize=(12, 6))
  #pics = pic.flatten()
  # select random images
  samp = df.sample(1)
  samp.reset_index(drop=True, inplace=True)
  #
  orig_img = PIL.Image.open(samp.fname[0])
  pic[0].imshow(orig_img)
  pic[0].set_title('Original')
  #
  img_numpy = numpy.array(orig_img)
  img = aug_album(image=img_numpy)['image']
  pic[1].imshow(img)
  pic[1].set_title(label)
  #
  canvas.tight_layout()
  self._drop_image(canvas)
  canvas.show()
  return

# prompt: write detail documentation for the following function: draw_image_teaser_brightness
@add_method(PacktDataAug)
def draw_image_teaser_brightness(self,df,brightness=0.2,label='Brightness'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    brightness (float): This is a float that will be used to adjust the brightness of the image.
      Default is 0.2.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "augment image"

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=brightness,
    contrast=0.0, saturation=0.0,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_flip
@add_method(PacktDataAug)
def draw_image_teaser_flip(self,df,label='Verticle Flip'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Vericle Flip"

  Returns:
    None
  """

  aug_album = albumentations.VerticalFlip(always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_crop
@add_method(PacktDataAug)
def draw_image_teaser_crop(self,df,label='Center Crop'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Center Crop"

  Returns:
    None
  """

  aug_album = albumentations.CenterCrop(500, 500, always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_resize

@add_method(PacktDataAug)
def draw_image_teaser_resize(self,df,label='Resize with squishing mode'):

    """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Center Crop"

  Returns:
    None
  """

  aug_album = albumentations.Resize(500,500, always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_rotate
@add_method(PacktDataAug)
def draw_image_teaser_rotate(self,df,label='Rotate and Reflection Padding'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Rotate and Reflection Padding"

  Returns:
    None
  """

  aug_album = albumentations.Rotate(limit=(40,70), always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_noise
@add_method(PacktDataAug)
def draw_image_teaser_noise(self,df,label='Noice injection using Gaussian method'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Noice injection using Gaussian method"

  Returns:
    None
  """

  aug_album = albumentations.GaussNoise((800, 1000), always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return

# prompt: write detail documentation for the following function: draw_image_teaser_hue
@add_method(PacktDataAug)
def draw_image_teaser_hue(self,df,label='Hue shifting'):

  """
  This function is used to draw the image teaser for the book.

  Args:
    df (pandas.DataFrame): This is a pandas DataFrame that contains the data for the dataset.
    label (str): This is a string that will be used to label the image that has been augmented.
      Default is "Hue shifting"

  Returns:
    None
  """

  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=0.0, saturation=0.0,hue=0.5,always_apply=True, p=1.0)
  self._draw_image_teaser(df,aug_album,label)
  return
