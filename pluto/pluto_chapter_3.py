
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

pluto = PacktDataAug("Pluto")

@add_method(PacktDataAug)
def say_sys_info(self):
  self._ph()
  now = datetime.datetime.now()
  self._pp("System time", now.strftime("%Y/%m/%d %H:%M"))
  self._pp("Platform", sys.platform)
  self._pp("Pluto Version (Chapter)", self.version)
  self._pp("Python (3.7.10)", 'actual: ' + ''.join(str(sys.version).splitlines()))
  self._pp("PyTorch (1.11.0)", 'actual: ' + str(torch.__version__))
  self._pp("Pandas (1.3.5)", 'actual: ' + str(pandas.__version__))
  self._pp("PIL (9.0.0)", 'actual: ' + str(PIL.__version__))
  self._pp("Matplotlib (3.2.2)", 'actual: ' + str(matplotlib.__version__))
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

pluto.version = 2.0
@add_method(PacktDataAug)
def remember_kaggle_access_keys(self,username,key):
  self.kaggle_username = username
  self.kaggle_key = key
  return

@add_method(PacktDataAug)
def _write_kaggle_credit(self):
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
  #self._write_kaggle_credit()  # need to run only once.
  path = pathlib.Path(cname)
  kaggle.api.competition_download_cli(str(path))
  zipfile.ZipFile(f'{path}.zip').extractall(path)
  return
#
#
@add_method(PacktDataAug)
def fetch_kaggle_dataset(self,url,dest="kaggle"):
  #self._write_kaggle_credit()    # need to run only once.
  opendatasets.download(url,data_dir=dest)
  return

import zipfile
import os

import zipfile
import os

@add_method(PacktDataAug)
def fetch_df(self, csv):
  df = pandas.read_csv(csv, encoding='latin-1')
  return df

@add_method(PacktDataAug)
def build_sf_fname(self, df):
  root = 'state-farm-distracted-driver-detection/imgs/train/'
  df["fname"] = root + df.classname + '/' + df.img
  return

# set internal counter for image to be zero, e.g. pluto0.jpg, pluto1.jpg, etc.
pluto.fname_id = 0

@add_method(PacktDataAug)
def _drop_image(self,canvas, fname=None,format=".jpg",dname="Data-Augmentation-with-Python/pluto_img"):
  if (fname is None):
    self.fname_id += 1
    if not os.path.exists(dname):
      os.makedirs(dname)
    fn = dname + "/pluto" + str(self.fname_id) + format
  else:
    fn = fname
  canvas.savefig(fn, cmap="Greys", bbox_inches="tight", pad_inches=0.25)
  return
#
@add_method(PacktDataAug)
def draw_batch(self,df_filenames, disp_max=10,is_shuffle=False, figsize=(16,8)):
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

@add_method(PacktDataAug)
def build_shoe_fname(self, start_path):
  df = pandas.DataFrame()
  for root, dirs, files in os.walk(start_path, topdown=False):
   for name in files:
      f = os.path.join(root, name)
      p = pathlib.Path(f).parent.name 
      d = pandas.DataFrame({'fname': [f], 'label': [p]})
      df = df.append(d, ignore_index=True)
  #
  # clean it up
  df = df.reset_index(drop=True)
  return df
#
# create the same with a generic function name
@add_method(PacktDataAug)
def make_dir_dataframe(self, start_path):
  return self.build_shoe_fname(start_path)

@add_method(PacktDataAug)
def print_batch_text(self,df_orig, disp_max=10, cols=["title", "description"]): 
  df = df_orig[cols] 
  with pandas.option_context("display.max_colwidth", None):
    display(df.sample(disp_max))
  return

@add_method(PacktDataAug)
def count_word(self, df, col_dest="description"):
  df['wordc'] = df[col_dest].apply(lambda x: len(x.split()))
  return

@add_method(PacktDataAug)
def draw_word_count(self,df, wc='wordc'):
  canvas, pic = matplotlib.pyplot.subplots(1,2, figsize=(16,5))
  df.boxplot(ax=pic[0],column=[wc],vert=False,color="black")
  df[wc].hist(ax=pic[1], color="cornflowerblue", alpha=0.9)
  #
  title=["Description BoxPlot", "Description Histogram"]
  yaxis=["Description", "Stack"]
  x1 = "Word Count: Mean: " + str(round(df[wc].mean(), 2)) + ", Min: " + str(df[wc].min()) + ", Max: " + str(df[wc].max())
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

import re
@add_method(PacktDataAug)
def _strip_punc(self,s):
  p = re.sub(r'[^\w\s]','',s)
  return(p)
#
@add_method(PacktDataAug)
def check_spelling(self,df, col_dest='description'):
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

import PIL
import PIL.ImageOps
@add_method(PacktDataAug)
def draw_image_flip_pil(self,fname):
  img = PIL.Image.open(fname)
  mirror_img = PIL.ImageOps.mirror(img)
  display(img, mirror_img)
  return

@add_method(PacktDataAug)
def _make_data_loader(self,df, tfms, i_tfms=None):
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
  aug = [fastai.vision.augment.Flip(p=0.8)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def draw_image_flip_both(self,df,bsize=15,pad_mode='zeros'):
  aug = fastai.vision.augment.Dihedral(p=0.8,pad_mode=pad_mode)
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def draw_image_crop(self,df,bsize=15,pad_mode="zeros",isize=480):
  aug = fastai.vision.augment.CropPad(isize,pad_mode=pad_mode)
  itfms = fastai.vision.augment.CropPad(isize, pad_mode=pad_mode)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def draw_image_rotate(self,df,bsize=15,max_rotate=45.0,pad_mode='zeros'):
  aug = [fastai.vision.augment.Rotate(max_rotate,p=0.75,pad_mode=pad_mode)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def draw_image_warp(self,df,bsize=15,magnitude=0.2,pad_mode='zeros'):
  aug = [fastai.vision.augment.Warp(magnitude=magnitude, pad_mode=pad_mode,p=0.75)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def draw_image_shift_pil(self,fname, x_axis, y_axis=0):
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

@add_method(PacktDataAug)
def _draw_image_album(self,df,aug_album,bsize=5):
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
  canvas.show()
  return

@add_method(PacktDataAug)
def draw_image_brightness(self,df,brightness=0.2,bsize=5):
  aug_album = albumentations.ColorJitter(brightness=brightness,
    contrast=0.0, saturation=0.0,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_grayscale(self,df,bsize=5):
  aug_album = albumentations.ToGray(p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_contrast(self,df,contrast=0.2,bsize=5):
  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=contrast, saturation=0.0,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_saturation(self,df,saturation=0.2,bsize=5):
  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=0.0, saturation=saturation,hue=0.0,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_hue(self,df,hue=0.2,bsize=5):
  aug_album = albumentations.ColorJitter(brightness=0.0,
    contrast=0.0, saturation=0.0,hue=hue,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_noise(self,df,var_limit=(10.0, 50.0),bsize=5):
  aug_album = albumentations.GaussNoise(var_limit=var_limit,
    always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_sunflare(self,df,flare_roi=(0, 0, 1, 0.5),src_radius=400,bsize=2):
  aug_album = albumentations.RandomSunFlare(flare_roi=flare_roi,
    src_radius=src_radius, always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_rain(self,df,drop_length=20, drop_width=1,blur_value=1,bsize=2):
  aug_album = albumentations.RandomRain(drop_length=drop_length, drop_width=drop_width,
    blur_value=blur_value,always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_sepia(self,df,bsize=5):
  aug_album = albumentations.ToSepia(always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_fancyPCA(self,df,alpha=0.1,bsize=5):
  aug_album = albumentations.FancyPCA(alpha=alpha, always_apply=True, p=1.0)
  self._draw_image_album(df,aug_album,bsize)
  return

@add_method(PacktDataAug)
def draw_image_erasing(self,df,bsize=8,max_count=5):
  aug = [fastai.vision.augment.RandomErasing(p=1.0,max_count=max_count)]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def print_safe_parameters(self):
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
  display(df)
  return

class AlbumentationsTransform(DisplayedTransform):
  split_idx,order=0,2
  def __init__(self, train_aug): store_attr()
  
  def encodes(self, img: fastai.vision.core.PILImage):
    aug_img = self.train_aug(image=numpy.array(img))['image']
    return fastai.vision.core.PILImage.create(aug_img)

@add_method(PacktDataAug)
def _fetch_album_covid19(self):
  return albumentations.Compose([
  albumentations.GaussNoise(var_limit=(100.0, 300.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_covid19(self,df,bsize=15):
  aug = [
    fastai.vision.augment.Brightness(max_lighting=0.3,p=0.5),
    fastai.vision.augment.Contrast(max_lighting=0.4, p=0.5),
    AlbumentationsTransform(self._fetch_album_covid19())
    ]
  itfms = fastai.vision.augment.Resize(480)
  dsl_org = self._make_data_loader(df, aug,itfms)
  dsl_org.show_batch(max_n=bsize)
  return dsl_org

@add_method(PacktDataAug)
def _fetch_album_people(self):
  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.3, contrast=0.4, saturation=3.5,hue=0.0, p=0.5),
  albumentations.ToSepia(p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(300.0, 500.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_people(self,df,bsize=15):
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
  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=2.0,hue=1.5, p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(200.0, 400.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_sea_animal(self,df,bsize=15):
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
  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.4, contrast=0.4, saturation=2.0,hue=1.5, p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(200.0, 400.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_food(self,df,bsize=15):
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
  return albumentations.Compose([
  albumentations.ColorJitter(brightness=0.3, contrast=0.4, saturation=3.5,hue=0.0, p=0.5),
  albumentations.ToSepia(p=0.5),
  albumentations.FancyPCA(alpha=0.5, p=0.5),
  albumentations.GaussNoise(var_limit=(300.0, 500.0), p=0.5)
  ])
#
@add_method(PacktDataAug)
def draw_augment_crowd(self,df,bsize=15):
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
