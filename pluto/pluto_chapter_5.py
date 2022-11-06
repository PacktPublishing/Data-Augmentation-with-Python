
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

pluto.version = 5.0
import missingno
@add_method(PacktDataAug)
def draw_text_null_data(self, df, color=(0.3,0.36,0.44)):
  canvas, pic = matplotlib.pyplot.subplots(1, 1, figsize=(16, 6))
  missingno.matrix(df,color=color,ax=pic)
  pic.set_title('Missing Data (Null Value)')
  pic.set_xlabel('Solid is has data. White line is missing/null data.')
  canvas.tight_layout()
  self._drop_image(canvas)
  canvas.show()
  return

import nltk
import wordcloud

import re
@add_method(PacktDataAug)
def _draw_image_wordcloud(self, words_str, xignore_words='cat', title='Word Cloud:'):
  canvas, pic = matplotlib.pyplot.subplots(1, 1, figsize=(16, 8))
  img = wordcloud.WordCloud(width = 1600, 
    height = 800, 
    background_color ='white',
    stopwords = xignore_words, 
    min_font_size = 10).generate(words_str) 
  pic.imshow(img)
  pic.set_title(title)
  pic.set_xlabel(f'Approximate Words: {int(len(words_str) / 5)}')
  pic.tick_params(left = False, right = False, labelleft = False,
    labelbottom = False, bottom = False)
  canvas.tight_layout()
  self._drop_image(canvas)
  canvas.show()
  return
  #
@add_method(PacktDataAug)
def draw_text_wordcloud(self, df_1column, xignore_words='cat', title='Word Cloud:'):
  orig = df_1column.str.cat()
  clean = re.sub('[^A-Za-z0-9 ]+', '', orig)
  self._draw_image_wordcloud(clean, xignore_words=xignore_words,title=title)
  return

@add_method(PacktDataAug)
def _drop_df_file(self, df,fname,type='csv',sep='~'):
  df.to_csv(fname,sep=sep)
  return 

@add_method(PacktDataAug)
def print_aug_ocr(self, df, col_dest="description",bsize=3, aug_name='Augmented'):
  aug_func = nlpaug.augmenter.char.OcrAug()
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_keyboard(self, df, col_dest="description",bsize=3, aug_name='Keyboard Augment'):
  aug_func = nlpaug.augmenter.char.KeyboardAug()
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_char_random(self, df, action='insert', col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.char.RandomCharAug(action=action)
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_misspell(self, df, col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.SpellingAug()
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_split(self, df, col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.SplitAug()
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_random(self, df, action='swap', col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.RandomWordAug(action=action)
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_synonym(self, df, col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.SynonymAug(aug_src='wordnet')
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_antonym(self, df, col_dest="description",bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.AntonymAug()
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

@add_method(PacktDataAug)
def print_aug_word_reserved(self, df, col_dest="description",reserved_tokens=None,bsize=3, aug_name='Augment'):
  aug_func = nlpaug.augmenter.word.ReservedAug(reserved_tokens=reserved_tokens)
  self._print_aug_batch(df, aug_func,col_dest=col_dest,bsize=bsize, aug_name=aug_name)
  return

pluto.reserved_control = [['wisdom', 'sagacity', 'intelligence', 'prudence'],
  ['foolishness', 'folly', 'idiocy', 'stupidity']]

pluto.reserved_netflix = [['family','household', 'brood', 'unit', 'families'],
  ['life','existance', 'entity', 'creation'],
  ['love', 'warmth', 'endearment','tenderness']]
pluto.reserved_netflix = pluto.reserved_control + pluto.reserved_netflix

pluto.reserved_twitter = [['user', 'users', 'customer', 'client','people','member','shopper'],
  ['happy', 'cheerful', 'joyful', 'carefree'],
  ['time','clock','hour']]
pluto.reserved_twitter = pluto.reserved_control + pluto.reserved_twitter