
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

pluto.version = 2.0
import opendatasets
#
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

@add_method(PacktDataAug)
def fetch_df(self, csv,sep=','):
  df = pandas.read_csv(csv, encoding='latin-1', sep=sep)
  return df
#
@add_method(PacktDataAug)
def _fetch_larger_font(self):
  heading_properties = [('font-size', '20px')]
  cell_properties = [('font-size', '18px')]
  dfstyle = [dict(selector="th", props=heading_properties),
    dict(selector="td", props=cell_properties)]
  return dfstyle

@add_method(PacktDataAug)
def build_sf_fname(self, df):
  root = 'state-farm-distracted-driver-detection/imgs/train/'
  df["fname"] = root + df.classname+'/'+df.img
  return

# set internal counter for image to be zero, e.g. pluto0.jpg, pluto1.jpg, etc.
pluto.fname_id = 0
#
@add_method(PacktDataAug)
def _drop_image(self,canvas, fname=None,format=".jpg",dname="Data-Augmentation-with-Python/pluto_img"):
  if (fname is None):
    self.fname_id += 1
    if not os.path.exists(dname):
      os.makedirs(dname)
    fn = f'{dname}/pluto{self.fname_id}{format}'
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
def print_batch_text(self,df_orig, disp_max=6, cols=["title", "description"],is_larger_font=True): 
  df = df_orig[cols] 
  with pandas.option_context("display.max_colwidth", None):
    if (is_larger_font):
      display(df.sample(disp_max).style.set_table_styles(self._fetch_larger_font()))
    else:
      display(df.sample(disp_max))
  return

@add_method(PacktDataAug)
def count_word(self, df, col_dest="description"):
  df['wordc'] = df[col_dest].apply(lambda x: len(x.split()))
  return

@add_method(PacktDataAug)
def draw_word_count(self,df, wc='wordc',is_stack_verticle=True):
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

import re
import spellchecker
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

pluto.version = 7.0
# augment full path
@add_method(PacktDataAug)
def _append_music_full_path(self,x):
  y = re.findall('([a-zA-Z ]*)\d*.*', x)[0]
  return (f'kaggle/musical-emotions-classification/Audio_Files/Audio_Files/Train/{y}/{x}')
#
@add_method(PacktDataAug)
def fetch_music_full_path(self, df):
  df['fname'] = df.ImageID.apply(self._append_music_full_path)
  return df

import IPython
import IPython.display
import librosa
import librosa.display
import matplotlib
import pandas
import numpy
import re

@add_method(PacktDataAug)
def _draw_area_with_neg(self,ndata,pic,tcolor='#007bff',bcolor='#ffc107',alpha=0.75,istart=0,izero=0):
  nlen = len(ndata)
  i = numpy.arange(istart, istart+nlen)
  xzero = numpy.zeros(nlen)
  xzero += izero
  # plot line
  pic.plot(i,ndata, linewidth=0.0)  # invisible line for shading
  pic.plot(i,xzero,color='gray',linewidth=0.5) # base line

  # fill top/positive section
  pic.fill_between(
    i, xzero, ndata, where=(ndata >= xzero), 
    interpolate=True, color=tcolor, alpha=alpha, 
    label="Positive"
  )
  # fill bottom/negative section
  pic.fill_between(
    i, xzero, ndata, where=(ndata < xzero), 
    interpolate=True, color=bcolor, alpha=alpha, 
    label="Negative"
  )
  return
#
@add_method(PacktDataAug)
def _fetch_audio_data(self,lname):
  # select random record
  # samp = df.sample(xsize)
  # samp.reset_index(drop=True, inplace=True)
  data_amp, sam_rate = librosa.load(lname, mono=True)
  fname = re.findall('[^\/]+$', lname)
  return data_amp, sam_rate, fname[0]
#
@add_method(PacktDataAug)
def _draw_audio(self,data_amp, sam_rate, fname):
  # define constant
  zlen = 100
  nrow = 2
  ncol = 1
  w = 11
  h = 6
  title = ['', 'Zoom In: From Mid-point to 100+']
  ylabel = ['Waveform Amplitude']
  #
  # define graph
  canvas, pic = matplotlib.pyplot.subplots(nrow, ncol, figsize=(w, h))
  pics = pic.flatten()
  # draw original
  dlen = len(data_amp)
  self._draw_area_with_neg(data_amp,pic[0])
  pics[0].set_title(f'{title[0]} {fname}', fontsize=18.)
  pics[0].set_xlabel(f'Second: Total {dlen/sam_rate:.2f} sec., Sampling Rate: {sam_rate/1000:.2f} kHz', fontsize=16.)
  pics[0].set_ylabel(ylabel[0])
  #
  # overide tick with time
  loc = numpy.array(pics[0].get_xticks())
  b = loc / sam_rate
  b = numpy.round(b,1)
  pics[0].set_xticklabels(b)
  pics[0].grid()
  #
  # draw zoom
  mid = int(len(data_amp) / 2)
  end = mid + zlen
  self._draw_area_with_neg(data_amp[mid:end],pic[1])
  pics[1].set_title(title[1],fontsize=18.)
  pics[1].set_xlabel(f'Time Series Index: Mid-point at {mid/sam_rate:.2f} sec. ({mid} ts)', fontsize=16.)
  pics[1].set_ylabel(ylabel[0])
  pics[1].grid()
  # 
  # display and save image
  canvas.tight_layout()
  self._drop_image(canvas)
  canvas.show()
  # 
  return 
#
@add_method(PacktDataAug)
def draw_audio(self,df):
  samp = df.sample(1)
  samp.reset_index(drop=True, inplace=True)
  data_amp, sam_rate, fname = self._fetch_audio_data(samp.fname[0])
  self._draw_audio(data_amp, sam_rate, 'Original: ' + fname)
  # display audio 
  display(IPython.display.Audio(data_amp,rate=sam_rate))
  return
#

import audiomentations
#
@add_method(PacktDataAug)
def _fetch_1_sample(self, df, dsize=1):
  p = df.sample(dsize)
  p.reset_index(drop=True, inplace=True)
  return p.fname[0]
#
@add_method(PacktDataAug)
def _audio_transform(self, df, xtransform, title=''):
  if (type(df) is list):
    data_amp, sam_rate, fname, lname = self.audio_control_dmajor
  else:
    lname = self._fetch_1_sample(df)
    data_amp, sam_rate, fname = self._fetch_audio_data(lname)
  #
  xaug = xtransform(data_amp, sample_rate=sam_rate)
  # augmented
  self._draw_audio(xaug, sam_rate, title + ' Augmented: ' + fname)
  display(IPython.display.Audio(xaug, rate=sam_rate))
  # original
  self._draw_audio(data_amp, sam_rate, 'Original: ' + fname)
  display(IPython.display.Audio(data_amp, rate=sam_rate))
  return

@add_method(PacktDataAug)
def play_aug_time_shift(self, df, min_fraction=-0.2,max_fraction=0.8,rollover=True,title='Time Shift'):
  xtransform = audiomentations.Shift(
    min_fraction = min_fraction,
    max_fraction = max_fraction,
    rollover = rollover,
    p=1.0
  )
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_time_stretch(self, df, min_rate=0.2,max_rate=6.8,leave_length_unchanged=True,title='Time Stretch'):
  xtransform = audiomentations.TimeStretch(
    min_rate = min_rate,
    max_rate = max_rate,
    leave_length_unchanged = leave_length_unchanged,
    p=1.0
  )
  self._audio_transform(df, xtransform, title=title)
  return
  # librosa.effects.time_stretch under the hood 

@add_method(PacktDataAug)
def play_aug_pitch_scaling(self, df, min_semitones=-6.0,max_semitones=6.0,title='Pitch Scaling'):
  xtransform = audiomentations.PitchShift(
    min_semitones = min_semitones,
    max_semitones = max_semitones,
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_noise_injection(self, df, min_amplitude=0.002,max_amplitude=0.2,title='Gaussian noise injection'):
  xtransform = audiomentations.AddGaussianNoise(
    min_amplitude = min_amplitude,
    max_amplitude = max_amplitude,
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_polar_inverse(self, df, title='Polarity inversion'):
  xtransform = audiomentations.PolarityInversion(
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_low_pass_filter(self, df, 
  min_cutoff_freq=150, max_cutoff_freq=7500,
  min_rolloff=12, max_rolloff=24,
  title='Low pass filter'):
  xtransform = audiomentations.LowPassFilter(
    min_cutoff_freq = min_cutoff_freq,
    max_cutoff_freq = max_cutoff_freq,
    min_rolloff = min_rolloff,
    max_rolloff = max_rolloff,
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_high_pass_filter(self, df, 
  min_cutoff_freq=20, max_cutoff_freq=2400,
  min_rolloff=12, max_rolloff=24,
  title='High pass filter'):
  xtransform = audiomentations.HighPassFilter(
    min_cutoff_freq = min_cutoff_freq,
    max_cutoff_freq = max_cutoff_freq,
    min_rolloff = min_rolloff,
    max_rolloff = max_rolloff,
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return

@add_method(PacktDataAug)
def play_aug_band_pass_filter(self, df, 
  min_center_freq=200, max_center_freq=4000,
  min_bandwidth_fraction=0.5,max_bandwidth_fraction=1.99,
  min_rolloff=12, max_rolloff=24,
  title='Band pass filter'):
  xtransform = audiomentations.BandPassFilter(
    min_center_freq = min_center_freq,
    max_center_freq = max_center_freq,
    min_bandwidth_fraction = min_bandwidth_fraction,
    max_bandwidth_fraction = max_bandwidth_fraction,
    min_rolloff = min_rolloff,
    max_rolloff = max_rolloff,
    p=1.0)
  self._audio_transform(df, xtransform, title=title)
  return
