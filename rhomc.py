# rhomc.py - A quick library of scripts used to determine error correlation coefficients
# using analytical and Monte Carlo approaches.
#
# Copyright (C) 2018, Cameron M. Mercer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# A quick library of scripts used to determine error correlation coefficients
# using analytical and Monte Carlo approaches. This is designed to be used with
# the isochron.py function library.
#

import matplotlib as mpl       #Import matplotlib alone to reset the graphics renderer.
import matplotlib.pyplot as p  #Import plotting tools.
import numpy as np             #Import math functions.
import numpy.random as r
import pandas as pd
import sys, os, csv

# process function.
def process(path,delim='\t',idx_col=0,val_idx=[0,2,4],unc_idx=[1,3,5],popsize=2**16):
  '''
  Quick function to compute rho values using data from the specified file. This function saves
  two output files with the same base name as the input file and the extensions '*_rho.txt'
  and '*rho_mc.txt'.

  Arguments:
  - path    - The path to the datafile.
  - delim   - The delimiter to use when reading the file; default = tab.
  - idx_col - The column to make the DataFrame index; default = 0.
  - popsize - The size of Monte Carlo synthetic populations; default = 2**16.
  '''
  # Load and extract data; preserve additional columns by set operations.
  ds = load_data(path,delim,idx_col)
  comp = extract_abc(ds,val_idx,unc_idx)
  ancillary = list(set(range(len(ds.columns))).difference(set(np.concatenate((val_idx,unc_idx)))))
  ancillary.sort()
  # Generate output file names.
  outdir, fname = os.path.split(path)
  out = os.path.join(outdir,os.path.splitext(fname)[0] + '_rho.txt')
  out_mc = os.path.join(outdir,os.path.splitext(fname)[0] + '_rho_mc.txt')
  # Define convenience variables, compute x, sx, y, sy (absolute uncerts)
  n = np.shape(comp['a'][0])[0]
  a = comp['a']; b = comp['b']; c = comp['c']
  x = np.array(a[0]/c[0])
  sx = np.array(x*np.sqrt(a[1]**2/a[0]**2 + c[1]**2/c[0]**2))
  y = np.array(b[0]/c[0])
  sy = np.array(y*np.sqrt(b[1]**2/b[0]**2 + c[1]**2/c[0]**2))
  xlbl = 'x ({:}/{:})'.format(ds.columns[val_idx[0]],ds.columns[val_idx[2]])
  sxlbl = 'sx ({:}/{:})'.format(ds.columns[val_idx[0]],ds.columns[val_idx[2]])
  ylbl = 'y ({:}/{:})'.format(ds.columns[val_idx[1]],ds.columns[val_idx[2]])
  sylbl = 'sy ({:}/{:})'.format(ds.columns[val_idx[1]],ds.columns[val_idx[2]])
  # Compute rho analytically and save file.
  rho_vals = np.zeros(n,)
  rho_vals_mc = np.zeros(n,)
  for i in range(n):
    rho_vals[i] = rho(a[0][i],a[1][i],b[0][i],b[1][i],c[0][i],c[1][i])
  analytical = pd.DataFrame(data={xlbl:x,sxlbl:sx,ylbl:y,sylbl:sy,'rho':rho_vals},index=ds.index)
  analytical = pd.concat([analytical,ds.iloc[:,ancillary]],axis='columns',sort=False)
  analytical.to_csv(out,'\t')
  # Compute rho by Monte Carlo and save file.
  for i in range(n):
    rho_vals_mc[i] = rho_mc(a[0][i],a[1][i],b[0][i],b[1][i],c[0][i],c[1][i],popsize)
    show_progress('Computing rho values by MC',i,n)
  montecarlo = pd.DataFrame({xlbl:x,sxlbl:sx,ylbl:y,sylbl:sy,'rho':rho_vals},index=ds.index)
  montecarlo = pd.concat([montecarlo,ds.iloc[:,ancillary]],axis='columns',sort=False)
  montecarlo.to_csv(out_mc,'\t')

# load_data function.
def load_data(path,delim='\t',idx_col=0):
  '''
  Reads the data from the specified file.

  Arguments:
  - path    - The path to the datafile.
  - delim   - The delimeter to use when reading the file; default = tab.
  - idx_col - The column to make the DataFrame index; default = 0.

  Returns: the dataset, as a Pandas DataFrame.
  '''
  ds = pd.read_csv(path,delim,index_col=idx_col)
  return ds

# extract_abc function.
def extract_abc(ds,val_idx=[0,2,4],unc_idx=[1,3,5]):
  '''
  Extracts values and their uncertainties in the order a, b, c according to the
  indexes specified. By default, absolute 1SD uncertainties are assumed to be
  located immediately to the right of their corresponding values. That is,
  if [a,b,c] = [0,2,4], then [sa,sb,sc] = [1,3,5].

  Arguments:
  - ds      - The DataFrame from which to extract a subset of data.
  - val_idx - The indexes specifying [a,b,c].
  - unc_idx - The indexes specifying [sa,sb,sc].

  Returns: a dict with keys ['a','b','c']; each key maps to a 2D array with
  values in [:,0] and uncertainties in [:,1].
  '''
  a = ds.iloc[:,val_idx[0]].values; sa = ds.iloc[:,unc_idx[0]].values
  b = ds.iloc[:,val_idx[1]].values; sb = ds.iloc[:,unc_idx[1]].values
  c = ds.iloc[:,val_idx[2]].values; sc = ds.iloc[:,unc_idx[2]].values
  return {'a':[a,sa],'b':[b,sb],'c':[c,sc]}

# rho_mc function.
def rho_mc(a,sa,b,sb,c,sc,popsize=2**16,show_plot=False):
  '''
  Computes the error correlation coefficient using a Monte Carlo approach assuming:
  
  x = a/c    and    y = b/c

  This method generates synthetic populations for x and y, then determines the linear
  correlation coefficient:

                     Sum[(xi - X)*(yi - Y)]
  rho(x,y) = -----------------------------------------
              sqrt[Sum[(xi - X)**2]*Sum[(yi - Y)**2]]

  where xi, yi are the synthetic coordinates, and X, Y are the arithmetic means.
  
  Arguments:
  - a, sa   - values and 1SD absolute uncertainties in component a
  - b, sb   - values and 1SD absolute uncertainties in component b
  - c, sc   - values and 1SD absolute uncertainties in component c
  - popsize - synthetic population size; default = 2**16.
  - show_plot - used for debugging.
  '''
  # Create source populations.
  bag_a = r.normal(a,sa,(popsize))
  bag_b = r.normal(b,sb,(popsize))
  bag_c = r.normal(c,sc,(popsize))
  # Create populations of ratios.
  x, y = np.zeros(popsize), np.zeros(popsize)
  for i in range(popsize):
    idx = r.randint(0,popsize-1)
    x[i] = bag_a[idx]/bag_c[idx]
    y[i] = bag_b[idx]/bag_c[idx]
  # Compute correlation coefficient.
  dev_x = x - np.mean(x)
  dev_y = y - np.mean(y)
  rho = np.sum(dev_x*dev_y)/np.sqrt(np.sum(dev_x**2)*np.sum(dev_y**2))
  # Show plot for debugging if needed.
  if show_plot:
    p.plot(x,y,c='#666666',linestyle='none',marker='.',markersize=1,alpha=0.2)
  return rho
  
# rho function.
def rho(a,sa,b,sb,c,sc):
  '''
  Computes the error correlation coefficient analytically assuming components
  [a,b,c] are combined in this way (and that errors in a, b, c, are independent):
  
  x = a/c    and    y = b/c

  WLOG: S_x = sx/x

  rho = S_c**2/(S_x*S_y)    where (WLOG)    S_x**2 = S_a**2 + S_c**2

  See the Isoplot manual for a derivation.

  Arguments:
  - a, sa - values and 1SD absolute uncertainties in component a
  - b, sb - values and 1SD absolute uncertainties in component b
  - c, sc - values and 1SD absolute uncertainties in component c
  '''
  # Compute x, y, and relative uncertainties.
  x = a/c; sx = np.sqrt(sa**2/a**2 + sc**2/c**2)
  y = b/c; sy = np.sqrt(sb**2/b**2 + sc**2/c**2)
  # Return value of rho.
  return (sc**2/c**2)/(sx*sy)

# -------------------------------------------------------------------------------- #
# Utility methods.
# -------------------------------------------------------------------------------- #
# show_progress method.
def show_progress(base,step,n,u=100,bar_length=50):
  """
  Displays a progress bar in the command window.

  Usage: show_progress(base_string,step,n,u,bar_length=50)

  Arguments:
  - base_string - a string to show at the left side of the progress bar.
  - step        - integer index in the loop whose progress is being shown.
  - n           - total number of steps that will execute in the loop.
  - u           - the number of times to update the progress bar.
  - bar_length  - specifies how many characters will be displayed in the progress bar.
  """
  if step < n - 1:
    if np.floor(step % ((n - 1)/u + 1)) != 0:
      return
    frac = float(step)/n
  else:
    frac = 1
  hashes = '#'*int(round(frac*bar_length))
  spaces = ' '*(bar_length - len(hashes))
  if step < n - 1:
    sys.stdout.write('\r{:} [{:}] {:6.2f}%'.format(base,hashes+spaces,frac*100.0));
  else:
    sys.stdout.write('\r{:} [{:}] {:6.2f}%\n'.format(base,hashes+spaces,frac*100.0));
  sys.stdout.flush()

