# isochron.py - A Python library to plot a nice isochron with a York regression.
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
# PURPOSE:
#   Python library to plot a nice isochron with a York regression. More flexible and
#   interactive than original plotIsochron script. Does not compute 40Ar/39Ar ages (yet).
#
# AUTHOR:
#   Cameron M. Mercer
#   School of Earth and Space Exploration
#   Arizona State University, Tempe, AZ 85287
# 

import matplotlib as mpl       #Import matplotlib alone to reset the graphics renderer.
import matplotlib.pyplot as p  #Import plotting tools.
import numpy as np             #Import math functions.
import numpy.random as r
import pandas as pd
import sys, csv
import scipy.stats as stats
import pint
from matplotlib.patches import Ellipse
# Make text labels text boxes rather than character paths.
mpl.rcParams['pdf.fonttype'] = 'truetype'
# mpl.rc('font',family='serif',serif='Times New Roman') #Times New Roman
mpl.rc('font',family='sans-serif',serif='Arial')
# Make plot frames and ticks thicker.
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1

# 40K decay constants.
ur = pint.UnitRegistry()
ur.load_definitions('geochron_units')
kdc = {'SJ77':{'Ref':'Steiger and Jager, 1977',
               'L_eps':ur('0.581e-10 1/a'),
               'L_eps 1SD':ur('0.00581e-10 1/a'),
               'L_beta':ur('4.962e-10 1/a'),
               'L_beta 1SD':ur('0.04962e-10 1/a'),
               'Notes':'Uncertainties assumed to be 1%.'},
       'Ren11':{'Ref':'Renne et al., 2011',
                'L_eps':ur('0.5757e-10 1/a'),
                'L_eps 1SD':ur('0.0016e-10 1/a'),
                'L_beta':ur('4.9548e-10 1/a'),
                'L_beta':ur('0.0134e-10 1/a'),
                'Notes':'Beware covariance with U decay constants. Apparently low-balled estimates of zircon residence times.'}}


# process function.
def process(path,delim='\t',idx_col=0,indexes=[0,1,2,3,4],selcol=None,selval=None,exclude=None,run_oli=True,
            xlimits=None,ylimits=None,cutoff=4.0):
  '''
  Reads the data from the specified file, runs a York regression, and plots the data. This is an 
  example of how to chain functions from the library together, and not a fully optioned replacement
  for using the library functions in homebrewed scripts or the command line.

  Arguments:
  - path    - The path to the datafile.
  - delim   - The delimeter to use when reading the file; default = tab.
  - idx_col - The column to make the DataFrame index; default = 0.
  - indexes - The default selection indexes, in the order: [x,sx,y,sy,rho]; default => [0,1,2,3,4].
  - selcol  - The name of a column that will be used to select data records.
  - selval  - A list of values that qualify a data record to be selected.
  - exclude - A list of items to exclude; these items must be values from the index
              column of the specified Pandas DataFrame.
  - xlimits, ylimits - plot domain and range, applied before regression is plotted. Must be two-element lists.
  - cutoff  - Hampel cutoff.

  Returns: the outliers, or None.
  '''
  ds = loadDataset(path,delim,idx_col)
  dat = extractData(ds,indexes,selcol,selval,exclude)
  res = yorkRegression(dat)
  print('')
  print_stats(dat,res)
  print('')
  (ax,zid,fig) = initPlot()
  zid = plotData(ax,dat,zid)
  if xlimits is not None:
    p.xlim(xlimits)
  if ylimits is not None:
    p.ylim(ylimits)
  zid = plotRegression(ax,res[0],res[2],zid)
  oli = None
  if run_oli:
    oli = weighted_oli_2d(dat,res)
    if oli is not None and len(oli) > 0:
      print('\nOutliers detected!')
      print(oli)
    else:
      print('\nNo outliers detected, huzzah!')
  return oli

# loadDataset function.
def loadDataset(path,delim='\t',idx_col=0):
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

# extractData function.
def extractData(ds,indexes=[0,1,2,3,4],selcol=None,selval=None,exclude=None):
  '''
  Extracts x, sx, y, sy, and rho,in that order, from the specified dataset.

  Arguments:
  - ds      - The DataFrame from which to extract a subset of data.
  - indexes - The default selection indexes, in the order: [x,sx,y,sy,rho]; default => [0,1,2,3,4].
  - selcol  - The name of a column that will be used to select data records.
  - selval  - A list of values that qualify a data record to be selected.
  - exclude - A list of items to exclude; these items must be values from the index
              column of the specified Pandas DataFrame.
  
  Returns a 2D matrix containing [x,sx,y,sy,rho].
  '''
  # Prepare structures.
  x, sx, y, sy, rho = [], [], [], [], []
  # Prepare extraction indexes.
  # Start with records meeting selection criteria.
  idx = []
  if selcol is None:
    idx = list(range(len(ds)))
  else:
    sci = list(ds.columns).index(selcol)
    if not isinstance(selval,list):
      selval = [selval]
    for i in range(len(ds)):
      if ds.iloc[i,sci] in selval:
        idx.append(i)
  # Make sure specified records are excluded.
  if exclude is not None:
    pidx = idx
    idx = []
    if not isinstance(exclude,list):
      exclude = [exclude]
    for i in range(len(pidx)):
      if ds.index[pidx[i]] not in exclude:
        idx.append(pidx[i])
  # Extract data.
  x = ds.iloc[idx,indexes[0]].values
  sx = ds.iloc[idx,indexes[1]].values
  y = ds.iloc[idx,indexes[2]].values
  sy = ds.iloc[idx,indexes[3]].values
  rho = ds.iloc[idx,indexes[4]].values
  ids = list(ds.index[idx])
  labels = [ds.columns[indexes[0]],ds.columns[indexes[1]]]
  return [x,sx,y,sy,rho,ids,labels]

# Returns: (ax, zid, fig)
def initPlot():
  '''
  Initializes a plot window.
  Returns: ax, zid, fig
  '''
  # Prepare to plot the data.
  fig = p.figure(1)
  ax = p.gca()
  zid = 1
  return (ax, zid, fig)

# plotData function. Returns zid.
def plotData(ax,data,zid,scale_unc=2.0,fc='#999999',ec='#666666'):
  '''
  Plots error ellipses for the specified data into the specified axes.
  Returns: zid
  '''
  # Unpack plot data.
  (x,sx,y,sy,rho) = unpackData(data,5)
  # Scale uncertainties.
  sx = scale_unc*sx
  sy = scale_unc*sy
  # Calculate the covariance.
  sxy = sx*sy*rho
  # Prepare to calculate ellipse parameters.
  dims = np.zeros((len(x),2),dtype='d') #dims[0] = x-dimension, dims[1] = y-dimension
  phi = np.zeros((len(x),),dtype='d')
  aspect = np.zeros((len(x),),dtype='d')
  for i in range(len(sxy)):
    # Make covariance matrix and calculate ellipse dimensions.
    covmat = np.array([[np.power(sx[i],2),sxy[i]],[sxy[i],np.power(sy[i],2)]])
    vals, vects = np.linalg.eigh(covmat)
    if sx[i] > sy[i]:
      dims[i,0] = np.sqrt(max(vals))
      dims[i,1] = np.sqrt(min(vals))
    else:
      dims[i,0] = np.sqrt(min(vals))
      dims[i,1] = np.sqrt(max(vals))
    phi = np.degrees(np.arctan(2.0*sxy/(np.power(sx,2)-np.power(sy,2)))/2.0)
  # Plot the data.
  zid = plot_errEllipses(ax,x,sx,y,sy,rho,dims,phi,zid,fc=fc,ec=ec)
  p.show()
  return zid

def unpackData(data,n):
  '''
  Unpacks the 2D matrix of data returned by the extractData function.

  Arguments:
  - data - The matrix of data from which to unpack objects.
  - n    - The number of objects to unpack, from left to right.

  Returns: a tuple with n objects.
  '''
  if n > len(data)+1:
    print('Error: n is greater than len(data).')
    return
  return tuple(data[0:n])

# Define convenience functions for plotting regular error ellipses.
def plot_errEllipses(ax,x,sx,y,sy,rho,dims,phi,zid,fc='#999999',ec='#666666'):
  '''
  Called by plotData function.
  Returns: zid
  '''
  # Plot data.
  for i in range(len(x)):
    ellFace = Ellipse(xy=(x[i],y[i]),width=2*dims[i,0],height=2*dims[i,1],angle=phi[i],\
                      facecolor=fc,edgecolor='none',alpha=0.8,zorder=zid)
    zid += 1
    ellEdge = Ellipse(xy=(x[i],y[i]),width=2*dims[i,0],height=2*dims[i,1],\
                      angle=phi[i],facecolor='none',edgecolor=ec,zorder=zid)
    zid += 1
    ax.add_artist(ellFace)
    ax.add_artist(ellEdge)
    #Add point to center of ellipse.
    ax.plot(x[i],y[i],marker='o',color=ec,mfc=ec,mec=ec,ms=3,zorder=zid)
    zid += 1
  return zid

def plotRegression(ax,a,b,zid,color='k',linestyle='-'):
  '''
  Used to plot a York regression (or other line).

  Returns: zid
  '''
  xl = ax.get_xlim()
  yv = [a + b*xl[0],a + b*xl[1]]
  ax.plot(xl,yv,c=color,ls=linestyle,zorder=zid)
  zid += 1
  p.show()
  return zid

def plotEnvelope(ax,a,sa,b,sb,zid,level=2,color='#999999'):
  '''
  Used to plot error envelopes based on a York regression.

  Arguments:
  - ax    - the axes to plot in
  - a, sa - intercept and uncertainty on the intercept
  - b, sb - slope and uncertainty on the slope
  - zid   - zorder integer
  - level - confidence level for error bounds (multiples of sigma)
  - color - line color for envelope lines

  Returns: zid
  '''
  # Get domain limits.
  xl = ax.get_xlim()
  # Generate synthetic populations.
  msize = 2**10
  psize = 2**12
  ra = r.normal(a,sa,(psize))
  rb = r.normal(b,sb,(psize))
  # Generate n-sigma values along mesh (based on confidence level)
  mx = np.linspace(0,xl[1],msize)
  m2s = np.zeros(msize)
  for i in range(msize):
    m2s[i] = level*np.std(ra + rb*mx[i])
  # Generate upper and lower envelope bounds.
  up = a + b*mx + m2s
  lo = a + b*mx - m2s
  # Plot the envelope.
  ax.plot(mx,up,c=color,zorder=zid)
  zid += 1
  ax.plot(mx,lo,c=color,zorder=zid)
  zid += 1
  p.show()
  return zid

# --------------------------------------------------------------------------------
# York regression functions
# Returns: (a,sa,b,sb)
def yorkRegression(data):
  # Unpack data.
  (x,sx,y,sy,rho) = unpackData(data,5)
  # Weights.
  wX, wY = 1/sx**2, 1/sy**2
  # Guess initial slope with simple least-squares regression.
  xmean, ymean = np.mean(x), np.mean(y)
  sumXY = sum((x-xmean)*(y-ymean))
  sumXX = sum((x-xmean)**2)
  b = sumXY/sumXX
  if abs(b - 1.0) < 1e-8 or abs(b - 0.0) < 1e-8:
    b = 10.0
  # Iteratively solve for best fit.
  (b,capW,capX,capY,betaI,icount) = iterativeSolver(x,y,wX,wY,rho,b)
  # Calculate the intercept.
  a = capY - b*capX
  # Calculate the weighted mean of the LSE adjusted x-coordinates, xi.
  xi = np.zeros(len(x))
  xwm = 0.0
  for i in range(len(x)):
    xi[i] = capX + betaI[i]
    xwm = xwm + capW[i]*xi[i]
  xwm = xwm/sum(capW)
  # Determine standard errors for the intercept and slope.
  sb = 0.0
  for i in range(len(x)):
    sb = sb + capW[i]*(xi[i] - xwm)**2
  sb = 1.0/sb
  sa = 1.0/sum(capW) + sb*xwm**2
  # Take square root of sa and sb (they're variances right now).
  sa = np.sqrt(sa)
  sb = np.sqrt(sb)
  # Return results.
  return (a,sa,b,sb)

def iterativeSolver(x,y,wX,wY,rho,b):
  # Iteratively solve for the best fit line.
  eps = 1e-12
  maxIter = 1024
  icount = 0
  capW, betaI = np.zeros(len(x)), np.zeros(len(x))
  sumCapWX, sumCapWY, sumCapW, capUi, capVi, sumWBV, sumWBU = [], [], [], [], [], [], []
  capX, capY, oldB, newB = 0.0, 0.0, 0.0, b
  while abs(newB - oldB) > eps:
    # Iterate counter, store b.
    icount = icount + 1
    oldB = newB
    # Check iteration limit.
    if icount > maxIter:
      print('York regression failed to converge.')
      return
    # Initialize sum values.
    sumCapWX, sumCapWY, sumCapW, sumWBV, sumWBU = 0.0, 0.0, 0.0, 0.0, 0.0
    # Determine capW, capX, and capY.
    for i in range(len(x)):
      capW[i] = (wX[i]*wY[i])/(wX[i] + (oldB**2)*wY[i] - 2.0*oldB*rho[i]*np.sqrt(wX[i]*wY[i]))
      sumCapWX = sumCapWX + capW[i]*x[i]
      sumCapWY = sumCapWY + capW[i]*y[i]
      sumCapW = sumCapW + capW[i]
    capX = sumCapWX/sumCapW
    capY = sumCapWY/sumCapW
    # Determine capU, capV, and betaI, and a new estimate for the slope.
    for i in range(len(x)):
      capUi = x[i] - capX
      capVi = y[i] - capY
      betaI[i] = capW[i]*(capUi/wY[i] + (oldB*capVi)/wX[i] -
                          (oldB*capUi + capVi)*rho[i]/np.sqrt(wX[i]*wY[i]))
      sumWBV = sumWBV + capW[i]*betaI[i]*capVi
      sumWBU = sumWBU + capW[i]*betaI[i]*capUi
    newB = sumWBV/sumWBU
  # Return results.
  return (newB, capW, capX, capY, betaI, icount)

# --------------------------------------------------------------------------------
# Descriptive statistics functions.

def print_stats(data,reg_results,conf=0.95,mswd_conf=0.95):
  '''
  Displays the specified regression results with some descriptive statistics. Note, if the
  value of the MSWD is higher than the upper confidence bound, confidence boundary uncertainties
  for the slope and intercept will be expanded by sqrt(MSWD) (as well as multiplied by a
  Student's t multiplier to achieve the specified confidence level).

  Arguments:
  - data        - The data being described.
  - reg_results - The regression results for the data.
  - conf        - Confidence level at which to display regression results. Default = 95%.
  - mswd_conf   - Confidence level for mswd acceptable bounds. Default = 95%.
  '''
  # Compute statistics.
  (x,sx,y,sy,rho) = unpackData(data,5)
  n, half_tail = len(x), (1.0 - conf)/2.0
  tcrit = stats.t.ppf(1.0 - half_tail,n-2)
  a,sa,b,sb = reg_results[0], reg_results[1], reg_results[2], reg_results[3]
  (mswd,smswd,mswd_ci) = mswd_2d(x,sx,y,sy,rho,a,b,mswd_conf)
  ci_a, ci_b = tcrit*sa, tcrit*sb
  expanded = ''
  if mswd > mswd_ci[1]:
    expanded = ' Exp.'
    ci_a, ci_b = ci_a*np.sqrt(mswd), ci_b*np.sqrt(mswd)
  r2 = calc_r2(x,y)
  # Prepare confidence level labels.
  ci_int = 'sa ({:.0f}%{:})'.format(conf*100,expanded)
  ci_slope = 'sb ({:.0f}%{:})'.format(conf*100,expanded)
  ci_mswd_lo = 'mswd lo {:.0f}%'.format(mswd_conf*100)
  ci_mswd_hi = 'mswd hi {:.0f}%'.format(mswd_conf*100)
  # Display results.
  print('\nLinear Regression Results (y = a + bx):\n')
  print('{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}'.format('a','1SD a',ci_int,'b','1SD b',ci_slope))
  print('{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n'.format(a,sa,ci_a,b,sb,ci_b))
  print('{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}'.format('mswd','1SD mswd',ci_mswd_lo,ci_mswd_hi,'r2','N'))
  print('{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:10d}\n'.format(mswd,smswd,mswd_ci[0],mswd_ci[1],r2,n))

def calc_r2(x,y):
  '''
  Returns the square of sample correlation coefficient, i.e., r**2.
  '''
  if len(x) != len(y):
    print('Error: input arrays must be the same length')
    return
  sxx, syy, sxy = 0.0, 0.0, 0.0
  xm, ym = np.mean(x), np.mean(y)
  for i in range(len(x)):
    sxx += (x[i] - xm)**2
    syy += (y[i] - ym)**2
    sxy += (x[i] - xm)*(y[i] - ym)
  return (sxy/np.sqrt(sxx*syy))**2

def mswd_2d(x,sx,y,sy,rho,a,b,conf=.95):
  '''
  Determines the MSWD for a line with intercept a and slope b that was fit to the
  specified data (x,sx,y,sy).

  Arguments:
  - x, sx - Data and uncertainties on the abscissa.
  - y, sy - Data and uncertainties on the ordinate.
  - rho   - Error correlation coefficients for data points.
  - a, b  - Intercept and slope of the best fit line, y = a + bx.
  - conf  - Confidence level (expressed on the interval (0,1)) to return for the MSWD.
            This method passes conf to the mswd_2d_conf method.

  Returns: (mswd, smswd, conf_mswd)
  Note: conf_mswd is a two element numpy array with the lower [0] and upper [1] bounds
  of the 2D mswd, as determined using the mswd_2d_conf method.
  '''
  # Check inputs.
  if len(x) != len(sx) or len(x) != len(y) or len(x) != len(sy) or len(x) != len(rho):
    print('Error: input arrays must be the same length')
    return
  # Calculate MSWD.
  n = len(x)
  mswd = 0
  terms = np.zeros(n)
  for i in range(n):
    terms[i] = (y[i] - a - b*x[i])**2/(sy[i]**2 + (b*sx[i])**2 - 2*b*rho[i]*sx[i]*sy[i])
  mswd = sum(terms)/(n-2)
  ci = mswd_2d_conf(n-2,conf)
  return (mswd,np.sqrt(2.0/(n-2)),ci)

def mswd_2d_conf(dof,conf=0.95):
  '''
  Determines the upper and lower confidence bounds for the 2D MSWD.

  Arguments:
  - dof  - Number of degrees of freedom.
  - conf - Confidence level (expressed on the interval (0,1)) to return for the MSWD.

  Returns: [lower_bound,upper_bound] as a numpy array.
  '''
  half_tail = (1.0 - conf)/2.0
  return np.array([stats.chi2.ppf(half_tail,dof),stats.chi2.ppf(1.0-half_tail,dof)])/float(dof)

# --------------------------------------------------------------------------------
# Outlier identifier(s) and related methods.

# weighted_oli_2d convenience method.
def weighted_oli_2d(data,york_res,cutoff=4.0):
  '''
  Applies the Hampel outlier identifier to the weighted residuals (chi**2 values)
  of the data from the specified York regression results.
  '''
  (capW, capS) = chi2_2d(data,york_res)
  idx = hampel(capS,cutoff)
  oli = []
  for i in idx:
    oli.append(data[5][i])
  return oli

# Determine chi squared values for regression results.
# chi2_2d
def chi2_2d(data,york_res):
  '''
  Determines the weighted residuals (chi**2 values) of the data from the
  specified York regression results.
  '''
  # Unpack data.
  (x,sx,y,sy,rho) = unpackData(data,5)
  (a,sa,b,sb) = unpackData(york_res,4)
  n = len(x)
  # Get weights for data, and model weights with error correlations.
  wX, wY = 1.0/sx**2, 1.0/sy**2
  capW = np.zeros(n)
  capS = np.zeros(n)
  for i in range(n):
    capW[i] = (wX[i]*wY[i])/(wX[i] + (b**2)*wY[i] - 2.0*b*rho[i]*np.sqrt(wX[i]*wY[i]))
    capS[i] = capW[i]*(y[i] - b*x[i] - a)**2
  return capW, capS

# Hampel.
def hampel(values,cutoff=4.0):
  '''
  Applies the Hampel outlier identifier to the specified set of values.

  Returns: a list of indexes of potential outliers, or an empty list if none were identified.
  '''
  # Calculate absolute deviations from the median, and scaled MADM value.
  med = np.median(values)
  viMed = abs(values - med)
  s = 1.4826*np.median(viMed)
  # Identify potential outliers.
  oli = viMed/s
  idx = []
  for i in range(len(oli)):
    if oli[i] > cutoff:
      idx.append(i)
  return idx

# --------------------------------------------------------------------------------
# Age calculation function(s).

def compute_date(data,reg_results,J,sJ,conf=0.95,mswd_conf=0.95,show_stats=False,
                 isochron_type='Normal',kdc_ref='SJ77',units='Ma'):
  '''
  Computes a 40Ar/39Ar date from York regression results using the specified J value and
  decay constants. This method can optionally display descriptive statistics for the
  regression. If the value of the MSWD is higher than the upper confidence bound, the age
  uncertainties will be expanded with a Student's t multiplier and sqrt(MSWD). Three levels
  of error are reported for the date: analytical, internal (analytical + J uncertainty),
  and external (internal uncertainty + decay constant uncertainty).

  Arguments:
  - data          - The extracted data set. Only quantity labels are used from this structure.
  - reg_results   - The regression results for the data.
  - J, sJ         - The J value and its 1SD absolute uncertainty.
  - conf          - Confidence level at which to display regression results. Default = 95%.
  - mswd_conf     - Confidence level for mswd acceptable bounds. Default = 95%.
  - show_stats    - Boolean indicating whether or not to show descriptive statistics for the
                    regression. Default = False. The computed date will always be shown.
  - isochron_type - Type of isochron; either 'Normal' or 'Inverse'. If Normal, this method uses
                    the slope (40Ar/39ArK) to compute the age; if Inverse, this method uses the
                    intercept (39ArK/40Ar) to compute the age (assuming you've regressed a plot
                    of 39ArK/40Ar versus 36Ar/40Ar). Default = Normal.
  - kdc_ref       - Key for retrieving decay constants from the kdc dict. Default = SJ77.
  - units         - Target units for the computed date.
  '''
  # Dummy check.
  if isochron_type not in ['Normal','Inverse']:
    print('Error: isochron_type must be either "Normal" or "Inverse"')
    return
  # Compute descriptive statistics.
  (x,sx,y,sy,rho) = unpackData(data,5)
  n, half_tail = len(x), (1.0 - conf)/2.0
  tcrit = stats.t.ppf(1.0 - half_tail,n-2)
  a,sa,b,sb = reg_results[0], reg_results[1], reg_results[2], reg_results[3]
  (mswd,smswd,mswd_ci) = mswd_2d(x,sx,y,sy,rho,a,b,mswd_conf)
  ci_a, ci_b = tcrit*sa, tcrit*sb
  expanded = ''
  if mswd > mswd_ci[1]:
    expanded = ' Exp.'
    ci_a, ci_b = ci_a*np.sqrt(mswd), ci_b*np.sqrt(mswd)
  r2 = calc_r2(x,y)
  # Compute total decay constant and F value (40Ar*/39ArK).
  L_eps = kdc[kdc_ref]['L_eps'].to('1/'+units)
  L_eps_sd = kdc[kdc_ref]['L_eps 1SD'].to('1/'+units)
  L_beta = kdc[kdc_ref]['L_beta'].to('1/'+units)
  L_beta_sd = kdc[kdc_ref]['L_beta 1SD'].to('1/'+units)
  L_total = L_eps + L_beta
  L_total_sd = np.sqrt(L_eps_sd**2 + L_beta_sd**2)
  F, sF = reg_results[2], reg_results[3]
  if isochron_type == 'Inverse':
    F, sF = reg_results[0], reg_results[1]
  # Compute partial derivatives for error propagation in age equation.
  dt_dF = 1.0/L_total.magnitude*J/(J*F + 1.0)
  dt_dJ = 1.0/L_total.magnitude*F/(J*F + 1.0)
  dt_dL = -1.0/L_total.magnitude**2*np.log(J*F + 1.0)
  # Compute age and uncertainties.
  t = 1.0/L_total*np.log(J*F + 1.0)
  st = dt_dF*sF
  s_analytical = tcrit*st
  s_internal = tcrit*np.sqrt(dt_dF**2*sF**2 + dt_dJ**2*sJ**2)
  s_external = tcrit*np.sqrt(dt_dF**2*sF**2 + dt_dJ**2*sJ**2 + dt_dL**2*L_total_sd.magnitude**2)
  # Expand results if needed.
  if mswd > mswd_ci[1]:
    s_analytical *= np.sqrt(mswd)
    s_internal *= np.sqrt(mswd)
    s_external *= np.sqrt(mswd)
  # Print results.
  if show_stats:
    print_stats(data,reg_results,conf,mswd_conf)
  print('{:>16s}\t{:>16s}\t{:>16s}\t{:>16s}\t{:>16s}'.format('Age ({:})'.format(units),'1SD ({:})'.format(units),
                                                             'CI {:.0f}% ({:}){:}'.format(conf*100,units,expanded),
                                                             'CI {:.0f}% Int ({:}){:}'.format(conf*100,units,expanded),
                                                             'CI {:.0f}% Ext ({:}){:}'.format(conf*100,units,expanded)))
  print('{:16.8e}\t{:16.8e}\t{:16.8e}\t{:16.8e}\t{:16.8e}\n'.format(t.magnitude,st,s_analytical,s_internal,s_external))
