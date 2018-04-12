# isochron v0.0
#
# PURPOSE:
#   Python script to plot a nice isochron with a York regression. More flexible and
#   interactive than plotIsochron script.
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
from scipy.stats import norm
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
  x = ds.iloc[idx,indexes[0]].as_matrix()
  sx = ds.iloc[idx,indexes[1]].as_matrix()
  y = ds.iloc[idx,indexes[2]].as_matrix()
  sy = ds.iloc[idx,indexes[3]].as_matrix()
  rho = ds.iloc[idx,indexes[4]].as_matrix()
  ids = list(ds.index[idx])
  return [x,sx,y,sy,rho,ids]

# Returns: (ax, zid, fig)
def initPlot():
  '''
  Initializes a plot window.
  Returns: ax, zid, fig
  '''
  # Prepare to plot the data.
  fig = p.figure(1)
  p.hold(True)
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

def print_stats(data,reg_results):
  '''
  Displays the specified regression results with some descriptive statistics.
  '''
  (x,sx,y,sy,rho) = unpackData(data,5)
  a,sa,b,sb = reg_results[0], reg_results[1], reg_results[2], reg_results[3]
  (mswd,smswd) = mswd_2d(x,sx,y,sy,a,b)
  r2 = calc_r2(x,y)
  print('{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}\t{:>10s}'.format('a','sa','b','sb','mswd','1s mswd','r2','n'))
  print('{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:10d}'.format(a,sa,b,sb,mswd,smswd,r2,len(x)))

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

def mswd_2d(x,sx,y,sy,a,b):
  '''
  Determines the MSWD for a line with intercept a and slope b that was fit to the
  specified data (x,sx,y,sy).

  Arguments:
  - x, sx - Data and uncertainties on the abscissa.
  - y, sy - Data and uncertainties on the ordinate.
  - a, b  - intercept and slope of the best fit line, y = a + bx.

  Returns: (mswd, smswd)
  '''
  # Check inputs.
  if len(x) != len(sx) or len(x) != len(y) or len(x) != len(sy):
    print('Error: input arrays must be the same length')
    return
  # Calculate MSWD.
  n = len(x)
  mswd = 0
  terms = np.zeros(n)
  for i in range(n):
    terms[i] = (y[i] - a - b*x[i])**2/(sy[i]**2 + (b*sx[i])**2)
  mswd = sum(terms)/(n-2)
  return (mswd,np.sqrt(2.0/(n-2)))

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
# Quick bootstrap function for 61015 and 60315 cosmochron search.

def bootstrap_cosmochron(path,rej=[],level=2):
  # This method assumes the data are sorted in descending order of uncertainty in 37/36.
  solar = 0.187
  ds = loadDataset(path)
  pd = extractData(ds,exclude=rej)
  res = yorkRegression(pd)
  count = 0
  limit = len(ds) - 1
  while abs(res[0] - solar) > level*res[1]:
    rej.append(pd[5][0])
    pd = extractData(ds,exclude=rej)
    res = yorkRegression(pd)
    count += 1
    if count == limit:
      break
  (ax,zid,fig) = initPlot()
  p.ylim([0,2]); p.xlim([0,5000])
  zid = plotRegression(ax,res[0],res[2],zid)
  zid = plotEnvelope(ax,res[0],res[1],res[2],res[3],zid,level=1)
  zid = plotEnvelope(ax,res[0],res[1],res[2],res[3],zid)
  zid = plotData(ax,pd,zid)
  print_stats(pd,res)

def bootstrap_isochron(path,rej=[],level=2):
  # This method assumes the data are sorted in descending order of uncertainty in 39/36.
  solar = 1
  ds = loadDataset(path)
  pd = extractData(ds,exclude=rej)
  res = yorkRegression(pd)
  count = 0
  limit = len(ds) - 1
  while abs(res[0] - solar) > level*res[1]:
    rej.append(pd[5][0])
    pd = extractData(ds,exclude=rej)
    res = yorkRegression(pd)
    count += 1
    if count == limit:
      break
  (ax,zid,fig) = initPlot()
  p.ylim([0,400000]); p.xlim([0,1640])
  zid = plotRegression(ax,res[0],res[2],zid)
  zid = plotEnvelope(ax,res[0],res[1],res[2],res[3],zid)
  zid = plotData(ax,pd,zid)
  print_stats(pd,res)
  for idi in pd[5]:
    print(idi)
