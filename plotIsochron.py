#plotIsochron v0.0
#
#PURPOSE:
#  Python script to plot a nice isochron with a York regression.
#
#AUTHOR:
#  Cameron M. Mercer
#  School of Earth and Space Exploration
#  Arizona State University, Tempe, AZ 85287
# 

import matplotlib as mpl       #Import matplotlib alone to reset the graphics renderer.
import matplotlib.pyplot as p  #Import plotting tools.
import numpy as np             #Import math functions.
import sys
import csv
from scipy.stats import norm
from matplotlib.patches import Ellipse
#Make text labels text boxes rather than character paths.
mpl.rcParams['pdf.fonttype'] = 'truetype'
#mpl.rc('font',family='serif',serif='Times New Roman') #Times New Roman
mpl.rc('font',family='sans-serif',serif='Arial')
#Make plot frames and ticks thicker.
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1

#Initialize list of tags expected from tag file; provide default values.
tags = np.array([['xlabel','Abscissa'],['ylabel','Ordinate'],['xmin','auto'],['xmax','auto'],['ymin','auto'],\
                 ['ymax','auto'],['lwidth',1.0],['lcolor','black'],['ind_marker','o'],['ind_color','black'],\
                 ['ind_size',5],['alt_first','False'],['alt_list',None],['alt_color','red'],['xkcd',0],\
                 ['capsize',5],['scale',1.0],['error_ellipses',0],['ellipse_color','white'],['ellipse_alpha',0.5]])

#Get arguments from command line: (1) file of tab-delimited data, in five columns (x, Sx, y, Sy, rho);
#(2) file with regression results (a, Sa, b, Sb for y = a + bx); (3) tag file listing plot parameters.
#Note: the plot data should have column names and row names.
#Note: the regression parameters should only have column names.
if len(sys.argv) == 2:  #Expecting 1 argument plus plotIsochron itself (argv[0]), hence the 2.
  dFile = sys.argv[1]
elif len(sys.argv) == 4:  #Expecting 3 arguments plus plotRS itself (argv[0]), hence the 4.
  dFile, rFile, tFile = sys.argv[1], sys.argv[2], sys.argv[3]

#Read in the plot data.
f = open(dFile,'rU')  #The 'rU' opens file with read permissions, universal line endings (\n or \r).
reader = csv.reader(f,delimiter='\t')
plotData = np.array(np.zeros((1,5)),dtype='d')
#Read file into plotData. Note, ignoring column names
for idx, row in enumerate(reader):
  if idx == 1:
    plotData = np.array([row[1],row[2],row[3],row[4],row[5]]) #Ignore point labels for now, just get data.
  elif idx > 1:
    plotData = np.vstack((plotData,np.array([row[1],row[2],row[3],row[4],row[5]])))
#Close plot data file now that data are ingested. Convert plotData to doubles.
f.close()
plotData = np.array(plotData[:,:],dtype='d')
#Read in the regression parameters.
f = open(rFile,'rU')  #The 'rU' opens file with read permissions, universal line endings (\n or \r).
reader = csv.reader(f,delimiter='\t')
yorkData = np.array(np.zeros((4)),dtype='d')
for idx, row in enumerate(reader):
    if idx == 1:
        yorkData = np.array(row,dtype='d')
    if idx > 1:
        yorkData = np.vstack((yorkData,np.array(row,dtype='d')))
#Close regression data file.
f.close()
yorkData = np.array(yorkData[:],dtype='d')

#Get tag file with plot customizations.
f = open(tFile,'rU')
reader = csv.reader(f,delimiter='=')
for idx, row in enumerate(reader):
    if row[0] in tags[:,0]:
        try: #Convert entry to a number, if possible.
            tags[idx,1] = np.double(row[1])
        except ValueError:
            if row[1] != 'None':  #If row[1] == None, don't change default value.
                tags[idx,1] = row[1]
                #Check if list, if so split and try to make numeric.
                if len(tags[idx,1].split(',')) > 1:
                    tags[idx,1] = tags[idx,1].split(',')
                    for i in range(len(tags[idx,1])):
                        try: #Convert entry to a number if possible.
                            tags[idx,1][i] = int(tags[idx,1][i])
                        except ValueError:
                            #Try to split further if RGB vector delimited by ';'.
                            if len(tags[idx,1][i].split(';')) > 1:
                                rgb = tags[idx,1][i].split(';')
                                tags[idx,1][i] = np.array([rgb[0],rgb[1],rgb[2]],dtype='d')
                elif len(tags[idx,1].split(';')) > 1:
                    rgb = tags[idx,1].split(';')
                    tags[idx,1] = np.array([rgb[0],rgb[1],rgb[2]],dtype='d')
# Close tagfile now that preferences are ingested.
f.close()

# Scale uncertainties.
plotData[:,1] *= tags[np.where(tags[:,0]=='scale')[0][0],1]
plotData[:,3] *= tags[np.where(tags[:,0]=='scale')[0][0],1]

# Enable xkcd style plotting, if desired.
if tags[np.where(tags[:,0]=='xkcd')[0][0],1] == 1:
    p.xkcd()

# Subtract 1 from alt_list elements.
try:
    for m, item in enumerate(tags[np.where(tags[:,0]=='alt_list')[0][0],1]):
        tags[np.where(tags[:,0]=='alt_list')[0][0],1][m] = tags[np.where(tags[:,0]=='alt_list')[0][0],1][m] - 1
except TypeError:
    pass

# Prepare to plot the data.
fig = p.figure(1)
ax = p.gca()
zid = 1
    
# Determine parameters for error ellipses if needed.
if tags[np.where(tags[:,0]=='error_ellipses')[0][0],1] == 1:
  # Calculate the covariance.
  Sx, Sy, rho = plotData[:,1], plotData[:,3], plotData[:,4]
  Sxy = Sx*Sy*rho
  # Prepare to calculate ellipse parameters.
  dims = np.zeros((plotData.shape[0],2),dtype='d') #dims[0] = x-dimension, dims[1] = y-dimension
  phi = np.zeros((plotData.shape[0],),dtype='d')
  aspect = np.zeros((plotData.shape[0],),dtype='d')
  for i in range(len(Sxy)):
    # Make covariance matrix and calculate ellipse dimensions.
    covmat = np.array([[np.power(Sx[i],2),Sxy[i]],[Sxy[i],np.power(Sy[i],2)]])
    vals, vects = np.linalg.eigh(covmat)
    if Sx[i] > Sy[i]:
      dims[i,0] = np.sqrt(max(vals))
      dims[i,1] = np.sqrt(min(vals))
    else:
      dims[i,0] = np.sqrt(min(vals))
      dims[i,1] = np.sqrt(max(vals))
    phi = np.degrees(np.arctan(2.0*Sxy/(np.power(Sx,2)-np.power(Sy,2)))/2.0)
    
# Define convenience function for plotting alternate error ellipses.
def plot_alt_errEllipses(zid):
  # Plot data with alternate appearance.
  for i in range(len(plotData[:,0])):
    try:
      if i in tags[np.where(tags[:,0]=='alt_list')[0][0],1]:
        ellFace = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                          angle=phi[i],facecolor=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                          edgecolor='none',alpha=tags[np.where(tags[:,0]=='ellipse_alpha')[0][0],1],\
                          zorder=zid)
        zid += 1
        ellEdge = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                          angle=phi[i],facecolor='none',\
                          edgecolor=tags[np.where(tags[:,0]=='lcolor')[0][0],1],zorder=zid)
        zid += 1
        ax.add_artist(ellFace)
        ax.add_artist(ellEdge)
        #Add point to center of ellipse.
        p.plot(plotData[i,0],plotData[i,2],\
               marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
               color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],zorder=zid)
        zid += 1
    except TypeError:
      ellFace = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                        angle=phi[i],facecolor=tags[np.where(tags[:,0]=='ellipse_color')[0][0],1],\
                        edgecolor='none',alpha=tags[np.where(tags[:,0]=='ellipse_alpha')[0][0],1],\
                        zorder=zid)
      zid += 1
      ellEdge = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                        angle=phi[i],facecolor='none',\
                        edgecolor=tags[np.where(tags[:,0]=='lcolor')[0][0],1],zorder=zid)
      zid += 1
      ax.add_artist(ellFace)
      ax.add_artist(ellEdge)
      #Add point to center of ellipse.
      p.plot(plotData[i,0],plotData[i,2],\
             marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
             color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],zorder=zid)
      zid += 1
  return zid

# Define convenience functions for plotting regular error ellipses.
def plot_errEllipses(zid):
  # Plot data with normal appearance.
  for i in range(len(plotData[:,0])):
    try:
      if i not in tags[np.where(tags[:,0]=='alt_list')[0][0],1]:
        ellFace = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                          angle=phi[i],facecolor=tags[np.where(tags[:,0]=='ellipse_color')[0][0],1],\
                          edgecolor='none',alpha=tags[np.where(tags[:,0]=='ellipse_alpha')[0][0],1],\
                          zorder=zid)
        zid += 1
        ellEdge = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                          angle=phi[i],facecolor='none',\
                          edgecolor=tags[np.where(tags[:,0]=='lcolor')[0][0],1],zorder=zid)
        zid += 1
        ax.add_artist(ellFace)
        ax.add_artist(ellEdge)
        #Add point to center of ellipse.
        p.plot(plotData[i,0],plotData[i,2],\
               marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
               color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
               ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],zorder=zid)
        zid += 1
    except TypeError:
      ellFace = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                        angle=phi[i],facecolor=tags[np.where(tags[:,0]=='ellipse_color')[0][0],1],\
                        edgecolor='none',alpha=tags[np.where(tags[:,0]=='ellipse_alpha')[0][0],1],\
                        zorder=zid)
      zid += 1
      ellEdge = Ellipse(xy=(plotData[i,0],plotData[i,2]),width=2*dims[i,0],height=2*dims[i,1],\
                        angle=phi[i],facecolor='none',\
                        edgecolor=tags[np.where(tags[:,0]=='lcolor')[0][0],1],zorder=zid)
      zid += 1
      ax.add_artist(ellFace)
      ax.add_artist(ellEdge)
      #Add point to center of ellipse.
      p.plot(plotData[i,0],plotData[i,2],\
             marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
             color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
             ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],zorder=zid)
      zid += 1
  return zid

# Plot up error ellipses in the correct order.
if tags[np.where(tags[:,0]=='error_ellipses')[0][0],1] == 1:
  if tags[np.where(tags[:,0]=='alt_first')[0][0],1] == 'True':
    zid = plot_alt_errEllipses(zid)
    zid = plot_errEllipses(zid)
  else:
    zid = plot_errEllipses(zid)
    zid = plot_alt_errEllipses(zid)
    
    '''
    #Add scatter points in centers of ellipses.
    for i in range(len(plotData[:,0])):
        try:
            if i in tags[np.where(tags[:,0]=='alt_list')[0][0],1]:
                if tags[np.where(tags[:,0]=='error_ellipses')[0][0],1] == 1:
                    p.plot(plotData[i,0],plotData[i,2],\
                               marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                               color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                               mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                               mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                               ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1])
                else:
                    p.plot(plotData[i,0],plotData[i,2],\
                               marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                               color=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                               mfc=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                               mec=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                               ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1])
            else:
                p.plot(plotData[i,0],plotData[i,2],\
                           marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                           color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                           mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                           mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                           ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1])
        except TypeError:
            p.plot(plotData[i,0],plotData[i,2],\
                       marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                       color=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                       mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                       mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                       ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1])
        '''
else:
  #Plot the data with normal error bars; the regressed line will be plotted later.
  fig = p.figure(1)
  #Note: np.where(tags[:,0]=='tag')[0][0] returns the row index for 'tag' in the tag
  #array ([0][0] deals with returned tuple of arrays).
  for i in range(len(plotData[:,0])):
    try:
      if i in tags[np.where(tags[:,0]=='alt_list')[0][0],1]:
        p.errorbar(plotData[i,0],plotData[i,2],xerr=plotData[i,1],yerr=plotData[i,3],\
                   marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                   ecolor=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                   mfc=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                   mec=tags[np.where(tags[:,0]=='alt_color')[0][0],1],\
                   ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],\
                   elinewidth=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                   capthick=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                   capsize=tags[np.where(tags[:,0]=='capsize')[0][0],1])
      else:
        p.errorbar(plotData[i,0],plotData[i,2],xerr=plotData[i,1],yerr=plotData[i,3],\
                   marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                   ecolor=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                   mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                   mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                   ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],\
                   elinewidth=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                   capthick=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                   capsize=tags[np.where(tags[:,0]=='capsize')[0][0],1])
    except TypeError:
      p.errorbar(plotData[i,0],plotData[i,2],xerr=plotData[i,1],yerr=plotData[i,3],\
                 marker=tags[np.where(tags[:,0]=='ind_marker')[0][0],1],\
                 ecolor=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                 mfc=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                 mec=tags[np.where(tags[:,0]=='ind_color')[0][0],1],\
                 ms=tags[np.where(tags[:,0]=='ind_size')[0][0],1],\
                 elinewidth=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                 capthick=tags[np.where(tags[:,0]=='lwidth')[0][0],1],\
                 capsize=tags[np.where(tags[:,0]=='capsize')[0][0],1])

#Make plot on log-log scale.
#ax = p.gca()
#ax.set_yscale('log')
#ax.set_xscale('log')

#Enforce plot limits.
xlims = p.xlim()
ylims = p.ylim()
try:
  if tags[np.where(tags[:,0]=='xmin')[0][0],1] == 'auto':
    tags[np.where(tags[:,0]=='xmin')[0][0],1] = xlims[0]
  if tags[np.where(tags[:,0]=='xmax')[0][0],1] == 'auto':
    tags[np.where(tags[:,0]=='xmax')[0][0],1] = xlims[1]
  if tags[np.where(tags[:,0]=='ymin')[0][0],1] == 'auto':
    tags[np.where(tags[:,0]=='ymin')[0][0],1] = ylims[0]
  if tags[np.where(tags[:,0]=='ymax')[0][0],1] == 'auto':
    tags[np.where(tags[:,0]=='ymax')[0][0],1] = ylims[1]
except ValueError:
  pass
p.xlim(tags[np.where(tags[:,0]=='xmin')[0][0],1],tags[np.where(tags[:,0]=='xmax')[0][0],1])
p.ylim(tags[np.where(tags[:,0]=='ymin')[0][0],1],tags[np.where(tags[:,0]=='ymax')[0][0],1])
#Plot the regressed line(s).
xlims = p.xlim()
ylims = p.ylim()
try:
  for i in range(len(yorkData[:,0])):
    p.plot([xlims[0],xlims[1]],[yorkData[i,0]+yorkData[i,2]*xlims[0],yorkData[i,0]+yorkData[i,2]*xlims[1]],\
           color=tags[np.where(tags[:,0]=='lcolor')[0][0],1],linestyle='--',\
           linewidth=tags[np.where(tags[:,0]=='lwidth')[0][0],1])
except IndexError:
  p.plot([xlims[0],xlims[1]],[yorkData[0]+yorkData[2]*xlims[0],yorkData[0]+yorkData[2]*xlims[1]],\
         color=tags[np.where(tags[:,0]=='lcolor')[0][0],1],linestyle='--',\
         linewidth=tags[np.where(tags[:,0]=='lwidth')[0][0],1])

#Set axis labels.
p.xlabel(tags[np.where(tags[:,0]=='xlabel')[0][0],1],fontsize=16)
p.ylabel(tags[np.where(tags[:,0]=='ylabel')[0][0],1],fontsize=16)

p.show()
                   
