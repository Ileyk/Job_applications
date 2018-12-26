import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
# from bokeh import mpl
# from bokeh.plotting import output_file, show, ColumnDataSource, figure, vplot
# from bokeh.models import HoverTool
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import holoviews as hv
# import colorcet as cc
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.patches import Rectangle

import os

import subprocess

from parfile import *

# display
# (M2,P)
# (3,4) | (3,8) | (3,12)
# (2,4) | (2,8) | (2,12)
# (1,4) | (1,8) | (1,12)
# => M2[i//3] and P[i%3]
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

hv.notebook_extension("matplotlib")

# x = np.arange(0, np.pi, 0.1)
# y = np.arange(0, 2*np.pi, 0.1)
# X, Y = np.meshgrid(x, y)
# Z = np.cos(X) * np.sin(Y) * 10
# colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]  # R -> G -> B
# n_bins = [4, 6, 10, 100]  # Discretizes the interpolation into bins
# cmap_name = 'my_list'
# # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# vmax=25
# cm = LinearSegmentedColormap.from_list(cmap_name, [(0,    'blue'),
#                                               (0.2/vmax, 'red'),
#                                               (0.6/vmax, 'green'),
# 											  (2./vmax, 'black'),
# 											  (6./vmax, 'yellow'),
# 											  (20./vmax, 'orange'),
# 											  (vmax/vmax, 'cyan')], N=7)

# im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm)
# ax.set_title("N bins: %s" % 4)
# fig.colorbar(im, ax=ax)
# plt.show()

# fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
# for ax in zip(axs.ravel()):
#     # Create the colormap
#     cm = LinearSegmentedColormap.from_list(
#         cmap_name, colors, N=4)
#     # Fewer bins will result in "coarser" colomap interpolation
#     im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm)
#     ax.set_title("N bins: %s" % 4)
#     fig.colorbar(im, ax=ax)
# plt.show()

# def plot_array():

def fmt(x, pos):
	a, b = '{:.1e}'.format(x).split('e')
	b = int(b)
	return r'{} $\cdot $10$^{{{}}}$'.format(a, b)

# def fmtCbar(x, pos):
# 	#a = '{:5.2f}'.format(x) #.split('e')[0]
# 	a = '{:5.1f}'.format(x)#.split('e')[0]
# 	return a
# 	# return r'{}'.format(a)

def fmtCbar(x, pos):
	a, b = '{:.1e}'.format(x).split('e')
	b = int(b)
	if (b!=0 and b!=1 and b!=-1):
		return r'{}E{}'.format(a, b)
	elif (b==0):
		return r'{}'.format(a)
	elif (b==1):
		return '{:2.0f}'.format(x)
	elif (b==-1):
		return '{:3.2f}'.format(x)

font = {'family' : 'normal',
'size'   : fontsize}

#'weight' : 'bold',

plt.rc('font', **font)

Bmag=[1.E14,1.E16]
MNS=[2.,2.4]

# xmin, xmax, ymin, ymax = np.amin(np.asarray(fl, dtype='float')), np.amax(np.asarray(fl, dtype='float')), np.amin(np.asarray(q, dtype='float')), np.amax(np.asarray(q, dtype='float'))
# extent = xmin, xmax, ymin, ymax
# fig = plt.figure(frameon=False,figsize=(2*figsize,figsize))
fig, fig1 = plt.subplots(1,sharex=True,figsize=(1.1*figsize,figsize))

hor_labels=[r'',r'M$_{NS}=$2$M_{\odot}$','M$_{NS}=$2.4$M_{\odot}$']
vert_labels=[r'',r'$B=10^{14}$G',r'$B=10^{16}$G']

leg1=2*['']
leg1[0] = plt.Line2D((0,0),(0,1),linestyle='solid',linewidth=3.0,color='r')
leg1[1] = plt.Line2D((0,0),(0,1),linestyle='solid',linewidth=3.0,color='g')
leg2=2*['']
leg2[0] = plt.Line2D((0,0),(0,1),linestyle='dashed',linewidth=3.0,color='r')
leg2[1] = plt.Line2D((0,0),(0,1),linestyle='dashed',linewidth=3.0,color='g')
# fact2=[1.9,1.6,1.5]
# fact3=[4.2,2.8,2.3]
# @ 15
# fact2=[2.0,1.8,1.7]
# fact3=[3.8,3.0,2.7]
# # @ 30
# fact2=[2.1,1.8,1.6]
# fact3=[3.9,3.1,2.7]
# # @ 60
# fact2=[2.0,1.8,1.6]
# fact3=[3.6,3.0,2.6]
# # @ 150
# fact2=[1.9,1.8,1.6]
# fact3=[3.6,3.0,2.6]

# fact2=[2.1,1.8,1.7]
# fact3=[3.9,3.1,2.7]

fact2=[2.3,2.1,1.9]
fact3=[4.4,3.7,3.2]

# TIME IN SECONDS
# LUMINOSITY IN 1E50 erg/s

N=100
time=np.logspace(-1.,6.,num=N,base=10.0)
f=np.zeros((np.size(MNS),np.size(Bmag),N))
tsd=np.zeros((np.size(MNS),np.size(Bmag)))
for j in range(np.size(Bmag)):
	for i in range(np.size(MNS)):
		tsd[i][j]=150.*((MNS[i]/1.4)**(1.5))*((Bmag[j]/1.E15)**(-2.))
		for k in range(N):
			if (MNS[i]!=2.4) :
				f[i][j][k]=7.*((MNS[i]/1.4)**(1.5))*((Bmag[j]/1.E15)**2.)/(1.+time[k]/tsd[i][j])**2.
			elif (time[k]<tsd[i][j]) :
				f[i][j][k]=7.*((MNS[i]/1.4)**(1.5))*((Bmag[j]/1.E15)**2.)/(1.+time[k]/tsd[i][j])**2.
			else :
				f[i][j][k]=np.min(f[i][j][np.nonzero(f[i][j][:])])
# print np.shape(time[f[1][1][:]==np.min(f[1][1][np.nonzero(f[1][1][:])])]), np.shape(np.min(f[1][1][np.nonzero(f[1][1][:])]))
# fig1.scatter(time[f[1][1][:]==np.min(f[1][1][np.nonzero(f[1][1][:])])],np.min(f[1][1][np.nonzero(f[1][1][:])]), s=5, c='g', marker='o')
fig1.scatter(2.72,464,s=3*fontsize,c='g',marker='o')
fig1.scatter(29142.,0.045,s=3*fontsize,c='g',marker='o')
plt.text(2.72*1.3,464.*0.87,'collapse',fontsize=3*fontsize/4, color='g')
plt.text(29142.*1.3,0.045*0.87,'collapse',fontsize=3*fontsize/4, color='g')

fig1.plot(time,f[0][0][:],linestyle='solid',linewidth=3,color='r')
fig1.plot(time[f[1][0][:]>np.min(f[1][0][np.nonzero(f[1][0][:])])],f[1][0][f[1][0][:]>np.min(f[1][0][np.nonzero(f[1][0][:])])],linestyle='solid',linewidth=3,color='g')
fig1.plot(time,f[0][1][:],linestyle='dashed',linewidth=3,color='r')
fig1.plot(time[f[1][1][:]>np.min(f[1][1][np.nonzero(f[1][1][:])])],f[1][1][f[1][1][:]>np.min(f[1][1][np.nonzero(f[1][1][:])])],linestyle='dashed',linewidth=3,color='g')
fig1.set_xscale('log')
fig1.set_yscale('log')
fig1.grid(which='major', linestyle='dotted', linewidth=2, alpha=0.9)
# fig1.grid(which='minor', linestyle='dotted', linewidth=2, alpha=0.9)
fig1.set_ylim(1.E-4,6.E3)
# plt.text(0.8, 2.E3, '1 second', rotation=90, fontsize=3*fontsize/4)
plt.text(3600.*0.8, 3.E3, '1 hour', rotation=90, fontsize=3*fontsize/4)
plt.text(86400.*0.8, 3.E3, '1 day', rotation=90, fontsize=3*fontsize/4)
plt.text(604800.*0.8, 3.E3, '1 week', rotation=90, fontsize=3*fontsize/4)
fig1.set_xlabel(r'Time since merger (in seconds)',fontweight='bold',fontsize=fontsize)
fig1.set_ylabel(r'Spinning-down luminosity (in $10^{50}$erg$\cdot$s$^{-1}$)', fontweight='bold', fontsize=fontsize)
# fig1.axvline(x=1.,linewidth=2,linestyle='dotted',alpha=0.6,color='k') # 1 second
# fig1.axvline(x=3600.) # 1 hour
# fig1.axvline(x=86400.) # 1 day
# fig1.axvline(x=604800.) # 1 week
# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# legend_handle = [extra, extra, extra, extra, leg1[0], leg2[0], extra, leg1[1], leg2[1]]
# leg_lab=np.concatenate([v_type,[hor_labels[0],'',''],[hor_labels[1],'','']])
# fig1.legend(legend_handle,leg_lab,
# 	loc='lower left',fontsize=16,ncol=3,shadow=True,handletextpad=-0.5,numpoints=1)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
legend_handle = [extra, extra, extra, extra, leg1[0], leg2[0], extra, leg1[1], leg2[1]]
leg_lab=np.concatenate([vert_labels,[hor_labels[1],'',''],[hor_labels[2],'','']])
fig1.legend(legend_handle,leg_lab,
	loc='lower left',fontsize=3*fontsize/4,ncol=3,shadow=True,handletextpad=-2.2,numpoints=1)
fig.tight_layout()
fig.savefig(outputs+'spinning-down_luminosity.png',bbox_inches='tight')
# plt.show()
# stop
# fig1.set_ylim(1.,5.)
# fig1.set_xlim(1.,2.1)
# # fig1.set_xscale('log')
# # fig1.set_yscale('log')
#
# extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# legend_handle = [extra, leg1[0], extra, leg1[1]]
# leg_lab=np.concatenate([[hor_labels[0],''],[hor_labels[1],'']])
# fig1.legend(legend_handle,leg_lab,
# 	loc='upper right',fontsize=16,ncol=3,shadow=True,handletextpad=-2,numpoints=1)
# fig1.grid(which='major', linestyle='dotted', linewidth=2, alpha=0.9)
# fig1.grid(which='minor', linestyle='dotted', linewidth=2, alpha=0.9)
# fig.tight_layout()
# fig.savefig(outputs+'CEDE.png',bbox_inches='tight')

# 			# mdot[j][k] = max(mdot[j][k],0.002)
# 			fast[j][k] = fast_all[i/len(bet)][i%len(bet)][k][j]
# 	# Here, convert mdot into X-ray luminosity by assuming :
# 	#	- a stellar mass-loss rate of 5E-6 solar masses per year
# 	# 	- a 1/3 efficiency conversion factor (both for the BH and the NS)
# 	# fact = (1./3.) * ( 5.E-6 * (2.E33/(3600.*24.*365.25)) ) * (3.E10)**2.
# 	# or convert in % if want the fraction of stellar wind captured
# 	fact = 100.
# 	mdot = mdot * fact
#
# 	# print np.min(mdot[np.nonzero(mdot)]),np.max(mdot)
# 	mdott = pd.DataFrame(mdot,columns=eta,index=fll) # , norm=LogNorm(vmin=0.001,vmax=0.3)
# 	fastt = pd.DataFrame(fast,columns=eta,index=fll)
# 	# if (i/len(bet)==0): # first row, BH
# 	# vmin=0.0002,vmax=0.225
# 	cax1 = ax.imshow(mdott, vmin=-100., vmax=100., cmap=cm.RdBu, interpolation='nearest', origin='upper',extent=[-0.5,float(len(eta))-0.5,float(len(fll))-0.5-0.05,-0.55]) # cbar.set_clim(1e35, 1e38)
# 	# cax1 = ax.imshow(mdott, norm=LogNorm(vmin=0.01, vmax=25.), cmap=cm.inferno, interpolation='nearest', origin='upper',extent=[-0.5,float(len(eta))-0.5,float(len(fll))-0.5-0.05,-0.55]) # cbar.set_clim(1e35, 1e38)
# 	# cax1 = ax.imshow(fastt, vmin=-1.,vmax=1., cmap=cm.binary, interpolation='nearest', origin='upper',extent=[-0.5,float(len(eta))-0.5,float(len(fll))-0.5-0.05,-0.55])
# 	# else: # second row, NS
# 	# 	# vmin=0.00005,vmax=0.076
# 	# 	cax2 = ax.imshow(mdott, norm=LogNorm(vmin=0.0001*fact,vmax=0.25*fact), cmap=cm.nipy_spectral, interpolation='nearest', origin='upper',extent=[-0.5,float(len(eta))-0.5,float(len(fll))-0.5-0.05,-0.55]) # cbar.set_clim(1e35, 1e38)
# 	# cax2 = ax.imshow(fastt, vmin=-1.,vmax=1., cmap=cm.binary, interpolation='nearest', origin='upper',extent=[-0.5,float(len(eta))-0.5,float(len(fll))-0.5-0.05,-0.55])
#
# 	if (i%3==0):
# 		# if (i/len(bet)==0): plt.title(r'$\beta=1$',fontsize=fontsize)
# 		if (i/len(bet)==0): plt.title(r'Fast acceleration',fontsize=fontsize)
# 	if (i%3==1):
# 		# if (i/len(bet)==0): plt.title(r'$\beta=2$',fontsize=fontsize)
# 		if (i/len(bet)==0): plt.title(r'Intermediate acceleration',fontsize=fontsize)
# 	if (i%3==2):
# 		# if (i/len(bet)==0): plt.title(r'$\beta=3$',fontsize=fontsize)
# 		if (i/len(bet)==0): plt.title(r'Slow acceleration',fontsize=fontsize)
#
# 	if (i%3==0):
# 		if (i/len(bet)==0):plt.text(-6., 5.1, r'BH', fontsize=2*fontsize, color='k')
# 		if (i/len(bet)==1):plt.text(-6., 5.1, r'NS', fontsize=2*fontsize, color='k')
#
# 	if (i%3==2 and i/len(bet)==0):
# 		# Add colorbar, make sure to specify tick locations to match desired ticklabels
# 		# divider = make_axes_locatable(ax)
# 		#cax = divider.append_axes("bottom", size="5%", pad=0.7)
# 		cax = inset_axes(ax,
# 		width="15%",
# 		height="200%",
# 		bbox_transform=ax.transAxes,
# 		bbox_to_anchor=(0.7, -1.05, 1., 1.),
# 		loc= 8)
# 		# cbar = f.colorbar(cax1, format=ticker.FuncFormatter(fmt),orientation='horizontal')
# 		# # cbar = f.colorbar(cax1)#orientation='vertical')
# 		# # tick_locator = ticker.MaxNLocator(nbins=4) # number of ticks in the colorbar
# 		# tick_locator = ticker.MaxNLocator(nbins=3) # number of ticks in the colorbar
# 		# plt.locator = tick_locator
# 		# cbar.update_ticks()
# 		# cbar.ax.get_yaxis().labelpad = 15
# 		# if (i/len(bet)==0): # first row, BH
# 		cbar = fig.colorbar(cax1,cax=cax,orientation='vertical')#,format=fmtCbar())#format=LogFormatterMathtext()) #
# 		# else: # second row, NS
# 		# 	cbar = fig.colorbar(cax2,cax=cax,orientation='vertical')#,format=fmtCbar())#format=LogFormatterMathtext()) #
# 		# cbar.ax.minorticks_on()
#
# 		cbar.ax.tick_params(axis='y',which='both',direction='out',width=2,length=6)
# 		# We need to nomalize the tick locations so that they're in the range from 0-1...
# 		# minorticks = cbar.norm([0.001*fact*(i+1) for i in range(2,10)]+[0.01*fact*(i+1) for i in range(10)]+[0.1*fact*(i+1) for i in range(2)])
# 		# cbar.ax.yaxis.set_ticks(minorticks, minor=True)
# 		cbar.ax.yaxis.set_ticks_position('right')
# 		# cbar.ax.yaxis.set_major_formatter(ScalarFormatter())
# 		# for axis in [cbar.ax.xaxis, ax.yaxis]:
# 		#     axis.set_major_formatter(ScalarFormatter())
# 		# tick_locator = ticker.MaxNLocator(nbins=40) #ticker.LogLocator(base=1.15,numticks=8) # number of ticks in the colorbar
# 		# cbar.locator = tick_locator
# 		# # if (i%3==0) : cbar.formatter = ticker.FuncFormatter(fmtCbar4)
# 		# if (i%3==1) : cbar.formatter = ticker.FuncFormatter(fmtCbar)
# 		# # if (i%3==2) : cbar.formatter = ticker.FuncFormatter(fmtCbar6)
# 		# # cbar.formatter = ticker.FixedFormatter([("%2.0f"%(mdot[j][k]))[0:] for j in range(0,len(fll)) for k in range(0,len(eta)) ] )
# 		# cbar.update_ticks()
# 		# cbar.ax.get_yaxis().labelpad = 15
# 		# # ax.axvline(100)
#
# 	if (i//3==1 and i%3==2):
# 		plt.text(1.6, 1., r'$\mu=\dot{M}/\dot{M}_{*} (\%)$', fontsize=2*fontsize, color='k', fontweight='bold', rotation=-90, horizontalalignment='center',verticalalignment='center',transform = ax.transAxes)
#
# 	ax.yaxis.tick_left()
# 	ax.yaxis.set_label_position('left')
# 	ax.yaxis.set_major_formatter(ticker.NullFormatter())
# 	ticks_y=[]
# 	for j in range(0,len(fll)) :
# 		ticks_y.append(float(j))
# 	ticks_y=np.array(ticks_y)
# 	ax.yaxis.set_minor_locator(ticker.FixedLocator(ticks_y))
# 	ax.set_ylim((ticks_y[0]-0.5,ticks_y[len(fll)-1]+0.5))
# 	if (i%3==0):
# 		# fll=100.*fll # convert to %
# 		ax.yaxis.set_minor_formatter(ticker.FixedFormatter([("%2.0f"%(100*fll[j]))[0:] for j in range(0,len(fll))]))
# 		# ax.yaxis.set_minor_formatter(ticker.FixedFormatter(fll))
# 		ax.set_ylabel(r'Filling factor (%)', fontsize=fontsize)# + '$10^{{{0:d}}}$'.format(scale_pow))
# 	ax.yaxis.set(ticks=np.arange(0.5,len(fll)))
# 	ax.grid(which='major', axis='y', linewidth=1.5, linestyle='-', color='k', alpha=0.7)
#
# 	ax.xaxis.tick_bottom()
# 	ax.xaxis.set_label_position('bottom')
# 	ax.xaxis.set_major_formatter(ticker.NullFormatter())
# 	ticks_x=[]
# 	for j in range(0,len(eta)) :
# 		ticks_x.append(float(j))
# 	ticks_x=np.array(ticks_x)
# 	ax.xaxis.set_minor_locator(ticker.FixedLocator(ticks_x))
# 	ax.set_xlim((ticks_x[0]-0.5,ticks_x[len(eta)-1]+0.5))
# 	if (i/len(bet)==1):
# 		# print ticks_x
# 		# for tick in ax.get_xticklabels():
# 		# 	tick.set_rotation(45)
# 		# locs, labels = plt.xticks()
# 		# plt.setp(labels, rotation=90)
# 		ax.xaxis.set_minor_formatter(ticker.FixedFormatter([("%6.2f"%eta[j]) for j in range(0,len(eta))]))
# 		plt.setp(ax.xaxis.get_minorticklabels(), rotation=40, ha="right")
# 		ax.set_xlabel('Speed ratio $\eta=v_{\infty}/v_{orb}$', fontsize=fontsize)# + '$10^{{{0:d}}}$'.format(scale_pow))
#
#
# 	# plt.xticks(rotation=40)
#
# 	# ax.xaxis.set_minor_formatter(ticker.FixedFormatter(eta),rotation=40)
#
# 	# if (i/len(bet)==1):
# 	# 	ax.set_xlabel('Speed ratio $\eta$', fontsize=fontsize)# + '$10^{{{0:d}}}$'.format(scale_pow))
# 	# else:
# 	# 	plt.tick_params(
# 	# 	axis='x',          # changes apply to the x-axis
# 	# 	which='both',      # both major and minor ticks are affected
# 	# 	bottom='on',      # ticks along the bottom edge are off
# 	# 	top='on',         # ticks along the top edge are off
# 	# 	labelbottom='off') # labels along the bottom edge are off
# 		# ax.get_xaxis().set_visible(False)
# 	ax.xaxis.set(ticks=np.arange(0.5,len(eta)))
#
# 	ax.grid(which='major', axis='x', linewidth=1.5, linestyle='-', color='k', alpha=0.7)
#
#
# 	plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.55, hspace=0.05)		# share y axis
#
#
# 	# ax.yaxis.tick_right()
# 	# ax.yaxis.set_label_position('right')
# 	# ax.grid(which='major', axis='y', linewidth=2.5, linestyle=':', color='k', alpha=1)
# 	# ax.xaxis.set_major_formatter(ticker.NullFormatter())
# 	# plt.tick_params(
# 	# axis='x',          # changes apply to the x-axis
# 	# which='both',      # both major and minor ticks are affected
# 	# bottom='on',      # ticks along the bottom edge are off
# 	# top='on',         # ticks along the top edge are off
# 	# labelbottom='off') # labels along the bottom edge are off
# 	# ax.grid(which='major', axis='x', linewidth=2.5, linestyle=':', color='k', alpha=1)
# 	# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=-0.36)		# share y axis
#
# # fig.savefig(outputs+'mdot_grid.png',bbox_inches='tight')
#
# plt.show()
