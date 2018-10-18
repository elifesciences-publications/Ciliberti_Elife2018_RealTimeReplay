#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:11:30 2017

@author: davide
"""
import numpy as np
import seaborn
from scipy.stats import cumfreq
from matplotlib import pyplot as plt
from pylab import rc
import seaborn as sns
from fklab.plot.plotting import noline_legend

def plot_latency_distribution( filepath, label, bw=None, gridsize=1000, color="gray", title=None ):
    '''
    
    '''
    
    latencies = np.load( filepath )
    seaborn.reset_orig()
    
    plt.figure(1)
#    ax = plt.subplot(111)
    print(bw)
    if bw is None: bw="scott"
    seaborn.kdeplot( latencies,  bw=bw, gridsize=gridsize, color=color, shade=True )
    plt.xlabel( label,  fontsize=20 )
    plt.ylabel( 'probability', fontsize=20, visible=False )
    ylim = plt.ylim()[1]
    print ylim
    plt.vlines(  np.median(latencies), 0, ylim, color='r', label='median', linewidth=4 )
    plt.text( np.median(latencies), ylim*.9,\
        str( np.round( np.median( latencies ), decimals=2 ))  + " ms" ,\
        color='r', fontsize=30, weight='bold')
#    xt = np.linspace( 0, latencies.max(), 20 )
#    xt = np.linspace(0, 300, 5, dtype='int')
    plt.xticks( fontsize=20 )
    plt.yticks( [0, ylim/2, ylim], fontsize=20, visible=False )
#        plt.xlim( 0, np.max( self.absolute_latencies()+40 ) )
    plt.xlim( 0, latencies.max()*1.15 )
#        plt.legend( loc="best", fontsize=fs_labels )
#    ax.spines['top'].set_color('none')
#    ax.spines['left'].set_color('none')
#    ax.spines['right'].set_color('none')
#    ax.tick_params( which=u'both',top='off', right='off', left='off', bottom='on', length=0 )
    plt.title( title )
    plt.grid()
    

def configure_latency_plots( ax, latencies, labels, nbins, colors, fs=15, loc="best", show_median=True, pad=7.5, lw=2, alpha=.25 ):
    
    for lat, lbl, col in zip(latencies, labels, colors):
        x = np.linspace( np.nanmin( lat ), np.nanmax( lat ), num=nbins )
        ax.plot( x, cumfreq( lat[~np.isnan(lat)], nbins )[0]/np.count_nonzero(~np.isnan(lat) ),\
                label=lbl, color=col, lw=lw )
    
    plt.ylabel( "cumulative fraction", fontsize=fs )
    plt.xlim( xmin=0 )
    plt.ylim( ymin=0, ymax=1.005 )
    
    leg = ax.legend( loc=loc, frameon=False )
    noline_legend( leg )
        
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    ax.xaxis.set_tick_params( direction='in' )
    ax.yaxis.set_tick_params( direction='in' )
    
    yticks = [0, .5, 1]
    ax.yaxis.set_ticks( yticks )
    ax.yaxis.set_ticklabels( [str(tk) for tk in yticks] )
    ax.tick_params( labelsize=fs, axis='both', which='major', pad=pad  )
    
    if show_median:
        for i, lat in enumerate(latencies):
            median = np.nanmedian( lat )
            plt.vlines( median, 0, .5, color=colors[i], linestyles='dashed', alpha=alpha )
            plt.hlines( .5, 0, median, color=colors[i], linestyles='dashed', alpha=alpha )

        
def plot_param_performance_dependency( params_range, xlabel, xticks, performances,\
      lw=1.5, fs=12, colors=sns.color_palette( "cubehelix", n_colors=3 ), fign=1  ):
    '''
    '''
    # GENERAL SETTINGS
    xlim = (0, max(xticks))
    ylim = (0, 1.01)
    yticks = np.linspace( 0, 1, num=5 )
    yticklabels = [ str(yt) if yt%0.5==0 else "" for yt in yticks ]
    yticklabels[0] = str(0)
    yticklabels[-1] = str(1)
    xticklabels = [ str(int(xt)) if xt%1==0 else str(xt) for xt in xticks ]
    fig = plt.figure( fign, figsize=(8, 6) )
    sns.reset_defaults()
    
    # COMPUTE PARAMS
    sens = np.array( [ p['sensitivity'] for p in performances ] )
    spec = np.array( [ p['specificity'] for p in performances ] )
    ic_acc = np.array( [ p['inter_content_accuracy'] for p in performances ] )    
    mrl = np.array( [ p['median relative latency'] for p in performances ] )
    yi = sens + spec - 1

    # SUB PANEL 1
    ax1 = fig.add_subplot( 221 )
    ax1.plot( params_range, sens, linewidth=lw, label="sensitivity",\
             color=colors[0] )
    ax1.plot( params_range, spec, linewidth=lw, label="specificity",\
             color=colors[1] )
    ax1.plot( params_range, yi, linewidth=lw, label="Youden's index",\
             color=colors[2] )
    plt.xlabel( xlabel, fontsize=fs )
    plt.ylabel( "detection index", fontsize=fs, rotation="vertical" )
    plt.xticks( xticks, fontsize=fs )
    plt.yticks( yticks, fontsize=fs )
    ax1.xaxis.set_tick_params( direction='in' )
    ax1.yaxis.set_tick_params( direction='in' )
    ax1.xaxis.set_ticklabels( xticklabels )
    ax1.yaxis.set_ticklabels( yticklabels )
    ax1.tick_params( labelsize=fs, axis='both', which='major', pad=7.5 )
    plt.xlim( xlim )
    plt.ylim( ylim )
    ax1.spines['top'].set_color( 'none' )
    ax1.spines['right'].set_color( 'none' )
    
    leg = ax1.legend( loc="lower left", frameon=False, handlelength=0, handletextpad=0,\
                     fontsize=8, prop={'weight':'bold'} )
    for item in leg.legendHandles:
        item.set_visible(False)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color( line.get_color() ) 
    
    # SUB PANEL 2
    ax2 = fig.add_subplot( 222 )
    ax2.plot( params_range, ic_acc, linewidth=lw, color="k"  )
    plt.xlabel( xlabel, fontsize=fs )
    plt.ylabel( "inter-content accuracy", fontsize=fs, rotation="vertical" )
    plt.xticks( xticks, fontsize=fs )
    plt.yticks( yticks, fontsize=fs )
    ax2.xaxis.set_tick_params( direction='in' )
    ax2.yaxis.set_tick_params( direction='in' )
    ax2.xaxis.set_ticklabels( xticklabels )
    ax2.yaxis.set_ticklabels( yticklabels )
    ax2.tick_params( labelsize=fs, axis='both', which='major', pad=7.5 )
    plt.xlim( xlim )
    plt.ylim( ylim )
    ax2.spines['top'].set_color( 'none' )
    ax2.spines['right'].set_color( 'none' )
    
    # SUB PANEL 3
    ax3 = fig.add_subplot( 224 )
    ax3.plot( params_range, mrl, linewidth=lw, color="k" )
    plt.xlabel( xlabel, fontsize=fs )
    plt.ylabel( "median relative latency", fontsize=fs, rotation="vertical" )
    plt.xticks( xticks, fontsize=fs )
    plt.yticks( yticks, fontsize=fs )
    ax3.xaxis.set_tick_params( direction='in' )
    ax3.yaxis.set_tick_params( direction='in' )
    ax3.xaxis.set_ticklabels( xticklabels )
    ax3.yaxis.set_ticklabels( yticklabels )
    ax3.tick_params( labelsize=fs, axis='both', which='major', pad=7.5 )
    plt.xlim( xlim )
    plt.ylim( ylim )
    ax3.spines['top'].set_color( 'none' )
    ax3.spines['right'].set_color( 'none' )
    
    # SUB PANEL 4
    ax4 = fig.add_subplot( 223 )
    max_val = 35
    xticks_4 = np.arange( max_val, step=5 )
    ax4.plot( params_range, [p['out_of_candidate_rate']*60 for p in performances ], linewidth=lw, color="k" )
    plt.xlabel( xlabel, fontsize=fs )
    plt.ylabel( "out-of-burst detections\n[events/min]", fontsize=fs, rotation="vertical" )
    plt.xticks( xticks, fontsize=fs )
    plt.yticks( xticks_4, fontsize=fs )    
    ax4.xaxis.set_tick_params( direction='in' )
    ax4.yaxis.set_tick_params( direction='in' )
    ax4.xaxis.set_ticklabels( xticklabels )
    ax4.yaxis.set_ticklabels( [ str(x) for x in xticks_4] )
    ax4.tick_params( labelsize=fs, axis='both', which='major', pad=7.5 )
    plt.xlim( xlim )
    plt.ylim( (0, max_val) )
    ax4.spines['top'].set_color( 'none' )
    ax4.spines['right'].set_color( 'none' )

    plt.subplots_adjust( left=0.1, bottom=0.1, right=0.98, top=0.98, wspace=0.4,\
                        hspace=0.5 )


def plot_roc( sensitivity, specificity, title="ROC curve", ticks=[0, .25, .5, .75, 1], lw=1.5, fs=12,\
     lw_ticks=2, padsize=8, figsize=(5, 8), lw_axes=1, fign=1  ):
    '''
    '''
    
    fig, ax = plt.subplots( fign, figsize=figsize )
    true_positive_rate = sensitivity
    false_positive_rate = 1 - specificity
    ax.plot( false_positive_rate, true_positive_rate, color='k', lw=lw )
    plt.xlabel( "False positive rate", fontsize=fs )
    plt.ylabel( "True positive rate", fontsize=fs )
    plt.title( title )
    ax.xaxis.set_tick_params( direction='in', width=lw_ticks )
    ax.yaxis.set_tick_params( direction='in', width=lw_ticks )
    plt.xticks( ticks, fontsize=fs )
    plt.yticks( ticks, fontsize=fs )
    ax.spines['bottom'].set_linewidth(lw_axes)
    ax.spines['top'].set_linewidth(lw_axes)
    ax.spines['left'].set_linewidth(lw_axes)
    ax.spines['right'].set_linewidth(lw_axes)
    ax.tick_params( labelsize=fs, axis='both', which='major', pad=padsize )
#    rc( 'xticks', which='both', linewidth=axes_thickness )
    

def plot_performance_index( fig, index, sel_sleep, sel_run, fs=16, ms=None ):
    '''
    '''
    ax = fig.add_subplot( 111 )
    import seaborn
    colors = seaborn.color_palette( "cubehelix", n_colors=2 )
    ax.plot( np.random.rand(3)/100-.005, index[sel_sleep], '*', color=colors[0], ms=ms, label="sleep", alpha=.5 )
    ax.plot( np.random.rand(3)/100-.005, index[sel_run], 'o', color=colors[1], ms=ms, label="postrun", alpha=.5 )
    ax.plot( 0, np.median(index), '+', color='k', ms=2*ms )
#    plt.xlabel( "no label", color='white' )
    plt.xlim( [-.05, .05] )
    plt.ylim( [0,1] )
    leg = ax.legend( loc='center right', fontsize=14  )
    ticks = [0, .2, .4, .6, .8, 1]
    ax.xaxis.set_ticks( [0] )
    ax.xaxis.set_ticklabels(['0'], color='white' )
    ax.yaxis.set_ticks( ticks )
    ax.yaxis.set_ticklabels( [str(tck) for tck in ticks], fontsize=fs )
    ax.xaxis.set_tick_params( direction='in' )
    ax.yaxis.set_tick_params( direction='in' )
    ax.spines['top'].set_color( 'none' )
    ax.spines['right'].set_color( 'none' )
    for item in leg.legendHandles:
        item.set_visible(False)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color( line.get_color() ) 