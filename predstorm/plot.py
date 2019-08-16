#!/usr/bin/env python
"""
This is the module for producing predstorm plots.

Author: C. Moestl, R. Bailey, IWF Graz, Austria
started May 2019, last update May 2019

Python 3.7

Issues:
- ...

To-dos:
- ...

Future steps:
- ...
"""

import os
import sys  
import copy
import logging
import logging.config
import numpy as np
import pdb
import seaborn as sns
import scipy.signal as signal
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from datetime import datetime, timedelta
from glob import iglob
import json
import urllib

from .data import get_stereo_beacon_data
from .data import get_dscovr_data_all, get_past_dst
from .data import get_time_lag_wrt_earth, getpositions, time_to_num_cat
from .config import plotting as pltcfg

logger = logging.getLogger(__name__)

# =======================================================================================
# --------------------------- PLOTTING FUNCTIONS ----------------------------------------
# =======================================================================================

def plot_solarwind_and_dst_prediction(DSCOVR_data, STEREOA_data, DST_data, DSTPRED_data, dst_label='Dst Temerin & Li 2002', past_days=3.5, future_days=7., verification_mode=False, timestamp=None, **kwargs):
    """
    Plots solar wind variables, past from DSCOVR and future/predicted from STEREO-A.
    Total B-field and Bz (top), solar wind speed (second), particle density (third)
    and Dst (fourth) from Kyoto and model prediction.

    Parameters
    ==========
    DSCOVR_data : list[minute data, hourly data]
        DSCOVR data in different time resolutions.
    STEREOA_data : list[minute data, hourly data]
        STEREO-A data in different time resolutions.
    DST_data : predstorm_module.SatData
        Kyoto Dst
    DSTPRED_data : predstorm_module.SatData
        Dst predicted by PREDSTORM.
    dst_method : str (default='temerin_li')
        Descriptor for Dst method being plotted.
    past_days : float (default=3.5)
        Number of days in the past to plot.
    future_days : float (default=7.)
        Number of days into the future to plot.
    lw : int (default=1)
        Linewidth for plotting functions.
    fs : int (default=11)
        Font size for all text in plot.
    ms : int (default=5)
        Marker size for markers in plot.
    figsize : tuple(float=width, float=height) (default=(14,12))
        Figure size (in inches) for output file.
    verification_mode : bool (default=False)
        If True, verification mode will produce a plot of the predicted Dst
        for model verification purposes.
    timestamp : datetime obj
        Time for 'now' label in plot.

    Returns
    =======
    plt.savefig : .png file
        File saved to XXX
    """

    figsize = kwargs.get('figsize', pltcfg.figsize)
    lw = kwargs.get('lw', pltcfg.lw)
    fs = kwargs.get('fs', pltcfg.fs)
    date_fmt = kwargs.get('date_fmt', pltcfg.date_fmt)
    c_dst = kwargs.get('c_dst', pltcfg.c_dst)
    c_dis = kwargs.get('c_dis', pltcfg.c_dis)
    c_sta = kwargs.get('c_sta', pltcfg.c_sta)
    c_sta_dst = kwargs.get('c_sta_dst', pltcfg.c_sta_dst)
    ms_dst = kwargs.get('c_dst', pltcfg.ms_dst)
    fs_legend = kwargs.get('fs_legend', pltcfg.fs_legend)
    fs_ylabel = kwargs.get('fs_legend', pltcfg.fs_ylabel)
    fs_title = kwargs.get('fs_title', pltcfg.fs_title)

    # Set style:
    sns.set_context(pltcfg.sns_context)
    sns.set_style(pltcfg.sns_style)

    # Make figure object:
    fig=plt.figure(1,figsize=figsize)
    axes = []

    # Set data objects:
    stam, sta = STEREOA_data
    dism, dis = DSCOVR_data
    dst = DST_data
    dst_pred = DSTPRED_data
    text_offset = past_days # days (for 'fast', 'intense', etc.)

    # For the minute data, check which are the intervals to show for STEREO-A until end of plot
    sta_index_future=np.where(np.logical_and(stam['time'] > dism['time'][-1], \
                              stam['time'] < dism['time'][-1]+future_days))[0]

    if timestamp == None:
        timestamp = datetime.utcnow()
    timeutc = mdates.date2num(timestamp)

    plotstart = timeutc - past_days
    plotend = timeutc + future_days

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    # Total B-field and Bz (DSCOVR)
    plt.plot_date(dism['time'], dism['btot'],'-', c=c_dis, label='B total L1', linewidth=lw)
    plt.plot_date(dism['time'], dism['bz'],'-', c=c_dis, alpha=0.5, label='Bz GSM L1', linewidth=lw)

    # STEREO-A minute resolution data with timeshift
    plt.plot_date(stam['time'][sta_index_future], stam['btot'][sta_index_future],
                  '-', c=c_sta, linewidth=lw, label='B STEREO-Ahead')
    plt.plot_date(stam['time'][sta_index_future], stam['bn'][sta_index_future],
                  '-', c=c_sta, alpha=0.5, linewidth=lw, label='Bn RTN STEREO-Ahead')

    # Indicate 0 level for Bz
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)
    plt.ylabel('Magnetic field [nT]',  fontsize=fs_ylabel)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    bplotmax=np.nanmax(np.concatenate((dism['btot'],stam['btot'][sta_index_future])))+5
    bplotmin=np.nanmin(np.concatenate((dism['bz'],stam['bn'][sta_index_future]))-5)

    plt.ylim(bplotmin, bplotmax)

    plt.title('L1 DSCOVR real time solar wind from NOAA SWPC for '+ datetime.strftime(timestamp, "%Y-%m-%d %H:%M")+ ' UT   STEREO-A beacon', fontsize=fs_title)

    # SUBPLOT 2: Solar wind speed
    # ---------------------------
    ax2 = fig.add_subplot(412)
    axes.append(ax2)

    # Plot solar wind speed (DSCOVR):
    plt.plot_date(dism['time'], dism['speed'],'-', c=c_dis, label='speed L1',linewidth=lw)
    plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fs_ylabel)

    # Plot STEREO-A data with timeshift and savgol filter
    plt.plot_date(stam['time'][sta_index_future],signal.savgol_filter(stam['speed'][sta_index_future],11,1),'-', 
                  c=c_sta, linewidth=lw, label='speed STEREO-Ahead')

    # Add speed levels:
    pltcfg.plot_speed_lines(xlims=[plotstart, plotend])

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    vplotmax=np.nanmax(np.concatenate((dism['speed'],signal.savgol_filter(stam['speed'][sta_index_future],11,1))))+100
    vplotmin=np.nanmin(np.concatenate((dism['speed'],signal.savgol_filter(stam['speed'][sta_index_future],11,1)))-50)
    plt.ylim(vplotmin, vplotmax)

    plt.annotate('now', xy=(timeutc,vplotmax-(vplotmax-vplotmin)*0.25), xytext=(timeutc+0.05,vplotmax-(vplotmax-vplotmin)*0.25), color='k', fontsize=14)

    # SUBPLOT 3: Solar wind density
    # -----------------------------
    ax3 = fig.add_subplot(413)
    axes.append(ax3)

    # Plot solar wind density:
    plt.plot_date(dism['time'], dism['density'],'-k', label='density L1',linewidth=lw)
    plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fs_ylabel)
    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    plt.ylim([0,np.nanmax(np.nanmax(np.concatenate((dism['density'],stam['density'][sta_index_future])))+10)])

    #plot STEREO-A data with timeshift and savgol filter
    plt.plot_date(stam['time'][sta_index_future], signal.savgol_filter(stam['density'][sta_index_future],5,1),
                  '-', c=c_sta, linewidth=lw, label='density STEREO-Ahead')

    # SUBPLOT 4: Actual and predicted Dst
    # -----------------------------------
    ax4 = fig.add_subplot(414)
    axes.append(ax4)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst['time'], dst['dst'],'o', c=c_dst, label='Dst observed',markersize=ms_dst)
    plt.ylabel('Dst [nT]', fontsize=fs_ylabel)

    dstplotmax = np.nanmax(np.concatenate((dst['dst'], dst_pred['dst'])))+20
    dstplotmin = np.nanmin(np.concatenate((dst['dst'], dst_pred['dst'])))-20

    if dstplotmin > -100:       # Low activity (normal)
        plt.ylim([-100,np.nanmax(dst['dst'])+20])
    else:                       # High activity
        plt.ylim([dstplotmin, dstplotmax])

    # Plot predicted Dst
    if not verification_mode:
        plt.plot_date(dst_pred['time'], dst_pred['dst'], '-', c=c_sta_dst, label=dst_label, markersize=3, linewidth=1)
        # Add generic error bars of +/-15 nT:
        error=15
        plt.fill_between(dst_pred['time'], dst_pred['dst']-error, dst_pred['dst']+error, alpha=0.2,
                         label='Error for high speed streams')
    else:
        #load saved data l prefix is for loaded - WARNING This will crash if called right now
        [timenowb, sta_ptime, sta_vr, sta_btime, sta_btot, sta_br,sta_bt, sta_bn, rbtime_num, rbtot, rbzgsm, rptime_num, rpv, rpn, lrdst_time, lrdst, lcom_time, ldst_burton, ldst_obrien,ldst_temerin_li]=pickle.load(open(verify_filename,'rb') )
        plt.plot_date(lcom_time, ldst_burton,'-b', label='Forecast Dst Burton et al. 1975',markersize=3, linewidth=1)
        plt.plot_date(lcom_time, ldst_obrien,'-r', label='Forecast Dst OBrien & McPherron 2000',markersize=3, linewidth=1)

    # Label plot with geomagnetic storm levels
    pltcfg.plot_dst_activity_lines(xlims=[plotstart, plotend])

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.legend(loc=2,ncol=4,fontsize=fs_legend)

        # Dates on x-axes:
        myformat = mdates.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(myformat)

        # Vertical line for NOW:
        ax.plot_date([timeutc,timeutc],[-2000,2000],'-k', linewidth=2)

    # Liability text:
    pltcfg.group_info_text()
    pltcfg.liability_text()

    #save plot
    if not verification_mode:
        plot_label = 'realtime'
    else:
        plot_label = 'verify'

    filename = os.path.join('results','predstorm_v1_{}_stereo_a_plot_{}.png'.format(
                            plot_label, datetime.strftime(timestamp, "%Y-%m-%d-%H_%M")))
    filename_eps = filename.replace('png', 'eps')

    if not verification_mode:
        plt.savefig('predstorm_real.png')
        print('Real-time plot saved as predstorm_real.png!')

    #if not server: # Just plot and exit
    #    plt.show()
    #    sys.exit()
    plt.savefig(filename)
    logger.info('Plot saved as png:\n'+ filename)


def plot_stereo_dscovr_comparison(stam, dism, dst, timestamp=None, look_back=20, outfile=None, **kwargs):
    """Plots the last days of STEREO-A and DSCOVR data for comparison alongside
    the predicted and real Dst.

    Parameters
    ==========
    stam : predstorm.SatData
        Object containing minute STEREO-A data
    dism : predstorm.SatData
        Object containing minute DSCOVR data.
    dst : predstorm.SatData
        Object containing Kyoto Dst data.
    timestamp : datetime obj
        Time for last datapoint in plot.
    look_back : float (default=20)
        Number of days in the past to plot.
    **kwargs : ...
        See config.plotting for variables that can be tweaked.

    Returns
    =======
    plt.savefig : .png file
        File saved to XXX
    """

    if timestamp == None:
        timestamp = datetime.utcnow()

    if outfile == None:
        outfile = 'sta_dsc_comparison_{}.png'.format(datetime.strftime(timestamp, "%Y-%m-%dT%H:%M"))

    figsize = kwargs.get('figsize', pltcfg.figsize)
    lw = kwargs.get('lw', pltcfg.lw)
    fs = kwargs.get('fs', pltcfg.fs)
    date_fmt = kwargs.get('date_fmt', pltcfg.date_fmt)
    c_dst = kwargs.get('c_dst', pltcfg.c_dst)
    c_dis = kwargs.get('c_dis', pltcfg.c_dis)
    c_sta = kwargs.get('c_sta', pltcfg.c_sta)
    c_sta_dst = kwargs.get('c_sta_dst', pltcfg.c_sta_dst)
    ms_dst = kwargs.get('c_dst', pltcfg.ms_dst)
    fs_legend = kwargs.get('fs_legend', pltcfg.fs_legend)
    fs_ylabel = kwargs.get('fs_legend', pltcfg.fs_ylabel)

    # READ DATA:
    # ----------
    # TODO: It would be faster to read archived hourly data rather than interped minute data...
    logger.info("plot_stereo_dscovr_comparison: Reading satellite data")
    # Get estimate of time diff:
    stam.shift_time_to_L1()
    sta = stam.make_hourly_data()
    sta.interp_nans()

    dis = dism.make_hourly_data()
    dis.interp_nans()

    # CALCULATE PREDICTED DST:
    # ------------------------
    sta.convert_RTN_to_GSE().convert_GSE_to_GSM()
    dst_pred = sta.make_dst_prediction()

    # PLOT:
    # -----
    # Set style:
    sns.set_context(pltcfg.sns_context)
    sns.set_style(pltcfg.sns_style)

    plotstart = timestamp - timedelta(days=look_back)
    plotend = timestamp

    # Make figure object:
    fig = plt.figure(1,figsize=figsize)
    axes = []

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    plt.plot_date(dis['time'], dis['bz'], '-', c=c_dis, linewidth=lw, label='DSCOVR')
    plt.plot_date(sta['time'], sta['bz'], '-', c=c_sta, linewidth=lw, label='STEREO-A')

    # Indicate 0 level for Bz
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)
    plt.ylabel('Magnetic field Bz [nT]',  fontsize=fs_ylabel)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    bplotmax=np.nanmax(np.concatenate((dis['bz'], sta['bz'])))+5
    bplotmin=np.nanmin(np.concatenate((dis['bz'], sta['bz'])))-5
    plt.ylim(bplotmin, bplotmax)
    plt.legend(loc=2,ncol=4,fontsize=fs_legend)

    plt.title('DSCOVR and STEREO-A solar wind projected to L1 for '+ datetime.strftime(timestamp, "%Y-%m-%d %H:%M")+ ' UT', fontsize=16)

    # SUBPLOT 2: Solar wind speed
    # ---------------------------
    ax2 = fig.add_subplot(412)
    axes.append(ax2)

    plt.plot_date(dis['time'], dis['speed'], '-', c=c_dis, linewidth=lw)
    plt.plot_date(sta['time'], sta['speed'], '-', c=c_sta, linewidth=lw)

    plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fs_ylabel)

    # Add speed levels:
    pltcfg.plot_speed_lines(xlims=[plotstart, plotend])

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    vplotmax=np.nanmax(np.concatenate((dis['speed'], sta['speed'])))+100
    vplotmin=np.nanmin(np.concatenate((dis['speed'], sta['speed'])))-50
    plt.ylim(vplotmin, vplotmax)

    # SUBPLOT 3: Solar wind density
    # -----------------------------
    ax3 = fig.add_subplot(413)
    axes.append(ax3)

    # Plot solar wind density:
    plt.plot_date(dis['time'], dis['density'], '-', c=c_dis, linewidth=lw)
    plt.plot_date(sta['time'], sta['density'], '-', c=c_sta, linewidth=lw)

    plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fs_ylabel)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    plt.ylim([0, np.nanmax(np.nanmax(np.concatenate((dis['density'], sta['density'])))+10)])

    # SUBPLOT 4: Actual and predicted Dst
    # -----------------------------------
    ax4 = fig.add_subplot(414)
    axes.append(ax4)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst['time'], dst['dst'],'o', c=c_dst, label='Observed Dst', ms=ms_dst)
    plt.plot_date(sta['time'], dst_pred['dst'],'-', c=c_sta_dst, label='Predicted Dst', lw=lw)
    # Add generic error bars of +/-15 nT:
    error=15
    plt.fill_between(sta['time'], dst_pred['dst']-error, dst_pred['dst']+error, alpha=0.2,
                     label='Error for high speed streams')

    # Label plot with geomagnetic storm levels
    pltcfg.plot_dst_activity_lines(xlims=[plotstart, plotend])

    dstplotmin = -10 + np.nanmin(np.nanmin(np.concatenate((dst['dst'], dst_pred['dst']))))
    dstplotmax = 10 + np.nanmax(np.nanmax(np.concatenate((dst['dst'], dst_pred['dst']))))
    plt.ylim([dstplotmin, dstplotmax])
    plt.legend(loc=2,ncol=4,fontsize=fs_legend)

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)

        # Dates on x-axes:
        myformat = mdates.DateFormatter('%b %d %Hh')
        ax.xaxis.set_major_formatter(myformat)

    plt.savefig(outfile)
    logger.info("Plot saved as {}".format(outfile))
    plt.close()

    return


def plot_dst_comparison(stam, dism, dst, timestamp=None, look_back=20, outfile=None, **kwargs):
    """Plots the last days of STEREO-A and DSCOVR data for comparison alongside
    the predicted and real Dst.

    Parameters
    ==========
    stam : predstorm.SatData
        Object containing minute STEREO-A data
    dism : predstorm.SatData
        Object containing minute DSCOVR data.
    dst : predstorm.SatData
        Object containing hourly Kyoto Dst data.
    timestamp : datetime obj
        Time for last datapoint in plot.
    look_back : float (default=20)
        Number of days in the past to plot.
    **kwargs : ...
        See config.plotting for variables that can be tweaked.

    Returns
    =======
    plt.savefig : .png file
        File saved to XXX
        """

    if timestamp == None:
        timestamp = datetime.utcnow()

    if outfile == None:
        outfile = 'dst_comparison_{}.png'.format(datetime.strftime(timestamp, "%Y-%m-%dT%H:%M"))

    figsize = kwargs.get('figsize', pltcfg.figsize)
    lw = kwargs.get('lw', pltcfg.lw)
    fs = kwargs.get('fs', pltcfg.fs)
    date_fmt = kwargs.get('date_fmt', pltcfg.date_fmt)
    c_dst = kwargs.get('c_dst', pltcfg.c_dst)
    c_dis = kwargs.get('c_dis', pltcfg.c_dis)
    c_dis_dst = kwargs.get('c_dis_dst', pltcfg.c_dis_dst)
    c_sta_dst = kwargs.get('c_sta_dst', pltcfg.c_sta_dst)
    c_sta = kwargs.get('c_sta', pltcfg.c_sta)
    ms_dst = kwargs.get('c_dst', pltcfg.ms_dst)
    fs_legend = kwargs.get('fs_legend', pltcfg.fs_legend)
    fs_title = kwargs.get('fs_title', pltcfg.fs_title)

    # PREPARE DATA:
    # -------------
    # TODO: It would be faster to read archived hourly data rather than interped minute data...
    logger.info("plot_dst_comparison: Preparing satellite data")
    # Correct for STEREO-A position:
    stam.shift_time_to_L1()
    sta = stam.make_hourly_data()
    sta = sta.cut(starttime=timestamp-timedelta(days=look_back), endtime=timestamp).interp_nans()

    dis = dism.make_hourly_data()
    dis.interp_nans()

    # CALCULATE PREDICTED DST:
    # ------------------------
    sta.convert_RTN_to_GSE().convert_GSE_to_GSM()

    dst_h = dst.interp_to_time(sta['time'])
    dis = dis.interp_to_time(sta['time'])
    dst_sta = sta.make_dst_prediction()
    dst_dis = dis.make_dst_prediction()

    # PLOT:
    # -----
    # Set style:
    sns.set_context(pltcfg.sns_context)
    sns.set_style(pltcfg.sns_style)

    plotstart = timestamp - timedelta(days=look_back)
    plotend = timestamp

    # Make figure object:
    fig = plt.figure(1, figsize=figsize)
    axes = []

    # SUBPLOT 1: Actual and predicted Dst
    # -----------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst['time'], dst['dst'], 'o', c=c_dst, label='Observed Dst', ms=ms_dst)
    plt.plot_date(sta['time'], dst_sta['dst'],'-', c=c_sta_dst, label='Predicted Dst (STEREO-A)', linewidth=lw)
    plt.plot_date(dis['time'], dst_dis['dst'],'-', c=c_dis_dst, label='Predicted Dst (DSCOVR)', linewidth=lw)
    # Add generic error bars of +/-15 nT:
    error=15
    plt.fill_between(sta['time'], dst_sta['dst']-error, dst_sta['dst']+error, facecolor=c_sta_dst, alpha=0.2, label='Error')
    plt.fill_between(dis['time'], dst_dis['dst']-error, dst_dis['dst']+error, facecolor=c_dis_dst, alpha=0.2, label='Error')

    # Label plot with geomagnetic storm levels
    pltcfg.plot_dst_activity_lines(xlims=[plotstart, plotend])

    dstplotmin = -10 + np.nanmin(np.nanmin(np.concatenate((dst_sta['dst'], dst_dis['dst']))))
    dstplotmax = 10 + np.nanmax(np.nanmax(np.concatenate((dst_sta['dst'], dst_dis['dst']))))
    plt.ylim([dstplotmin, dstplotmax])
    plt.title("Dst(real) vs Dst(predicted)", fontsize=fs_title)

    # SUBPLOT 2: Actual vs predicted Dst STEREO
    # -----------------------------------------
    diff_sta = dst_h['dst'] - dst_sta['dst']
    diff_dis = dst_h['dst'] - dst_dis['dst']
    if np.nanmax((np.abs(dstplotmin), dstplotmax)) > 50:
        maxval = np.nanmax((np.abs(dstplotmin), dstplotmax))
    else:
        maxval = 50.
    ax2 = fig.add_subplot(412)
    axes.append(ax2)

    # Observed Dst Kyoto (past):
    gradient_fill(sta['time'], dst_sta['dst']-dst_h['dst'], maxval=maxval, ls='-', c='k', label='Dst(Kyoto) - Dst(STEREO-A-pred)', ms=0, lw=lw)

    # SUBPLOT 3: Actual vs predicted Dst DSCOVR
    # -----------------------------------------
    ax3 = fig.add_subplot(413)
    axes.append(ax3)

    # Observed Dst Kyoto (past):
    gradient_fill(dis['time'], dst_dis['dst']-dst_h['dst'], maxval=maxval, ls='-', c='k', label='Dst(Kyoto) - Dst(DSCOVR-pred)', ms=0, lw=lw)

    # SUBPLOT 3: Predicted vs predicted Dst
    # -------------------------------------
    ax4 = fig.add_subplot(414)
    axes.append(ax4)

    # Observed Dst Kyoto (past):
    gradient_fill(dis['time'], dst_dis['dst']-dst_sta['dst'], maxval=maxval, ls='-', c='k', label='Dst(DSCOVR-pred) - Dst(STEREO-A-pred)', ms=0, lw=lw)

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.legend(loc=2, ncol=5, fontsize=fs_legend)

        # Dates on x-axes:
        myformat = mdates.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(myformat)

    plt.savefig(outfile)
    logger.info("Plot saved as {}".format(outfile))
    plt.close()

    return


def plot_indices(dism, timestamp=None, look_back=20, outfile=None, **kwargs):
    """
    Plots solar wind variables, past from DSCOVR and future/predicted from STEREO-A.
    Total B-field and Bz (top), solar wind speed (second), particle density (third)
    and Dst (fourth) from Kyoto and model prediction.

    Parameters
    ==========
    dism : predstorm.SatData
        Object containing minute satellite L1 data.
    timestamp : datetime obj
        Time for last datapoint in plot.
    look_back : float (default=20)
        Number of days in the past to plot.
    **kwargs : ...
        See config.plotting for variables that can be tweaked.

    Returns
    =======
    plt.savefig : .png file
        File saved to XXX
    """

    if timestamp == None:
        timestamp = datetime.utcnow()

    if outfile == None:
        outfile = 'indices_{}.png'.format(datetime.strftime(timestamp, "%Y-%m-%dT%H:%M"))

    figsize = kwargs.get('figsize', pltcfg.figsize)
    lw = kwargs.get('lw', pltcfg.lw)
    fs = kwargs.get('fs', pltcfg.fs)
    date_fmt = kwargs.get('date_fmt', pltcfg.date_fmt)
    c_dst = kwargs.get('c_dst', pltcfg.c_dst)
    c_dis = kwargs.get('c_dis', pltcfg.c_dis)
    c_dis_dst = kwargs.get('c_dis', pltcfg.c_dis_dst)
    c_ec = kwargs.get('c_dis', pltcfg.c_ec)
    c_kp = kwargs.get('c_dis', pltcfg.c_kp)
    c_aurora = kwargs.get('c_dis', pltcfg.c_aurora)
    ms_dst = kwargs.get('c_dst', pltcfg.ms_dst)
    fs_legend = kwargs.get('fs_legend', pltcfg.fs_legend)
    fs_ylabel = kwargs.get('fs_legend', pltcfg.fs_ylabel)
    fs_title = kwargs.get('fs_title', pltcfg.fs_title)

    # READ DATA:
    # ----------
    # TODO: It would be faster to read archived hourly data rather than interped minute data...
    logger.info("plot_indices: Preparing satellite data")
    # Get estimate of time diff:

    # Read DSCOVR data:
    dis = dism.make_hourly_data()
    dis.interp_nans()
    dst = get_past_dst(filepath="data/dstarchive/WWW_dstae00016185.dat",
                       starttime=timestamp-timedelta(days=look_back),
                       endtime=timestamp)

    # Calculate Dst from prediction:
    dst_dis = dis.make_dst_prediction()
    # Kp:
    kp_dis = dis.make_kp_prediction()
    # Newell coupling ec:
    ec_dis = dis.get_newell_coupling()
    # Aurora power:
    aurora_dis = dis.make_aurora_power_prediction()

    # PLOT:
    # -----
    # Set style:
    sns.set_context(pltcfg.sns_context)
    sns.set_style(pltcfg.sns_style)

    # Make figure object:
    fig = plt.figure(1, figsize=figsize)
    axes = []

    if timestamp == None:
        timestamp = datetime.utcnow()
    timeutc = mdates.date2num(timestamp)

    plotstart = timestamp - timedelta(days=look_back)
    plotend = timestamp

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(511)
    axes.append(ax1)

    # Total B-field and Bz (DSCOVR)
    plt.plot_date(dism['time'], dism['btot'],'-', c=c_dis, label='B total L1', linewidth=lw)
    plt.plot_date(dism['time'], dism['bz'],'-', c=c_dis, alpha=0.5, label='Bz GSM L1', linewidth=lw)

    # Indicate 0 level for Bz
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)
    plt.ylabel('Magnetic field [nT]',  fontsize=fs_ylabel)
    plt.ylim(np.nanmin(dism['bz'])-5, np.nanmax(dism['btot'])+5)

    plt.title('DSCOVR data and derived indices for {}'.format(datetime.strftime(timestamp, "%Y-%m-%d %H:%M")), fontsize=fs_title)

    # SUBPLOT 2: Actual and predicted Dst
    # -----------------------------------
    ax3 = fig.add_subplot(512)
    axes.append(ax3)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst['time'], dst['dst'],'o', c=c_dst, label='Observed Dst', markersize=ms_dst)
    plt.ylabel('Dst [nT]', fontsize=fs_ylabel)

    dstplotmax = np.nanmax(np.concatenate((dst['dst'], dst_dis['dst'])))+20
    dstplotmin = np.nanmin(np.concatenate((dst['dst'], dst_dis['dst'])))-20
    plt.ylim([dstplotmin, dstplotmax])

    plt.plot_date(dst_dis['time'], dst_dis['dst'],'-', c=c_dis_dst, label='Predicted Dst (DSCOVR)', linewidth=lw)
    error=15
    plt.fill_between(dst_dis['time'], dst_dis['dst']-error, dst_dis['dst']+error, facecolor=c_dis_dst, alpha=0.2, label='Error')

    # Label plot with geomagnetic storm levels
    pltcfg.plot_dst_activity_lines(xlims=[plotstart, plotend])

    # SUBPLOT 3: kp
    # -----------------------------
    ax5 = fig.add_subplot(513)
    axes.append(ax5)

    # Plot Newell coupling (DSCOVR):
    plt.plot_date(kp_dis['time'], kp_dis['kp'],'-', c=c_kp, linewidth=lw)
    plt.ylabel('$\mathregular{k_p}$', fontsize=fs_ylabel)
    plt.ylim([0., 10.])

    # SUBPLOT 4: Newell Coupling
    # --------------------------
    ax2 = fig.add_subplot(514)
    axes.append(ax2)

    # Plot Newell coupling (DSCOVR):
    plt.plot_date(ec_dis['time'], ec_dis['ec'],'-', c=c_ec, linewidth=lw)
    plt.ylabel('Newell coupling $ec$', fontsize=fs_ylabel)
    plt.ylim([0., np.nanmax(ec_dis['ec'])*1.1])

    # SUBPLOT 5: Aurora power
    # -----------------------
    ax4 = fig.add_subplot(515)
    axes.append(ax4)

    # Plot Newell coupling (DSCOVR):
    plt.plot_date(aurora_dis['time'], aurora_dis['aurora'],'-', c=c_aurora, linewidth=lw)
    plt.ylabel('Aurora power [?]', fontsize=fs_ylabel)
    plt.ylim([0., np.nanmax(aurora_dis['aurora'])*1.1])

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.legend(loc=2,ncol=4,fontsize=fs_legend)

        # Dates on x-axes:
        myformat = mdates.DateFormatter(date_fmt)
        ax.xaxis.set_major_formatter(myformat)

    plt.savefig(outfile)
    logger.info('Plot saved as png:\n'+ outfile)

    plt.close()
    return


# =======================================================================================
# --------------------------- EXTRA FUNCTIONS -------------------------------------------
# =======================================================================================

def gradient_fill(x, y, ax=None, maxval=None, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.
    Adapted from https://stackoverflow.com/a/29331211.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    maxval : float
        Maximum value (x/-) in plots for gradient scaling.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot_date(x, y, **kwargs)

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha
    maxval if maxval is None else maxval

    z_up, z_down = np.empty((100, 1, 4), dtype=float), np.empty((100, 1, 4), dtype=float)
    rgb_b = mcolors.colorConverter.to_rgb('b')
    rgb_r = mcolors.colorConverter.to_rgb('r')
    z_down[:,:,:3] = rgb_r
    z_down[:,:,-1] = np.linspace(0, alpha, 100)[:,None]
    z_up[:,:,:3] = rgb_b
    z_up[:,:,-1] = np.linspace(0, alpha, 100)[:,None]

    # Fill above zero:
    xmin, xmax, ymin, ymax = x.min(), x.max(), 0., maxval
    im = ax.imshow(z_up, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    # Fill below zero:
    xmin, xmax, ymin, ymax = x.min(), x.max(), -maxval, 0.
    im = ax.imshow(z_down, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='upper', zorder=zorder)

    xy = np.column_stack([x, y])
    #xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    xy = np.vstack([[xmin, 0.], xy, [xmax, 0.], [xmin, 0.]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    ax.autoscale(True)
    return line, im


def plot_all(timestamp=None, plotdir="plots", download=True):
    """Makes plots of a time range ending with timestamp using all functions."""

    import predstorm as ps

    if not os.path.isdir(plotdir):
        os.mkdir(plotdir)

    if timestamp == None:
        timestamp = datetime.utcnow()
    look_back = 30
    lag_L1, lag_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
    est_timelag = lag_L1 + lag_r

    logger = ps.init_logging(verbose=True)
    logger.info("Plotting all plots...")

    # STEREO DATA
    stam = ps.get_stereoa_data_beacon(starttime=timestamp-timedelta(days=look_back+est_timelag+0.5), 
                                      endtime=timestamp)
    stam.load_positions('data/positions/STEREOA-pred_20070101-20250101_HEEQ_6h.p')
    # DSCOVR DATA
    dism = ps.get_dscovr_data_all(P_filepath="data/dscovrarchive/*",
                                  M_filepath="data/dscovrarchive/*",
                                  starttime=timestamp-timedelta(days=look_back),
                                  endtime=timestamp, download=download)
    dism.load_positions('data/positions/Earth_20000101-20250101_HEEQ_6h.p', l1_corr=True)
    # KYOTO DST
    dst = ps.get_past_dst(filepath="data/dstarchive/WWW_dstae00019594.dat",
                          starttime=timestamp-timedelta(days=look_back),
                          endtime=timestamp)

    logger.info("\n-------------------------\nDst comparison\n-------------------------")
    outfile = os.path.join(plotdir, "dst_prediction_{}day_plot.png".format(look_back))
    plot_dst_comparison(stam, dism, dst, timestamp=timestamp, look_back=look_back, outfile=outfile)

    logger.info("\n-------------------------\nSTEREO-A vs DSCOVR\n-------------------------")
    outfile = os.path.join(plotdir, "stereoa_vs_dscovr_{}day_plot.png".format(look_back))
    plot_stereo_dscovr_comparison(stam, dism, dst, timestamp=timestamp, look_back=look_back, outfile=outfile)

    logger.info("\n-------------------------\nPredicted indices\n-------------------------")
    outfile = os.path.join(plotdir, "indices_{}day_plot.png".format(look_back))
    plot_indices(dism, timestamp=timestamp, look_back=look_back, outfile=outfile)


if __name__ == '__main__':

    plot_all()





