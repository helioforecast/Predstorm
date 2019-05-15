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
import scipy
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from glob import iglob
import json
import urllib
try:
	import IPython
except:
	pass

from predstorm_module import read_stereoa_data_beacon, getpositions, time_to_num_cat
from predstorm_module import get_dscovr_data_all, get_past_dst, make_dst_from_wind
from predstorm_module import get_time_lag_wrt_earth


# =======================================================================================
# ------------------------------------ FUNCTIONS ----------------------------------------
# =======================================================================================

def plot_stereo_dscovr_comparison(timestamp=None, look_back=30, lw=1, fs=11, ms=5, figsize=(14,12)):
    """Plots the last days of STEREO-A and DSCOVR data for comparison alongside
    the predicted and real Dst."""

    if timestamp == None:
        timestamp = datetime.utcnow()
    timestamp = datetime.strptime("2019-05-05", "%Y-%m-%d")
    lw, fs, ms = 1, 11, 5
    figsize = (14,12)
    look_back = 20 # 60 # days

    # READ DATA:
    # ----------
    # TODO: It would be faster to read archived hourly data rather than interped minute data...
    logger.info("plot_60_day_comparison: Reading satellite data")
    # Get estimate of time diff:
    lag_L1, lag_r = get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
    est_timelag = lag_L1 + lag_r
    stam = read_stereoa_data_beacon(starttime=timestamp-timedelta(days=look_back+est_timelag), 
    								endtime=timestamp)
    sta = stam.make_hourly_data()
    sta.interp_nans()
    dism = get_dscovr_data_all(P_filepath="data/dscovrarchive/*",
                               M_filepath="data/dscovrarchive/*",
                               starttime=timestamp-timedelta(days=look_back),
                               endtime=timestamp)
    dis = dism.make_hourly_data()
    dis.interp_nans()
    dst = get_past_dst(filepath="data/dstarchive/WWW_dstae00016185.dat",
                       starttime=timestamp-timedelta(days=look_back),
                       endtime=timestamp)

    # CORRECT FOR STEREO-A POSITION:
    # ------------------------------
    sta.shift_time_to_L1()

    # CALCULATE PREDICTED DST:
    # ------------------------
    pos=getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
    pos_time_num=time_to_num_cat(pos.time)
    sta.convert_RTN_to_GSE(pos.sta, pos_time_num).convert_GSE_to_GSM()
    [dst_burton, dst_obrien, dst_temerin_li] = make_dst_from_wind(sta['btot'],sta['bx'],sta['by'],sta['bz'],sta['speed'],sta['speed'],sta['density'],sta['time'])

    # PLOT:
    # -----
    # Set style:
    sns.set_context("talk")
    sns.set_style("darkgrid")

    plotstart = timestamp - timedelta(days=look_back)
    plotend = timestamp

    # Colours for datasets:
    c_dis = 'k'
    c_sta = 'r'

    # Make figure object:
    fig=plt.figure(1,figsize=figsize)
    axes = []

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    plt.plot_date(dis['time'], dis['bz'], '-'+c_dis, linewidth=lw, label='DSCOVR')
    plt.plot_date(sta['time'], sta['bz'], '-'+c_sta, linewidth=lw, label='STEREO-A')

    # Indicate 0 level for Bz
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)
    plt.ylabel('Magnetic field Bz [nT]',  fontsize=fs+2)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    bplotmax=np.nanmax(np.concatenate((dis['bz'], sta['bz'])))+5
    bplotmin=np.nanmin(np.concatenate((dis['bz'], sta['bz'])))-5
    plt.ylim(bplotmin, bplotmax)
    plt.legend(loc=2,ncol=4,fontsize=fs-2)

    plt.title('DSCOVR and STEREO-A solar wind projected to L1 for '+ datetime.strftime(timestamp, "%Y-%m-%d %H:%M")+ ' UT', fontsize=16)

    # SUBPLOT 2: Solar wind speed
    # ---------------------------
    ax2 = fig.add_subplot(412)
    axes.append(ax2)

    plt.plot_date(dis['time'], dis['speed'], '-'+c_dis, linewidth=lw)
    plt.plot_date(sta['time'], sta['speed'], '-'+c_sta, linewidth=lw)

    plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fs+2)

    # Add speed levels:
    for hline, linetext in zip([400, 800], ['slow', 'fast']):
        plt.plot_date([dis['time'][0], dis['time'][-1]],
                    [hline, hline],'--k', alpha=0.3, linewidth=1)
        plt.annotate(linetext,xy=(mdates.date2num(timestamp)-look_back,hline),
                     xytext=(mdates.date2num(timestamp)-look_back,hline),
                     color='k', fontsize=10)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    vplotmax=np.nanmax(np.concatenate((dis['speed'], sta['speed'])))+100
    vplotmin=np.nanmin(np.concatenate((dis['speed'], sta['speed'])))-50
    plt.ylim(vplotmin, vplotmax)

    # SUBPLOT 3: Solar wind density
    # -----------------------------
    ax3 = fig.add_subplot(413)
    axes.append(ax3)

    # Plot solar wind density:
    plt.plot_date(dis['time'], dis['density'], '-'+c_dis, linewidth=lw)
    plt.plot_date(sta['time'], sta['density'], '-'+c_sta, linewidth=lw)

    plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fs+2)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    plt.ylim([0, np.nanmax(np.nanmax(np.concatenate((dis['density'], sta['density'])))+10)])

    # SUBPLOT 4: Actual and predicted Dst
    # -----------------------------------
    ax4 = fig.add_subplot(414)
    axes.append(ax4)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst['time'], dst['dst'],'ko', label='Observed Dst',markersize=4)
    plt.plot_date(sta['time'], dst_temerin_li,'-b', label='Predicted Dst', markersize=3, linewidth=1)
    # Add generic error bars of +/-15 nT:
    error=15
    plt.fill_between(sta['time'], dst_temerin_li-error, dst_temerin_li+error, alpha=0.2,
                     label='Error for high speed streams')

    # Label plot with geomagnetic storm levels
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.3, linewidth=1)
    for hline, linetext in zip([-50, -100, -250], ['moderate', 'intense', 'super-storm']):
        plt.plot_date([plotstart,plotend], [hline,hline],'--k', alpha=0.3, linewidth=1)
        plt.annotate(linetext,xy=(mdates.date2num(timestamp)-look_back, hline+2),
                     xytext=(mdates.date2num(timestamp)-look_back, hline+2), color='k', fontsize=10)

    dstplotmin = -10 + np.nanmin(np.nanmin(np.concatenate((dst['dst'], dst_temerin_li))))
    dstplotmax = 10 + np.nanmax(np.nanmax(np.concatenate((dst['dst'], dst_temerin_li))))
    plt.ylim([dstplotmin, dstplotmax])
    plt.legend(loc=2,ncol=4,fontsize=fs-2)

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)

        # Dates on x-axes:
        myformat = mdates.DateFormatter('%b %d %Hh')
        ax.xaxis.set_major_formatter(myformat)

    plt.savefig("results/sta_dsc_{}day_plot.png".format(look_back))


if __name__ == '__main__':

	logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
	logger = logging.getLogger(__name__)

	plot_stereo_dscovr_comparison()




