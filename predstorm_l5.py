#!/usr/bin/env python
"""
PREDSTORM is a real time solar wind and magnetic storm forecasting python package
using time-shifted data from a spacecraft east of the Sun-Earth line.
Currently STEREO-A beacon data is used, but it is also suited for using data from a
possible future L5 mission or interplanetary CubeSats.

Author: C. Moestl, R. Bailey, IWF Graz, Austria
twitter @chrisoutofspace, https://github.com/cmoestl
started April 2018, last update April 2019

python 3.7
packages needed to add to anaconda installation: sunpy, cdflib (https://github.com/MAVENSDC/cdflib)


issues:
- predstorm_L5_stereob_errors.py needs to be rewritten for new data structures (recarrays)
- rewrite verification in predstorm_l5.py
- use astropy instead of ephem in predstorm_module.py


current status:
The code works with STEREO-A beacon and DSCOVR data and downloads STEREO-A beacon files
into the sta_beacon directory 14 days prior to current time
tested for correctly handling missing PLASTIC files

things to add:
- add verification mode
- add error bars for the Temerin/Li Dst model with 1 and 2 sigma
- fill data gaps from STEREO-A beacon data with reasonable Bz fluctuations etc.
  based on a assessment of errors with the ... stereob_errors program
- add timeshifts from L1 to Earth
- add approximate levels of Dst for each location to see the aurora (depends on season)
  taken from correlations of ovation prime, SuomiNPP data in NASA worldview and Dst
- check coordinate conversions again, GSE to GSM is ok
- deal with CMEs at STEREO, because systematically degrades prediction results
- add metrics ROC for verification etc.
- proper status/debugging logging system

future larger steps:
(1) add the semi-supervised learning algorithm from the predstorm_L1 program; e.g. with older
	STEREO data additionally, so a #combined L1/L5 forecast
	most important: implement pattern recognition for STEREO-A streams,
	and link this to the most probably outcome days later at L1
	train with STB data around the location where STA is at the moment
(2) fundamental issue: by what amount is Bz stable for HSS from L5 to L1? are there big changes?
is Bz higher for specific locations with respect to the HCS and the solar equator?
temporal and spatial coherence of Bz
(3) probabilities for magnetic storm magnitude, probabilities for aurora for many locations

-----------------
MIT LICENSE
Copyright 2019, Christian Moestl
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:
The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


##########################################################################################
######################################### CODE START #####################################
##########################################################################################

import os
import sys
import getopt

# READ INPUT OPTIONS FROM COMMAND LINE
argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"hv",["server", "help", "historic=", "verbose"])

server = False
if "--server" in [o for o, v in opts]:
    server = True
    print("In server mode!")

import matplotlib
if server:
    matplotlib.use('Agg') # important for server version, otherwise error when making figures
else:
    matplotlib.use('Qt5Agg') # figures are shown on mac
try:
   import IPython
except:
   pass

from datetime import datetime, timedelta
import json
import logging
import logging.config
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pdb
import pickle
import scipy.io
import seaborn as sns
import sunpy.time
import urllib
from scipy.signal import savgol_filter

from predstorm_module import get_dscovr_data_real, get_dscovr_data_all
from predstorm_module import download_stereoa_data_beacon, read_stereoa_data_beacon
from predstorm_module import time_to_num_cat
from predstorm_module import converttime
from predstorm_module import make_dst_from_wind
from predstorm_module import make_kp_from_wind
from predstorm_module import make_aurora_power_from_wind
from predstorm_module import getpositions
from predstorm_module import convert_GSE_to_GSM
from predstorm_module import sphere2cart
from predstorm_module import convert_RTN_to_GSE_sta_l1
from predstorm_module import get_noaa_dst
from predstorm_module import logger

# GET INPUT PARAMETERS
from predstorm_l5_input import *

#========================================================================================
#--------------------------------- FUNCTIONS --------------------------------------------
#========================================================================================

def plot_solarwind_and_dst_prediction(DSCOVR_data, STEREOA_data, DST_data, dst_label='Dst Temerin & Li 2002', past_days=3.5, future_days=7., lw=1, fs=11, ms=5, figsize=(14,12), verification_mode=False, timestamp=None):
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
    Dst : list(dst_time, dst, dst_pred)
        Predicted Dst
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

    Returns
    =======
    plt.savefig : .png file
        File saved to XXX
    """

    # Set style:
    sns.set_context("talk")
    sns.set_style("darkgrid")

    # Make figure object:
    fig=plt.figure(1,figsize=figsize)
    axes = []

    # Set data objects:
    stam, sta = STEREOA_data
    dism, dis = DSCOVR_data
    dst_time, dst, com_time, dst_pred = DST_data

    plotstart = dism.time[-1] - past_days
    plotend = dis.time[-1] + future_days

    # For the minute data, check which are the intervals to show for STEREO-A until end of plot
    sta_index_future=np.where(np.logical_and(stam.time > dism.time[-1], \
                              stam.time < dism.time[-1]+plot_future_days))

    if timestamp == None:
        timestamp = datetime.utcnow()
    timeutc = mdates.date2num(timestamp)

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    # Total B-field and Bz (DSCOVR)
    plt.plot_date(dism.time, dism.btot,'-k', label='B total L1', linewidth=lw)
    plt.plot_date(dism.time, dism.bzgsm,'-g', label='Bz GSM L1', linewidth=lw)

    # STEREO-A minute resolution data with timeshift
    plt.plot_date(stam.time[sta_index_future], stam.btot[sta_index_future],
                  '-r', linewidth=lw, label='B STEREO-Ahead')
    plt.plot_date(stam.time[sta_index_future], stam.bn[sta_index_future], markersize=0,
                  linestyle='-', color='darkolivegreen', linewidth=lw, label='Bn RTN STEREO-Ahead')

    # Indicate 0 level for Bz
    plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)
    plt.ylabel('Magnetic field [nT]',  fontsize=fs+2)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    bplotmax=np.nanmax(np.concatenate((dism.btot,stam.btot[sta_index_future])))+5
    bplotmin=np.nanmin(np.concatenate((dism.bzgsm,stam.bn[sta_index_future]))-5)

    plt.ylim(bplotmin, bplotmax)

    plt.title('L1 DSCOVR real time solar wind from NOAA SWPC for '+ datetime.strftime(timestamp, "%Y-%m-%d %H:%M")+ ' UT   STEREO-A beacon', fontsize=16)

    # SUBPLOT 2: Solar wind speed
    # ---------------------------
    ax2 = fig.add_subplot(412)
    axes.append(ax2)

    # Plot solar wind speed (DSCOVR):
    plt.plot_date(dism.time, dism.speed,'-k', label='speed L1',linewidth=lw)
    plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fs+2)


    # Plot STEREO-A data with timeshift and savgol filter
    try:
        plt.plot_date(stam.time[sta_index_future],savgol_filter(stam.speedr[sta_index_future],11,1),'-r', linewidth=lw, label='speed STEREO-Ahead')
    except:
        print("Savgol filter failed for STEREO-A data. Continuing without...")  # TODO Why does this happen? chris: not exactly sure! maybe NaNs?
        plt.plot_date(stam.time[sta_index_future], stam.speedr[sta_index_future],
                      '-r', linewidth=lw, label='speed STEREO-Ahead')

    # Add speed levels:
    for hline, linetext in zip([400, 800], ['slow', 'fast']):
        plt.plot_date([dis.time[0], dis.time[-1]+future_days],
                    [hline, hline],'--k', alpha=0.3, linewidth=1)
        plt.annotate(linetext,xy=(dis.time[0]+past_days,hline),
                    xytext=(dis.time[0]+past_days,hline),
                    color='k', fontsize=10)

    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    try:
        vplotmax=np.nanmax(np.concatenate((dism.speed,savgol_filter(stam.speedr[sta_index_future],11,1))))+100
        vplotmin=np.nanmin(np.concatenate((dism.speed,savgol_filter(stam.speedr[sta_index_future],11,1)))-50)
    except:
        vplotmax=np.nanmax(np.concatenate((dism.speed,stam.speedr[sta_index_future])))+100
        vplotmin=np.nanmin(np.concatenate((dism.speed,stam.speedr[sta_index_future]))-50)
    plt.ylim(vplotmin, vplotmax)


    plt.annotate('now',xy=(timeutc,vplotmax-100),xytext=(timeutc+0.05,vplotmax-100),color='k', fontsize=14)

    # SUBPLOT 3: Solar wind density
    # -----------------------------
    ax3 = fig.add_subplot(413)
    axes.append(ax3)

    # Plot solar wind density:
    plt.plot_date(dism.time, dism.den,'-k', label='density L1',linewidth=lw)
    plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fs+2)
    # For y limits check where the maximum and minimum are for DSCOVR and STEREO taken together:
    plt.ylim([0,np.nanmax(np.nanmax(np.concatenate((dism.den,stam.den[sta_index_future])))+10)])

    #plot STEREO-A data with timeshift and savgol filter
    try:
        plt.plot_date(stam.time[sta_index_future], savgol_filter(stam.den[sta_index_future],5,1),
                      '-r', linewidth=lw, label='density STEREO-Ahead')
    except:
        plt.plot_date(stam.time[sta_index_future], stam.den[sta_index_future],
                      '-r', linewidth=lw, label='density STEREO-Ahead')

    # SUBPLOT 4: Actual and predicted Dst
    # -----------------------------------
    ax4 = fig.add_subplot(414)
    axes.append(ax4)

    # Observed Dst Kyoto (past):
    plt.plot_date(dst_time, dst,'ko', label='Dst observed',markersize=4)
    plt.ylabel('Dst [nT]', fontsize=fs+2)
    plt.ylim([np.nanmin(dst)-50,np.nanmax(dst)+20])

    if not verification_mode:
        plt.plot_date(com_time, dst_pred,'-r', label=dst_label,markersize=3, linewidth=1)
        # Add generic error bars of +/-15 nT:
        error=15
        plt.fill_between(com_time, dst_pred-error, dst_pred+error, alpha=0.2,
                         label='Error for high speed streams')
    else:
        #load saved data l prefix is for loaded - WARNING This will crash if called right now
        [timenowb, sta_ptime, sta_vr, sta_btime, sta_btot, sta_br,sta_bt, sta_bn, rbtime_num, rbtot, rbzgsm, rptime_num, rpv, rpn, lrdst_time, lrdst, lcom_time, ldst_burton, ldst_obrien,ldst_temerin_li]=pickle.load(open(verify_filename,'rb') )
        plt.plot_date(lcom_time, ldst_burton,'-b', label='Forecast Dst Burton et al. 1975',markersize=3, linewidth=1)
        plt.plot_date(lcom_time, ldst_obrien,'-r', label='Forecast Dst OBrien & McPherron 2000',markersize=3, linewidth=1)

    # Label plot with geomagnetic storm levels
    plt.plot_date([dis.time[0], dis.time[-1]+future_days], [0,0],'--k', alpha=0.3, linewidth=1)
    for hline, linetext in zip([-50, -100, -250], ['moderate', 'intense', 'super-storm']):
        plt.plot_date([dis.time[0], dis.time[-1]+future_days],
                      [hline,hline],'--k', alpha=0.3, linewidth=1)
        plt.annotate(linetext,xy=(dis.time[0]+past_days,hline+2),
                     xytext=(dis.time[0]+past_days,hline+2),color='k', fontsize=10)

    # GENERAL FORMATTING
    # ------------------
    for ax in axes:
        ax.set_xlim([plotstart,plotend])
        ax.tick_params(axis="x", labelsize=fs)
        ax.tick_params(axis="y", labelsize=fs)
        ax.legend(loc=2,ncol=4,fontsize=fs-2)

        # Dates on x-axes:
        myformat = mdates.DateFormatter('%b %d %Hh')
        ax.xaxis.set_major_formatter(myformat)

        # Vertical line for NOW:
        ax.plot_date([timeutc,timeutc],[-2000,2000],'-k', linewidth=2)

    # Liability text:
    plt.figtext(0.99,0.05,'C. Moestl, IWF Graz, Austria', fontsize=12, ha='right')
    plt.figtext(0.99,0.025,'https://twitter.com/chrisoutofspace', fontsize=12, ha='right')
    plt.figtext(0.01,0.03,'We take no responsibility or liability for the frequency of provision and accuracy of this forecast.' , fontsize=8, ha='left')
    plt.figtext(0.01,0.01,'We will not be liable for any losses and damages in connection with using the provided information.' , fontsize=8, ha='left')

    #save plot
    if not verification_mode:
        plot_label = 'realtime'
    else:
        plot_label = 'verify'

    filename = os.path.join(outputdirectory,'predstorm_v1_{}_stereo_a_plot_{}.png'.format(
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


def return_stereoa_details(positions, timenow):
    """Returns a string describing STEREO-A's current whereabouts.

    Parameters
    ==========
    positions : ???
        Array containing spacecraft positions at given time.
    DSCOVR_lasttime : float
        Date of last DSCOVR measurements in number form.

    Returns
    =======
    stereostr : str
        Nicely formatted string with info on STEREO-A's location with
        with respect to Earth and L5/L1.
    """

    # Find index of current position:
    pos_time_num = time_to_num_cat(positions.time)
    pos_time_now_ind = np.where(timenow < pos_time_num)[0][0]
    sta_r=positions.sta[0][pos_time_now_ind]

    # Get longitude and latitude
    sta_long_heeq = positions.sta[1][pos_time_now_ind]*180./np.pi
    sta_lat_heeq = positions.sta[2][pos_time_now_ind]*180./np.pi

    # Define time lag from STEREO-A to Earth
    timelag_sta_l1 = abs(sta_long_heeq)/(360./sun_syn)
    arrival_time_l1_sta = timenow + timelag_sta_l1
    arrival_time_l1_sta_str = str(mdates.num2date(arrival_time_l1_sta))

    stereostr = ''
    stereostr += 'STEREO-A HEEQ longitude wrt Earth is {:.1f} degrees.\n'.format(sta_long_heeq)
    stereostr += 'This is {:.2f} times the location of L5.\n'.format(abs(sta_long_heeq)/60.)
    stereostr += 'STEREO-A HEEQ latitude is {:.1f} degrees.\n'.format(sta_lat_heeq)
    stereostr += 'Earth L1 HEEQ latitude is {:.1f} degrees.\n'.format(positions.earth_l1[2][pos_time_now_ind]*180./np.pi,1)
    stereostr += 'Difference HEEQ latitude is {:.1f} degrees.\n'.format(abs(
                sta_lat_heeq-positions.earth_l1[2][pos_time_now_ind]*180./np.pi))
    stereostr += 'STEREO-A heliocentric distance is {:.3f} AU.\n'.format(sta_r)
    stereostr += 'The solar rotation period with respect to Earth is chosen as {:.2f} days.\n'.format(sun_syn)
    stereostr += 'This is a time lag of {:.2f} days.\n'.format(timelag_sta_l1)
    stereostr += 'Arrival time of now STEREO-A wind at L1: {}\n'.format(arrival_time_l1_sta_str[0:16])

    return stereostr


#========================================================================================
#--------------------------------- MAIN PROGRAM -----------------------------------------
#========================================================================================

def main():
    """The main code."""

    print('------------------------------------------------------------------------')
    print('')
    print('PREDSTORM L5 v1 method for geomagnetic storm and aurora forecasting. ')
    print('Christian Moestl, IWF Graz, last update May 2019.')
    print('')
    print('Time shifting magnetic field and plasma data from STEREO-A, ')
    print('or from an L5 mission or interplanetary CubeSats, to predict')
    print('the solar wind at Earth and the Dst index for magnetic storm strength.')
    print('')
    print('')
    print('------------------------------------------------------------------------')
    logger.info("Starting PREDSTORM_L5 script. Running in mode {}".format(run_mode.upper()))

    #================================== (1) GET DATA ========================================


    #------------------------ (1a) real time SDO image --------------------------------------
    #not PFSS
    sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg'
    #PFSS
    #sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193pfss.jpg'
    try: urllib.request.urlretrieve(sdo_latest,'latest_1024_0193.jpg')
    except urllib.error.URLError as e:
        logger.error('Failed downloading ', sdo.latest,' ',e.reason)
    #convert to png
    #check if ffmpeg is available locally in the folder or systemwide
    if os.path.isfile('ffmpeg'):
        os.system('./ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
        ffmpeg_avail=True
        logger.info('downloaded SDO latest_1024_0193.jpg converted to png')
        os.system('rm latest_1024_0193.jpg')
    else:
        os.system('ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
        os.system('rm latest_1024_0193.jpg')

    #TO DO: at some point make own PFSS model with heliopy

    #------------------------ (1b) Get real-time DSCOVR data --------------------------------

    # Get real time DSCOVR data with minute/hourly time resolution as recarray
    if run_mode == 'normal':
        [dism,dis] = get_dscovr_data_real()
        # Get time of the last entry in the DSCOVR data
        timenow = dism.time[-1]
        # Get UTC time now
        timeutc = mdates.date2num(datetime.utcnow())
    elif run_mode == 'historic':
        # TODO: add in function to download DSCOVR data
        [dism,dis] = get_dscovr_data_all(P_filepath="data/dscovrarchive/*",
                                         M_filepath="data/dscovrarchive/*",
                                         starttime=historic_date-timedelta(days=plot_past_days+1),
                                         endtime=historic_date)
        timeutc = mdates.date2num(historic_date)
        timenow = timeutc
    elif run_mode == 'verification':
        print("Verification mode coming soon.")

    tstr_format = "%Y-%m-%d-%H_%M" # "%Y-%m-%d_%H%M" would be better
    timenowstr = datetime.strftime(mdates.num2date(timenow), tstr_format)
    timeutcstr = datetime.strftime(datetime.utcnow(), tstr_format)

    # Open file for logging results:        # TODO use logging module
    logfile=outputdirectory+'/predstorm_v1_realtime_stereo_a_results_'+timeutcstr[0:10]+'-' \
             +timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
    logger.info('Logfile for results is: '+logfile)

    logger.info('Current time UTC')
    logger.info('\t'+timeutcstr)
    logger.info('UTC Time of last datapoint in real time DSCOVR data')
    logger.info('\t'+timenowstr)
    logger.info('Time lag in minutes: {}'.format(int(round((timeutc-timenow)*24*60))))

    resultslog = open(logfile,'wt')
    resultslog.write('')
    resultslog.write('PREDSTORM L5 v1 results \n')
    resultslog.write('For UT time: \n')
    resultslog.write(timenowstr)
    resultslog.write('\n')

    #------------------------ (1c) Get real-time STEREO-A beacon data -----------------------

    #get real time STEREO-A data with minute/hourly time resolution as recarray
    if run_mode == 'normal':
        download_stereoa_data_beacon()
        [stam,sta] = read_stereoa_data_beacon()
    elif run_mode == 'historic':
        download_stereoa_data_beacon(starttime=mdates.num2date(timenow), 
                                     endtime=mdates.num2date(timenow))
        [stam,sta] = read_stereoa_data_beacon()

    #use hourly interpolated data - the 'sta' recarray for further calculations,
    #'stam' for plotting

    #get spacecraft position
    logger.info('loading spacecraft and planetary positions')
    pos=getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
    pos_time_num=time_to_num_cat(pos.time)
    #take position of STEREO-A for time now from position file
    pos_time_now_ind=np.where(timenow < pos_time_num)[0][0]
    sta_r=pos.sta[0][pos_time_now_ind]

    laststa=stam.time[-1]
    laststa_time_str=str(mdates.num2date(laststa))[0:16]
    logger.info('UTC Time of last datapoint in STEREO-A beacon data')
    logger.info('\t'+laststa_time_str)
    logger.info('Time lag in hours: {}'.format(int(round((timeutc-laststa)*24))))

    #========================== (2) PREDICTION CALCULATIONS ==================================

    #------------------------ (2a)  Time lag for solar rotation ------------------------------

    # Get longitude and latitude
    sta_long_heeq = pos.sta[1][pos_time_now_ind]*180./np.pi
    sta_lat_heeq = pos.sta[2][pos_time_now_ind]*180./np.pi

    # define time lag from STEREO-A to Earth
    timelag_sta_l1=abs(sta_long_heeq)/(360/sun_syn) #days
    arrival_time_l1_sta=dis.time[-1]+timelag_sta_l1
    arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))

    #print a few important numbers for current prediction
    stereostr = return_stereoa_details(pos, dism.time[-1])
    resultslog.write('\n')
    resultslog.write(stereostr)
    resultslog.write('\n')

    #------------------------ (2b) Corrections to time-shifted STEREO-A data ----------------

    # (1) make correction for heliocentric distance of STEREO-A to L1 position
    # take position of Earth and STEREO-A from positions file
    # for B and N, makes a difference of about -5 nT in Dst

    # ********* TO DO CHECK exponents - are others better?

    logger.info("Doing corrections to STEREO-A data...")
    earth_r=pos.earth_l1[0][pos_time_now_ind]
    sta.btot=sta.btot*(earth_r/sta_r)**-2
    sta.br=sta.br*(earth_r/sta_r)**-2
    sta.bt=sta.bt*(earth_r/sta_r)**-2
    sta.bn=sta.bn*(earth_r/sta_r)**-2
    sta.den=sta.den*(earth_r/sta_r)**-2
    resultslog.write('corrections to STEREO-A data:')
    resultslog.write('\t1: decline of B and N by factor {}\n'.format(round(((earth_r/sta_r)**-2),3)))

    # (2) correction for timing for the Parker spiral see
    # Simunac et al. 2009 Ann. Geophys. equation 1, see also Thomas et al. 2018 Space Weather
    # difference in heliocentric distance STEREO-A to Earth,
    # actually different for every point so take average of solar wind speed
    # Omega is 360 deg/sun_syn in days, convert to seconds; sta_r in AU to m to km;
    # convert to degrees
    # minus sign: from STEREO-A to Earth the diff_r_deg needs to be positive
    # because the spiral leads to a later arrival of the solar wind at larger
    # heliocentric distances (this is reverse for STEREO-B!)
    #************** problem -> MEAN IS NOT FULLY CORRECT
    AU=149597870.700 #AU in km
    diff_r_deg=-(360/(sun_syn*86400))*((sta_r-earth_r)*AU)/np.nanmean(sta.speedr)
    time_lag_diff_r=round(diff_r_deg/(360/sun_syn),2)
    resultslog.write('\t2: time lag due to Parker spiral in hours: {}\n'.format(round(time_lag_diff_r*24,1)))

    ## ADD BOTH time shifts to the sta.time
    #for hourly data
    sta.time=sta.time+timelag_sta_l1+time_lag_diff_r
    #for minute data
    stam.time=stam.time+timelag_sta_l1+time_lag_diff_r

    #(3) conversion from RTN to HEEQ to GSE to GSM - but done as if STA was along the Sun-Earth line
    #convert STEREO-A RTN data to GSE as if STEREO-A was along the Sun-Earth line
    [dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta.br,sta.bt,sta.bn,sta.time, pos.sta, pos_time_num)
    #GSE to GSM
    [sta.br,sta.bt,sta.bn]=convert_GSE_to_GSM(dbr,dbt,dbn,sta.time)

    resultslog.write('\t3: coordinate conversion of magnetic field components RTN > HEEQ > GSE > GSM.\n')

    #interpolate one more time after time shifts, so that the time is in full hours
    #and the STEREO-A data now start with the end of the dscovr data +1 hour

    #deleted: this leads to shifts in < seconds that result in hours and minutes like 19 59 instead of 20 00
    #sta_time=np.arange(dis.time[-1]+1.000/24,sta.time[-1],1.0000/(24))

    #count how many hours until end of sta.time 
    sta_time_array_len=len(np.arange(dis.time[-1]+1.000/24,sta.time[-1],1.0000/(24)))  
    #make time array with exact full hours
    sta_time= mdates.num2date(dis.time[-1])+ timedelta(hours=1) + np.arange(0,sta_time_array_len) * timedelta(hours=1) 
    #convert back to matplotlib time
    sta_time=mdates.date2num(sta_time)

    sta_btot=np.interp(sta_time,sta.time,sta.btot)
    sta_br=np.interp(sta_time,sta.time,sta.br)
    sta_bt=np.interp(sta_time,sta.time,sta.bt)
    sta_bn=np.interp(sta_time,sta.time,sta.bn)
    sta_speedr=np.interp(sta_time,sta.time,sta.speedr)
    sta_den=np.interp(sta_time,sta.time,sta.den)

    #------------------- (2c) COMBINE DSCOVR and time-shifted STEREO-A data -----------------

    # make combined array of DSCOVR and STEREO-A data
    com_time=np.concatenate((dis.time, sta_time))
    com_btot=np.concatenate((dis.btot, sta_btot))
    com_bx=np.concatenate((dis.bxgsm, sta_br))
    com_by=np.concatenate((dis.bygsm, sta_bt))
    com_bz=np.concatenate((dis.bzgsm, sta_bn))
    com_vr=np.concatenate((dis.speed, sta_speedr))
    com_den=np.concatenate((dis.den, sta_den))

    #if there are nans interpolate them (important for Temerin/Li method)
    if sum(np.isnan(com_den)) > 0:
        good = np.where(np.isfinite(com_den))
        com_den=np.interp(com_time,com_time[good],com_den[good])

    if sum(np.isnan(com_vr)) > 0:
        good = np.where(np.isfinite(com_vr))
        com_vr=np.interp(com_time,com_time[good],com_vr[good])

    #---------------------- (2d) calculate Dst for combined data ----------------------------

    logger.info('Making Dst prediction for L1 calculated from time-shifted STEREO-A beacon data.')

    #This function works as result=make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):
    # ******* PROBLEM: USES vr twice (should be V and Vr in Temerin/Li 2002), take V from STEREO-A data too
    [dst_burton, dst_obrien, dst_temerin_li]=make_dst_from_wind(com_btot, com_bx,com_by, \
                                             com_bz, com_vr,com_vr, com_den, com_time)



    #make_kp_from_wind(btot_in,by_in,bz_in,v_in,density_in) and round to 1 decimal
    kp_newell=np.round(make_kp_from_wind(com_btot,com_by,com_bz,com_vr, com_den),1)

    #make_kp_from_wind(btot_in,by_in,bz_in,v_in,density_in) and round to 2 decimals in GW
    aurora_power=np.round(make_aurora_power_from_wind(com_btot,com_by,com_bz,com_vr, com_den),2)
    #make sure that no values are < 0
    aurora_power[np.where(aurora_power < 0)]=0.0

    #get NOAA Dst for comparison
    [dst_time,dst]=get_noaa_dst()
    logger.info('Loaded Kyoto Dst from NOAA for last 7 days.')

    #========================== (3) PLOT RESULTS ============================================

    # Prediction Dst from L1 and STEREO-A:
    if dst_method == 'temerin_li':      # Can compare methods later to see which is most accurate
        dst_pred = dst_temerin_li
        dst_label = 'Dst Temerin & Li 2002'
    elif dst_method == 'obrien':
        dst_pred = dst_obrien
        dst_label = 'Dst OBrien & McPherron 2000'
    elif dst_method == 'burton':
        dst_pred = dst_burton
        dst_label = 'Dst Burton et al. 1975'
    dst_pred = dst_pred + dst_offset

    # ********************************************************************
    logger.info("Creating output plot...")
    plot_solarwind_and_dst_prediction([dism, dis], [stam, sta], 
                                      [dst_time, dst, com_time, dst_pred],
                                      past_days=plot_past_days,
                                      future_days=plot_future_days,
                                      dst_label=dst_label)
    # ********************************************************************


    #========================== (4) WRITE OUT RESULTS AND VARIABLES =========================

    #-------------- (4a) Write prediction variables (plot) to pickle and txt ASCII file -----

    filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
                  timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.p'

    #make recarrays
    combined=np.rec.array([com_time,com_btot,com_bx,com_by,com_bz,com_den,com_vr,dst_temerin_li,kp_newell,aurora_power,], \
    dtype=[('time','f8'),('btot','f8'),('bx','f8'),('by','f8'),('bz','f8'),('den','f8'),\
           ('vr','f8'),('dst_temerin_li','f8'),('kp_newell','f8'),('aurora_power','f8')])

    pickle.dump(combined, open(filename_save, 'wb') )

    logger.info('PICKLE: Variables saved in: \n'+filename_save)

    filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
                  timenowstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'

    ########## ASCII file

    vartxtout=np.zeros([np.size(com_time),16])

    #get date in ascii
    for i in np.arange(np.size(com_time)):
       time_dummy=mdates.num2date(com_time[i])
       vartxtout[i,0]=time_dummy.year
       vartxtout[i,1]=time_dummy.month
       vartxtout[i,2]=time_dummy.day
       vartxtout[i,3]=time_dummy.hour
       vartxtout[i,4]=time_dummy.minute
       vartxtout[i,5]=time_dummy.second

    vartxtout[:,6]=com_time
    vartxtout[:,7]=com_btot
    vartxtout[:,8]=com_bx
    vartxtout[:,9]=com_by
    vartxtout[:,10]=com_bz
    vartxtout[:,11]=com_den
    vartxtout[:,12]=com_vr
    vartxtout[:,13]=dst_temerin_li
    vartxtout[:,14]=kp_newell
    vartxtout[:,15]=aurora_power

    #description
    #np.savetxt(filename_save, ['time     Dst [nT]     Kp     aurora [GW]   B [nT]    Bx [nT]     By [nT]     Bz [nT]    N [ccm-3]   V [km/s]    '])
    np.savetxt(filename_save, vartxtout, delimiter='',fmt='%4i %2i %2i %2i %2i %2i %10.6f %5.1f %5.1f %5.1f %5.1f   %7.0i %7.0i   %5.0f %5.1f %5.1f', \
               header='        time      matplotlib_time B[nT] Bx   By     Bz   N[ccm-3] V[km/s] Dst[nT]   Kp   aurora [GW]')


    #save the file with the same name to be overwritten and in working directory
    np.savetxt('predstorm_real.txt', vartxtout, delimiter='',fmt='%4i %2i %2i %2i %2i %2i %10.6f %5.1f %5.1f %5.1f %5.1f   %7.0i %7.0i   %5.0f %5.1f %5.1f', \
               header='        time      matplotlib_time B[nT] Bx   By     Bz   N[ccm-3] V[km/s] Dst[nT]   Kp   aurora [GW]')

    logger.info('TXT: Variables saved in:\n'+filename_save)

    ################################# VERIFICATION MODE BRANCH ###############################
    #######**********************
    if verification_mode:       # TODO: Use/adjust/remove this
      print('Verification results for interval:')


      #rdst_time rdst includes the observed Dst of the event
      verify_ind_obs=np.where(np.logical_and(rdst_time > verify_int_start,rdst_time < verify_int_end))
      #lcom_time ldst_burton ldst_obrien are the forecasted indices
      verify_ind_for=np.where(np.logical_and(lcom_time > verify_int_start,lcom_time < verify_int_end-1/24))

      print('Scores:')
      print()
      print('How well was the magnitude?')


      #******check this is not totally correct because some Dst > 0 and some <0 - first verification on May 4 is wrong!!
      #print('Mean absolute difference real Dst to Dst forecast Burton:', int(round(np.mean(abs(ldst_burton[verify_ind_for])-abs(rdst[verify_ind_obs])))), ' +/- ', int(round(np.std(abs(ldst_burton[verify_ind_for])-abs(rdst[verify_ind_obs])))), ' nT' )
      #print('Mean absolute difference real Dst to Dst forecast OBrien:', int(round(np.mean(abs(ldst_obrien[verify_ind_for])-abs(rdst[verify_ind_obs])))), ' +/- ', int(round(np.std(abs(ldst_obrien[verify_ind_for])-abs(rdst[verify_ind_obs])))), ' nT' )

      print('Mean absolute difference real Dst to Dst forecast Burton:', int(round(np.mean(abs(ldst_burton[verify_ind_for]-rdst[verify_ind_obs])))), ' +/- ', int(round(np.std(abs(ldst_burton[verify_ind_for]-rdst[verify_ind_obs])))), ' nT' )
      print('Mean absolute difference real Dst to Dst forecast OBrien:', int(round(np.mean(abs(ldst_obrien[verify_ind_for]-rdst[verify_ind_obs])))), ' +/- ', int(round(np.std(abs(ldst_obrien[verify_ind_for]-rdst[verify_ind_obs])))), ' nT' )
      print('Mean absolute difference real Dst to Dst forecast TemerinLi:', int(round(np.mean(abs(ldst_temerin_li[verify_ind_for]-rdst[verify_ind_obs])))), ' +/- ', int(round(np.std(abs(ldst_temerin_li[verify_ind_for]-rdst[verify_ind_obs])))), ' nT' )


      print('minimum in real Dst and Burton / OBrien: ')
      print('real: ', int(round(np.min(rdst[verify_ind_obs]))) ,' forecast: ', int(round(np.min(ldst_burton[verify_ind_for]))), ' ',int(round(np.min(ldst_obrien[verify_ind_for]))) )
      print()
      print('How well was the timing?')

      print('Time of Dst minimum observed:', str(mdates.num2date(rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])]))[0:16] )
      print('Time of Dst minimum Burton:', str(mdates.num2date(lcom_time[verify_ind_for][np.argmin(ldst_burton[verify_ind_for])]+1/3600))[0:16] )
      print('Time of Dst minimum OBrien:', str(mdates.num2date(lcom_time[verify_ind_for][np.argmin(ldst_obrien[verify_ind_for])]+1/3600))[0:16] )

      print('Time difference of Dst minimum Burton:', int( (lcom_time[verify_ind_for][np.argmin(ldst_burton[verify_ind_for])]-rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])])*24), ' hours' )
      print('Time difference of Dst minimum OBrien:', int( (lcom_time[verify_ind_for][np.argmin(ldst_obrien[verify_ind_for])]-rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])])*24), ' hours' )


      print('')
      print('Best correlation time-shift, at max +/- 24 hours are allowed:')

      timecorr=np.zeros(48)
      for k in np.arange(0,48):
        r=rdst[verify_ind_obs]
        p=ldst_burton[verify_ind_for]

        #shift by up to 12 hours and cut ends off
        r=np.roll(r,k-24)[24:-24]
        p=np.roll(p,k-24)[24:-24]
        timecorr[k]=np.corrcoef(r,p)[0,1]
        #print(k,timecorr[k])

      print('correlation of forecast in time:',round(timecorr[24],2))
      print('best correlation of forecast in time:',round(np.max(timecorr),2))
      print('correlation difference:',round(np.max(timecorr)-timecorr[24],2))
      print('best correlation time difference:',np.argmax(timecorr)-24, ' hours')


      print()
      print('----------------------------')


      sys.exit()

    #----------------------------- (4b) CALCULATE FORECAST RESULTS --------------------------

    # WRITE PREDICTION RESULTS TO LOGFILE
    resultslog.write('-------------------------------------------------')
    resultslog.write('\n')

    #check future times in combined Dst with respect to the end of DSCOVR data
    future_times=np.where(com_time > timenow)
    #past times in combined dst
    past_times=np.where(com_time < timenow)

    resultslog.write('PREDSTORM L5 (STEREO-A) prediction results:')
    resultslog.write('\n')
    resultslog.write('Current time: '+ timeutcstr+ ' UT')

    mindst_time=com_time[past_times[0][0]+np.nanargmin(dst_temerin_li[past_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.
    resultslog.write('\n')
    resultslog.write('Minimum of Dst (past times):\n')
    resultslog.write(str(int(round(np.nanmin(dst_temerin_li[past_times])))) + ' nT \n')
    resultslog.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    mindst_time=com_time[future_times[0][0]+np.nanargmin(dst_temerin_li[future_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.

    resultslog.write('\n')
    resultslog.write('Predicted minimum of Dst (future times):\n')
    resultslog.write(str(int(round(np.nanmin(dst_temerin_li[future_times])))) + ' nT \n')
    resultslog.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of moderate storm levels (-50 to -100 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_temerin_li[future_times] < -50, dst_temerin_li[future_times] > -100))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
     for i in np.arange(0,len(storm_times_ind),1):
      resultslog.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of intense storm levels (-100 to -200 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_temerin_li[future_times] < -100, dst_temerin_li[future_times] > -200))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of super storm levels (< -200 nT):\n')
    storm_times_ind=np.where(dst_temerin_li[future_times] < -200)[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')

    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('------ Other parameters')
    resultslog.write('\n \n')


    ### speed
    maxvr_time=com_time[past_times[0][0]+np.nanargmax(com_vr[past_times])]
    resultslog.write('Maximum speed (past times):\n')
    resultslog.write(str(int(round(np.nanmax(com_vr[past_times]))))+ ' km/s at '+ \
          str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxvr_time=com_time[future_times[0][0]+np.nanargmax(com_vr[future_times])]
    resultslog.write('Maximum speed (future times):\n')
    resultslog.write(str(int(round(np.nanmax(com_vr[future_times]))))+ ' km/s at '+ \
          str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### btot
    maxb_time=com_time[past_times[0][0]+np.nanargmax(com_btot[past_times])]
    resultslog.write('Maximum Btot (past times):\n')
    resultslog.write(str(round(np.nanmax(com_btot[past_times]),1))+ ' nT at '+ \
          str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxb_time=com_time[future_times[0][0]+np.nanargmax(com_btot[future_times])]
    resultslog.write('Maximum Btot (future times):\n')
    resultslog.write(str(round(np.nanmax(com_btot[future_times]),1))+ ' nT at '+ \
          str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### bz
    minbz_time=com_time[past_times[0][0]+np.nanargmin(com_bz[past_times])]
    resultslog.write('Minimum Bz (past times):\n')
    resultslog.write(str(round(np.nanmin(com_bz[past_times]),1))+ ' nT at '+ \
          str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    minbz_time=com_time[future_times[0][0]+np.nanargmin(com_bz[future_times])]
    resultslog.write('Minimum Bz (future times):\n')
    resultslog.write(str(round(np.nanmin(com_bz[future_times]),1))+ ' nT at '+ \
          str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    resultslog.close()

    logger.info("PREDSTORM_L5 run complete!")

    # Print results for overview of run:
    if verbose:
        print("")
        print("-----------------------------")
        f = open(logfile,'r')
        print(*f.readlines())

#========================================================================================
#--------------------------------- RUN SCRIPT -------------------------------------------
#========================================================================================

if __name__ == '__main__':

    run_mode = 'normal'
    verbose = False
    for opt, arg in opts:
        if opt == "--server":
            server = True
        if opt == '-v' or opt == "--verbose":
            verbose = True
        elif opt == '--historic':
            run_mode = 'historic'
            historic_date = datetime.strptime(arg, "%Y-%m-%d %H:%M")
            print("Using historic mode for date: {}".format(historic_date))
        elif opt == '-h' or opt == '--help':
            print("")
            print("-----------------------------------------------------------------")
            print("DESCRIPTION:")
            print("This PREDSTORM L5 script uses time-shifted data from a spacecraft")
            print("east of the Sun-Earth line, currently STEREO-A, to provide real-")
            print("time solar wind and magnetic storm forecasting.")
            print("-------------------------------------")
            print("RUN OPTIONS:")
            print("--server      : Run script in server mode.")
            print("                python predstorm_l5.py --server")
            print("--historic    : Run script with a historic data set. Must have")
            print("                archived data available.")
            print("                python predstorm_l5.py --historic='2017-09-07 23:00'")
            print("GENERAL OPTIONS:")
            print("-h/--help     : print this help data")
            print("-v/--verbose  : print logging output to shell for debugging")
            print("-------------------------------------")
            print("EXAMPLE USAGE:")
            print("  Most basic:")
            print("    python predstorm_l5.py")
            print("    --> See results/ folder for output.")
            print("-----------------------------------------------------------------")
            print("")
            sys.exit()

    # DEFINE OUTPUT DIRECTORIES:
    outputdirectory='results'
    # Check if directory for output exists (for plots and txt files)
    if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
    # Make directory for savefiles
    if os.path.isdir(outputdirectory+'/savefiles') == False:
        os.mkdir(outputdirectory+'/savefiles')
    # Check if directory for beacon data exists
    if os.path.isdir('sta_beacon') == False: os.mkdir('sta_beacon')

    # DEFINE LOGGING MODULE:
    logging.config.fileConfig('logging.ini', disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    # Add handler for logging to shell:
    sh = logging.StreamHandler()
    if verbose:
        sh.setLevel(logging.INFO)
    else:
        sh.setLevel(logging.ERROR)
    shformatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    sh.setFormatter(shformatter)
    logger.addHandler(sh)

    # Closes all plots
    plt.close('all')

    main()


