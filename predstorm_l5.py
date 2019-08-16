#!/usr/bin/env python
"""
PREDSTORM is a real time solar wind and magnetic storm forecasting python package
using time-shifted data from a spacecraft east of the Sun-Earth line.
Currently STEREO-A beacon data is used, but it is also suited for using data from a
possible future L5 mission or interplanetary CubeSats.

Author: C. Moestl, R. Bailey, IWF Graz, Austria
twitter @chrisoutofspace, https://github.com/cmoestl
started April 2018, last update June 2019

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
opts, args = getopt.getopt(argv,"hv=",["server", "help", "historic=", "verbose="])

server = False
if "--server" in [o for o, v in opts]:
    server = True
    print("In server mode!")

import matplotlib
if server:
    matplotlib.use('Agg') # important for server version, otherwise error when making figures
else:
    matplotlib.use('Qt5Agg') # figures are shown on mac

from datetime import datetime, timedelta
import json
import logging
import logging.config
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num
import numpy as np
import pdb
import pickle
import scipy.io
import seaborn as sns
import sunpy.time
import urllib

# Local imports
import heliosat
import predstorm as ps
from predstorm.config.constants import AU, dist_to_L1
from predstorm.plot import plot_solarwind_and_dst_prediction
from predstorm.predict import dst_loss_function
# Old imports (remove later):
from predstorm.data import getpositions, time_to_num_cat

# GET INPUT PARAMETERS
from predstorm_l5_input import *

#========================================================================================
#--------------------------------- FUNCTIONS --------------------------------------------
#========================================================================================

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
    arrival_time_l1_sta_str = str(num2date(arrival_time_l1_sta))

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
    print('IWF-Helio Group, Space Research Institute Graz, last update August 2019.')
    print('')
    print('Time shifting magnetic field and plasma data from STEREO-A, ')
    print('or from an L5 mission or interplanetary CubeSats, to predict')
    print('the solar wind at Earth and the Dst index for magnetic storm strength.')
    print('')
    print('')
    print('------------------------------------------------------------------------')
    logger.info("Starting PREDSTORM_L5 script. Running in mode {}".format(run_mode.upper()))

    timestamp = datetime.utcnow()
    startread = timestamp - timedelta(days=plot_future_days+7)

    #================================== (1) GET DATA ========================================

    logger.info("\n-------------------------\nDATA READS\n-------------------------")
    #get_sdo_realtime_image()

    #------------------------ (1a) Get real-time DSCOVR data --------------------------------

    tstr_format = "%Y-%m-%d-%H_%M" # "%Y-%m-%dT%H%M" would be better
    logger.info("(1) Getting DSCOVR data...")
    if run_mode == 'normal':
        dism = ps.get_dscovr_realtime_data()
        # Get time of the last entry in the DSCOVR data
        timenow = dism['time'][-1]
        # Get UTC time now
        timeutc = date2num(timestamp)
    elif run_mode == 'historic':
        timestamp = historic_date
        if (datetime.utcnow() - historic_date).days < (7.-plot_past_days):
            logger.info("Using historic mode with current DSCOVR data and {} timestamp".format(timestamp))
            dism = ps.get_dscovr_data_real()
            dism = dism.cut(endtime=timestamp)
            dis = dism.make_hourly_data()
        else:
            logger.info("Using historic mode with archive DSCOVR data")
            dism = ps.get_dscovr_data_all(P_filepath="data/dscovrarchive/*",
                                       M_filepath="data/dscovrarchive/*",
                                       starttime=timestamp-timedelta(days=plot_past_days+1),
                                       endtime=timestamp)
        timeutc = date2num(timestamp)
        timenow = timeutc
    elif run_mode == 'verification':
        print("Verification mode coming soon.")
    dism.interp_nans()
    dis = dism.make_hourly_data()

    timeutcstr = datetime.strftime(timestamp, tstr_format)
    timenowstr = datetime.strftime(num2date(timenow), tstr_format)

    # Open file for logging results:        # TODO use logging module
    logfile=outputdirectory+'/predstorm_v1_realtime_stereo_a_results_'+timeutcstr[0:10]+'-' \
             +timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
    logger.info('Logfile for results is: '+logfile)

    logger.info('Current time UTC')
    logger.info('\t{}'.format(timestamp))
    logger.info('UTC Time of last datapoint in real time DSCOVR data')
    logger.info('\t{}'.format(num2date(dism['time'][-1])))
    logger.info('Time lag in minutes: {}'.format(int(round((timeutc-dism['time'][-1])*24*60))))

    resultslog = open(logfile,'wt')

    #------------------------ (1b) Get real-time STEREO-A beacon data -----------------------

    logger.info("(2) Getting STEREO-A data...")
    if run_mode == 'normal':
        stam = ps.get_stereo_beacon_data(starttime=startread, endtime=timestamp)
    elif run_mode == 'historic':
        lag_L1, lag_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
        est_timelag = lag_L1 + lag_r
        stam = ps.get_stereo_beacon_data(starttime=num2date(timenow)-timedelta(days=plot_future_days+est_timelag),
                                           endtime=num2date(timenow)+timedelta(days=2))

    logger.info('UTC time of last datapoint in STEREO-A beacon data:')
    logger.info('\t{}'.format(num2date(stam['time'][-1])))
    logger.info('Time lag in minutes: {}'.format(int(round((timeutc-stam['time'][-1])*24*60))))

    # Prepare data:
    stam.interp_nans()
    stam.load_positions()

    #------------------------- (1c) Load NOAA Dst for comparison ----------------------------

    logger.info("(3) Getting Kyoto Dst data...")
    if run_mode == 'normal':
        dst = ps.get_noaa_dst()
    elif run_mode == 'historic':
        if (datetime.utcnow() - historic_date).days < (7.-plot_past_days):
            dst = ps.get_noaa_dst()
        else:
            dst = ps.get_past_dst(filepath="data/dstarchive/WWW_dstae00016185.dat",
                                  starttime=num2date(timenow)-timedelta(days=plot_past_days+1),
                                  endtime=num2date(timenow))
            if len(dst) == 0.:
                raise Exception("Kyoto Dst data for historic mode is missing! Go to http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html")
        dst = dst.cut(endtime=timestamp)

    #========================== (2) PREDICTION CALCULATIONS ==================================

    logger.info("\n-------------------------\nL5-to-L1 CORRECTIONS\n-------------------------")

    #------------------------ (2a) Corrections to time-shifted STEREO-A data ----------------

    logger.info("Doing corrections to STEREO-A data...")

    logger.info("(1) Shift time at STEREO-A according to solar wind rotation")
    stam.shift_time_to_L1()
    sta = stam.make_hourly_data()
    logger.info("STA-to-L1 adjusted time of last datapoint in STEREO-A:")
    logger.info("\t{}".format(num2date(stam['time'][-1])))

    logger.info("(2) Make correction for difference in heliocentric distance")
    sta.shift_wind_to_L1()

    logger.info("(3) Conversion from RTN to GSM as if STEREO was on Sun-Earth line")
    sta['bx'], sta['by'], sta['bz'] = sta['br'], -sta['bt'], sta['bn']
    stam['bx'], stam['by'], stam['bz'] = stam['br'], -stam['bt'], stam['bn']
    # sta.convert_RTN_to_GSE().convert_GSE_to_GSM()
    # stam.convert_RTN_to_GSE().convert_GSE_to_GSM()

    resultslog.write('\t3: coordinate conversion of magnetic field components RTN > HEEQ > GSE > GSM.\n')

    #------------------- (2b) COMBINE DSCOVR and time-shifted STEREO-A data -----------------

    dis_sta = ps.merge_Data(dis, sta)
    dism_stam = ps.merge_Data(dism, stam)

    #---------------------- (2c) calculate Dst for combined data ----------------------------

    logger.info("\n-------------------------\nINDEX PREDICTIONS\n-------------------------")
    logger.info('Making index predictions for L1 calculated from time-shifted STEREO-A beacon data.')

    # Predict Kp
    kp_newell = dis_sta.make_kp_prediction()
    # Predict Auroral Power
    aurora_power = dis_sta.make_aurora_power_prediction()
    # Calculate Newell coupling parameter
    newell_coupling = dis_sta.get_newell_coupling()

    # Predict Dst from L1 and STEREO-A:
    if dst_method == 'temerin_li':
        dst_pred = dis_sta.make_dst_prediction()
        dst_label = 'Dst Temerin & Li 2002'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'temerin_li_2006':
        dst_pred = dis_sta.make_dst_prediction(method='temerin_li_2006')
        dst_label = 'Dst Temerin & Li 2002'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'obrien':
        dst_pred = dis_sta.make_dst_prediction(method='obrien')
        dst_label = 'Dst OBrien & McPherron 2000'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'burton':
        dst_pred = dis_sta.make_dst_prediction(method='burton')
        dst_label = 'Dst Burton et al. 1975'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'ml':
        with open('dst_pred_model_final.pickle', 'rb') as f:
            model = pickle.load(f)
        dst_pred = dis_sta.make_dst_prediction_from_model(model)
        dst_label = 'Dst predicted using ML (GBR)'
        #dst_pred['dst'] = dst_pred['dst'] + 10

    #========================== (3) PLOT RESULTS ============================================

    logger.info("\n-------------------------\nPLOTTING\n-------------------------")
    # ********************************************************************
    logger.info("Creating output plot...")
    plot_solarwind_and_dst_prediction([dism, dis], [stam, sta], 
                                      dst, dst_pred,
                                      past_days=plot_past_days,
                                      future_days=plot_future_days,
                                      dst_label=dst_label,
                                      timestamp=timestamp)
    # ********************************************************************


    #========================== (4) WRITE OUT RESULTS AND VARIABLES =========================

    #-------------- (4a) Write prediction variables (plot) to pickle and txt ASCII file -----

    logger.info("\n-------------------------\nWRITING RESULTS\n-------------------------")

    # Standard data:
    filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
                  timenowstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
    ps.save_to_file(filename_save, wind=dis_sta, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    logger.info('Variables saved in TXT form: '+filename_save)

    # Realtime 1-hour data:
    ps.save_to_file('predstorm_real.txt', wind=dis_sta, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    # Realtime 1-min data:
    ps.save_to_file('predstorm_real_1m.txt', wind=dism_stam, 
                    dst=dst_pred.interp_to_time(dism_stam['time']),
                    kp=kp_newell.interp_to_time(dism_stam['time']), 
                    aurora=aurora_power.interp_to_time(dism_stam['time']), 
                    ec=newell_coupling.interp_to_time(dism_stam['time']))

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

      print('Time of Dst minimum observed:', str(num2date(rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])]))[0:16] )
      print('Time of Dst minimum Burton:', str(num2date(lcom_time[verify_ind_for][np.argmin(ldst_burton[verify_ind_for])]+1/3600))[0:16] )
      print('Time of Dst minimum OBrien:', str(num2date(lcom_time[verify_ind_for][np.argmin(ldst_obrien[verify_ind_for])]+1/3600))[0:16] )

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
    future_times=np.where(dis_sta['time'] > timenow)
    #past times in combined dst
    past_times=np.where(dis_sta['time'] < timenow)

    resultslog.write('PREDSTORM L5 (STEREO-A) prediction results:')
    resultslog.write('\n')
    resultslog.write('Current time: '+ timeutcstr+ ' UT')

    mindst_time=dis_sta['time'][past_times[0][0]+np.nanargmin(dst_pred['dst'][past_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.
    resultslog.write('\n')
    resultslog.write('Minimum of Dst (past times):\n')
    resultslog.write(str(int(round(float(np.nanmin(dst_pred['dst'][past_times]))))) + ' nT \n')
    resultslog.write('at time: '+str(num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    mindst_time=dis_sta['time'][future_times[0][0]+np.nanargmin(dst_pred['dst'][future_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.

    resultslog.write('\n')
    resultslog.write('Predicted minimum of Dst (future times):\n')
    resultslog.write(str(int(round(float(np.nanmin(dst_pred['dst'][future_times]))))) + ' nT \n')
    resultslog.write('at time: '+str(num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of moderate storm levels (-50 to -100 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_pred['dst'][future_times] < -50, dst_pred['dst'][future_times] > -100))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
     for i in np.arange(0,len(storm_times_ind),1):
      resultslog.write(str(num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of intense storm levels (-100 to -200 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_pred['dst'][future_times] < -100, dst_pred['dst'][future_times] > -200))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of super storm levels (< -200 nT):\n')
    storm_times_ind=np.where(dst_pred['dst'][future_times] < -200)[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')

    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('------ Other parameters')
    resultslog.write('\n \n')


    ### speed
    maxvr_time=dis_sta['time'][past_times[0][0]+np.nanargmax(dis_sta['speed'][past_times])]
    resultslog.write('Maximum speed (past times):\n')
    resultslog.write(str(int(round(float(np.nanmax(dis_sta['speed'][past_times])))))+ ' km/s at '+ \
          str(num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxvr_time=dis_sta['time'][future_times[0][0]+np.nanargmax(dis_sta['speed'][future_times])]
    resultslog.write('Maximum speed (future times):\n')
    resultslog.write(str(int(round(float(np.nanmax(dis_sta['speed'][future_times])))))+ ' km/s at '+ \
          str(num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### btot
    maxb_time=dis_sta['time'][past_times[0][0]+np.nanargmax(dis_sta['btot'][past_times])]
    resultslog.write('Maximum Btot (past times):\n')
    resultslog.write(str(round(float(np.nanmax(dis_sta['btot'][past_times])),1))+ ' nT at '+ \
          str(num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxb_time=dis_sta['time'][future_times[0][0]+np.nanargmax(dis_sta['btot'][future_times])]
    resultslog.write('Maximum Btot (future times):\n')
    resultslog.write(str(round(float(np.nanmax(dis_sta['btot'][future_times])),1))+ ' nT at '+ \
          str(num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### bz
    minbz_time=dis_sta['time'][past_times[0][0]+np.nanargmin(dis_sta['bz'][past_times])]
    resultslog.write('Minimum Bz (past times):\n')
    resultslog.write(str(round(float(np.nanmin(dis_sta['bz'][past_times])),1))+ ' nT at '+ \
          str(num2date(minbz_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    minbz_time=dis_sta['time'][future_times[0][0]+np.nanargmin(dis_sta['bz'][future_times])]
    resultslog.write('Minimum Bz (future times):\n')
    resultslog.write(str(round(float(np.nanmin(dis_sta['bz'][future_times])),1))+ ' nT at '+ \
          str(num2date(minbz_time+1/(24*60)))[0:16])
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
    verbose = True
    for opt, arg in opts:
        if opt == "--server":
            server = True
        if opt == '-v' or opt == "--verbose":
            if arg == 'False':
                verbose = False
        elif opt == '--historic':
            run_mode = 'historic'
            historic_date = datetime.strptime(arg, "%Y-%m-%dT%H:%M")
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
            print("                python predstorm_l5.py --historic='2017-09-07T23:00'")
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

    # INITIATE LOGGING:
    logger = ps.init_logging(verbose=verbose)

    # CHECK OUTPUT DIRECTORIES AND REQUIRED FILES:
    outputdirectory='results'
    # Check if directory for output exists (for plots and txt files)
    if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
    # Make directory for savefiles
    if os.path.isdir(outputdirectory+'/savefiles') == False:
        os.mkdir(outputdirectory+'/savefiles')
    # Check if directory for beacon data exists:
    if os.path.isdir('data/sta_beacon') == False: 
        logger.info("Creating folder data/stereoarchive for data downloads...")
        os.mkdir('data/sta_beacon')
    # Files listing satellite positions:
    if not os.path.exists('data/positions/STEREOA-pred_20070101-20250101_HEEQ_6h.p'):
        logger.warning("Need position files to run script! Generating files under data/positions...")
        ps.spice.generate_all_position_files()

    # Closes all plots
    plt.close('all')

    main()


