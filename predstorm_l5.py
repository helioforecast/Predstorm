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
import matplotlib.dates as mdates
import numpy as np
import pdb
import pickle
import scipy.io
import seaborn as sns
import sunpy.time
import urllib

# Local imports
import predstorm as ps
from predstorm.plot import plot_solarwind_and_dst_prediction
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

    timestamp = datetime.utcnow()

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
    tstr_format = "%Y-%m-%d-%H_%M" # "%Y-%m-%d_%H%M" would be better
    logger.info("Getting DSCOVR data...")
    if run_mode == 'normal':
        dism = ps.get_dscovr_data_real()
        # Get time of the last entry in the DSCOVR data
        timenow = dism['time'][-1]
        # Get UTC time now
        timeutc = mdates.date2num(timestamp)
    elif run_mode == 'historic':
        timestamp = historic_date
        if (datetime.utcnow() - historic_date).days < (7.-plot_past_days):
            dism = ps.get_dscovr_data_real()
            dis = dism.make_hourly_data()
        else:
            dism = ps.get_dscovr_data_all(P_filepath="data/dscovrarchive/*",
                                       M_filepath="data/dscovrarchive/*",
                                       starttime=timestamp-timedelta(days=plot_past_days+1),
                                       endtime=timestamp)
        timeutc = mdates.date2num(timestamp)
        timenow = timeutc
    elif run_mode == 'verification':
        print("Verification mode coming soon.")
    dism.interp_nans()
    dis = dism.make_hourly_data()

    timeutcstr = datetime.strftime(timestamp, tstr_format)
    timenowstr = datetime.strftime(mdates.num2date(timenow), tstr_format)

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
    logger.info("Getting STEREO-A data...")
    if run_mode == 'normal':
        ps.download_stereoa_data_beacon()
        stam = ps.read_stereoa_data_beacon()
    elif run_mode == 'historic':
        lag_L1, lag_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
        est_timelag = lag_L1 + lag_r
        ps.download_stereoa_data_beacon(starttime=mdates.num2date(timenow)-timedelta(days=plot_future_days+est_timelag),
                                        endtime=mdates.num2date(timenow)+timedelta(days=2))
        stam = ps.read_stereoa_data_beacon(starttime=mdates.num2date(timenow)-timedelta(days=plot_future_days+est_timelag),
                                           endtime=mdates.num2date(timenow)+timedelta(days=2))
    stam.interp_nans()
    stam.shift_time_to_L1()
    sta = stam.make_hourly_data()

    #use hourly interpolated data - the 'sta' recarray for further calculations,
    #'stam' for plotting

    laststa=stam['time'][-1]
    laststa_time_str=str(mdates.num2date(laststa))[0:16]
    logger.info('UTC Time of last datapoint in STEREO-A beacon data')
    logger.info('\t'+laststa_time_str)
    logger.info('Time lag in hours: {}'.format(int(round((timeutc-laststa)*24))))

    #------------------------- (1d) Load NOAA Dst for comparison ----------------------------

    logger.info("Getting Kyoto Dst data...")
    if run_mode == 'normal':
        dst = ps.get_noaa_dst()
    elif run_mode == 'historic':
        dst = ps.get_past_dst(filepath="data/dstarchive/WWW_dstae00016185.dat",
                              starttime=mdates.num2date(timenow)-timedelta(days=plot_past_days+1),
                              endtime=mdates.num2date(timenow))

    #========================== (2) PREDICTION CALCULATIONS ==================================

    #------------------------ (2a)  Time lag for solar rotation ------------------------------

    # Get spacecraft position
    AU=149597870.700 #in km
    logger.info('loading spacecraft and planetary positions')

    # NEW METHOD:
    try:
        [sta_r, sta_long_heeq, sta_lat_heeq] = sta.get_position(timestamp, refframe='HEEQ')
        [earth_r, elon, elat] = ps.spice.get_satellite_position('EARTH', timestamp, refframe='HEEQ', rlonlat=True)
        sta_r = sta_r/AU
        earth_r = earth_r/AU * 0.99   # estimated correction to L1
        sta_long_heeq, sta_lat_heeq = sta_long_heeq*180./np.pi, sta_lat_heeq*180./np.pi
        old_pos_method = False
    # OLD METHOD:
    except:
        logger.warning("SPICE methods for position determination failed. Using old method.")
        pos = getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
        pos_time_num = time_to_num_cat(pos.time)
        #take position of STEREO-A for time now from position file
        pos_time_now_ind=np.where(timenow < pos_time_num)[0][0]
        sta_r=pos.sta[0][pos_time_now_ind]
        earth_r=pos.earth_l1[0][pos_time_now_ind]

        # Get longitude and latitude
        sta_long_heeq = pos.sta[1][pos_time_now_ind]*180./np.pi
        sta_lat_heeq = pos.sta[2][pos_time_now_ind]*180./np.pi

        #print a few important numbers for current prediction
        stereostr = return_stereoa_details(pos, dism['time'][-1])
        resultslog.write('\n')
        resultslog.write(stereostr)
        resultslog.write('\n')
        old_pos_method = True

    # define time lag from STEREO-A to Earth
    timelag_sta_l1=abs(sta_long_heeq)/(360./sun_syn) #days
    arrival_time_l1_sta=dis['time'][-1]+timelag_sta_l1
    arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))

    #------------------------ (2b) Corrections to time-shifted STEREO-A data ----------------

    # (1) make correction for heliocentric distance of STEREO-A to L1 position
    # take position of Earth and STEREO-A from positions file
    # for B and N, makes a difference of about -5 nT in Dst

    # ********* TO DO CHECK exponents - are others better?

    logger.info("Doing corrections to STEREO-A data...")
    sta['btot']=sta['btot']*(earth_r/sta_r)**-2
    sta['br']=sta['br']*(earth_r/sta_r)**-2
    sta['bt']=sta['bt']*(earth_r/sta_r)**-2
    sta['bn']=sta['bn']*(earth_r/sta_r)**-2
    sta['density']=sta['density']*(earth_r/sta_r)**-2
    resultslog.write('corrections to STEREO-A data:\n')
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
    diff_r_deg=-(360/(sun_syn*86400))*((sta_r-earth_r)*AU)/np.nanmean(sta['speed'])
    time_lag_diff_r=round(diff_r_deg/(360/sun_syn),2)
    resultslog.write('\t2: time lag due to Parker spiral in hours: {}\n'.format(round(time_lag_diff_r*24,1)))

    ## ADD BOTH time shifts to the sta['time']
    #for hourly data
    # sta['time']=sta['time']+timelag_sta_l1+time_lag_diff_r
    # #for minute data
    # stam['time']=stam['time']+timelag_sta_l1+time_lag_diff_r

    #(3) conversion from RTN to HEEQ to GSE to GSM - but done as if STA was along the Sun-Earth line
    #convert STEREO-A RTN data to GSE as if STEREO-A was along the Sun-Earth line
    if old_pos_method == False:
        sta.convert_RTN_to_GSE().convert_GSE_to_GSM()
        stam.convert_RTN_to_GSE().convert_GSE_to_GSM()
    else:
        sta.convert_RTN_to_GSE(pos_obj=pos.sta, pos_tnum=pos_time_num).convert_GSE_to_GSM()
        stam.convert_RTN_to_GSE(pos_obj=pos.sta, pos_tnum=pos_time_num).convert_GSE_to_GSM()

    resultslog.write('\t3: coordinate conversion of magnetic field components RTN > HEEQ > GSE > GSM.\n')

    #------------------- (2c) COMBINE DSCOVR and time-shifted STEREO-A data -----------------

    dis_sta = ps.merge_Data(dis, sta)
    dism_stam = ps.merge_Data(dism, stam)

    #---------------------- (2d) calculate Dst for combined data ----------------------------

    logger.info('Making Dst prediction for L1 calculated from time-shifted STEREO-A beacon data.')
    #This function works as result=make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):
    # ******* PROBLEM: USES vr twice (should be V and Vr in Temerin/Li 2002), take V from STEREO-A data too
    dst_temerin_li = dis_sta.make_dst_prediction()
    dst_obrien = dis_sta.make_dst_prediction(method='obrien')
    dst_burton = dis_sta.make_dst_prediction(method='burton')

    # Predict Kp
    kp_newell = dis_sta.make_kp_prediction()
    # Predict Auroral Power
    aurora_power = dis_sta.make_aurora_power_prediction()
    # Calculate Newell coupling parameter
    newell_coupling = dis_sta.get_newell_coupling()

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
    dst_pred['dst'] = dst_pred['dst'] + dst_offset

    # ********************************************************************
    logger.info("Creating output plot...")
    plot_solarwind_and_dst_prediction([dism, dis], [stam, sta], 
                                      dst, dst_pred,
                                      past_days=plot_past_days,
                                      future_days=plot_future_days,
                                      dst_label=dst_label,
                                      timestamp=mdates.num2date(timeutc))
    # ********************************************************************


    #========================== (4) WRITE OUT RESULTS AND VARIABLES =========================

    #-------------- (4a) Write prediction variables (plot) to pickle and txt ASCII file -----

    filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
                  timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.p'

    #make recarrays
    combined=np.rec.array([dis_sta['time'],dis_sta['btot'],dis_sta['bx'],dis_sta['by'],dis_sta['bz'],dis_sta['density'],dis_sta['speed'],
        dst_temerin_li['dst'], kp_newell['kp'], aurora_power['aurora'], newell_coupling['ec'],], \
    dtype=[('time','f8'),('btot','f8'),('bx','f8'),('by','f8'),('bz','f8'),('den','f8'),\
           ('vr','f8'),('dst_temerin_li','f8'),('kp_newell','f8'),('aurora_power','f8'),('newell_coupling','f8')])

    pickle.dump(combined, open(filename_save, 'wb') )

    logger.info('PICKLE: Variables saved in: \n'+filename_save)

    filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
                  timenowstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'

    ########## ASCII file

    vartxtout=np.zeros([np.size(dis_sta['time']),17])

    #get date in ascii
    for i in np.arange(np.size(dis_sta['time'])):
       time_dummy=mdates.num2date(dis_sta['time'][i])
       vartxtout[i,0]=time_dummy.year
       vartxtout[i,1]=time_dummy.month
       vartxtout[i,2]=time_dummy.day
       vartxtout[i,3]=time_dummy.hour
       vartxtout[i,4]=time_dummy.minute
       vartxtout[i,5]=time_dummy.second

    vartxtout[:,6]=dis_sta['time']
    vartxtout[:,7]=dis_sta['btot']
    vartxtout[:,8]=dis_sta['bx']
    vartxtout[:,9]=dis_sta['by']
    vartxtout[:,10]=dis_sta['bz']
    vartxtout[:,11]=dis_sta['density']
    vartxtout[:,12]=dis_sta['speed']
    vartxtout[:,13]=dst_temerin_li['dst']
    vartxtout[:,14]=kp_newell['kp']
    vartxtout[:,15]=aurora_power['aurora']
    vartxtout[:,16]=newell_coupling['ec']*100. # convert to Wb/s

    #description
    column_vals = '{:>17}{:>16}{:>7}{:>7}{:>7}{:>7}{:>9}{:>9}{:>8}{:>7}{:>8}{:>12}'.format(
        'Y  m  d  H  M  S', 'matplotlib_time', 'B[nT]', 'Bx', 'By', 'Bz', 'N[ccm-3]', 'V[km/s]',
        'Dst[nT]', 'Kp', 'AP[GW]', 'Ec[Wb/s]')
    time_cols_fmt = '%4i %2i %2i %2i %2i %2i %15.6f'
    b_cols_fmt = 4*'%7.2f'
    p_cols_fmt = '%9.0i%9.0i'
    indices_fmt = '%8.0f%7.2f%8.1f%12.1f'
    float_fmt = time_cols_fmt + b_cols_fmt + p_cols_fmt + indices_fmt
    np.savetxt(filename_save, vartxtout, delimiter='',fmt=float_fmt, header=column_vals)

    # Save real-time files with the same name to be overwritten and in working directory
    # 1-hour data
    np.savetxt('predstorm_real.txt', vartxtout, delimiter='',fmt=float_fmt, header=column_vals)

    vartxtout_m=np.zeros([np.size(dism_stam['time']),17])

    #get date in ascii
    for i in np.arange(np.size(dism_stam['time'])):
       time_dummy=mdates.num2date(dism_stam['time'][i])
       vartxtout_m[i,0]=time_dummy.year
       vartxtout_m[i,1]=time_dummy.month
       vartxtout_m[i,2]=time_dummy.day
       vartxtout_m[i,3]=time_dummy.hour
       vartxtout_m[i,4]=time_dummy.minute
       vartxtout_m[i,5]=time_dummy.second

    vartxtout_m[:,6]=dism_stam['time']
    vartxtout_m[:,7]=dism_stam['btot']
    vartxtout_m[:,8]=dism_stam['bx']
    vartxtout_m[:,9]=dism_stam['by']
    vartxtout_m[:,10]=dism_stam['bz']
    vartxtout_m[:,11]=dism_stam['density']
    vartxtout_m[:,12]=dism_stam['speed']
    vartxtout_m[:,13]=dst_temerin_li.interp_to_time(dism_stam['time'])['dst']
    vartxtout_m[:,14]=kp_newell.interp_to_time(dism_stam['time'])['kp']
    vartxtout_m[:,15]=aurora_power.interp_to_time(dism_stam['time'])['aurora']
    vartxtout_m[:,16]=newell_coupling.interp_to_time(dism_stam['time'])['ec']*100.
    # 1-min data
    np.savetxt('predstorm_real_1m.txt', vartxtout_m, delimiter='',fmt=float_fmt, header=column_vals)

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
    future_times=np.where(dis_sta['time'] > timenow)
    #past times in combined dst
    past_times=np.where(dis_sta['time'] < timenow)

    resultslog.write('PREDSTORM L5 (STEREO-A) prediction results:')
    resultslog.write('\n')
    resultslog.write('Current time: '+ timeutcstr+ ' UT')

    mindst_time=dis_sta['time'][past_times[0][0]+np.nanargmin(dst_temerin_li['dst'][past_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.
    resultslog.write('\n')
    resultslog.write('Minimum of Dst (past times):\n')
    resultslog.write(str(int(round(float(np.nanmin(dst_temerin_li['dst'][past_times]))))) + ' nT \n')
    resultslog.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    mindst_time=dis_sta['time'][future_times[0][0]+np.nanargmin(dst_temerin_li['dst'][future_times])]
    #added 1 minute manually because of rounding errors in time 19:59:9999 etc.

    resultslog.write('\n')
    resultslog.write('Predicted minimum of Dst (future times):\n')
    resultslog.write(str(int(round(float(np.nanmin(dst_temerin_li['dst'][future_times]))))) + ' nT \n')
    resultslog.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of moderate storm levels (-50 to -100 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_temerin_li['dst'][future_times] < -50, dst_temerin_li['dst'][future_times] > -100))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
     for i in np.arange(0,len(storm_times_ind),1):
      resultslog.write(str(mdates.num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of intense storm levels (-100 to -200 nT):\n')
    storm_times_ind=np.where(np.logical_and(dst_temerin_li['dst'][future_times] < -100, dst_temerin_li['dst'][future_times] > -200))[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(mdates.num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
    else:
      resultslog.write('None')
    resultslog.write('\n')

    resultslog.write('\n')
    resultslog.write('Predicted times of super storm levels (< -200 nT):\n')
    storm_times_ind=np.where(dst_temerin_li['dst'][future_times] < -200)[0]
    #when there are storm times above this level, indicate:
    if len(storm_times_ind) >0:
      for i in np.arange(0,len(storm_times_ind),1):
       resultslog.write(str(mdates.num2date(dis_sta['time'][future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
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
          str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxvr_time=dis_sta['time'][future_times[0][0]+np.nanargmax(dis_sta['speed'][future_times])]
    resultslog.write('Maximum speed (future times):\n')
    resultslog.write(str(int(round(float(np.nanmax(dis_sta['speed'][future_times])))))+ ' km/s at '+ \
          str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### btot
    maxb_time=dis_sta['time'][past_times[0][0]+np.nanargmax(dis_sta['btot'][past_times])]
    resultslog.write('Maximum Btot (past times):\n')
    resultslog.write(str(round(float(np.nanmax(dis_sta['btot'][past_times])),1))+ ' nT at '+ \
          str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    maxb_time=dis_sta['time'][future_times[0][0]+np.nanargmax(dis_sta['btot'][future_times])]
    resultslog.write('Maximum Btot (future times):\n')
    resultslog.write(str(round(float(np.nanmax(dis_sta['btot'][future_times])),1))+ ' nT at '+ \
          str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')


    ### bz
    minbz_time=dis_sta['time'][past_times[0][0]+np.nanargmin(dis_sta['bz'][past_times])]
    resultslog.write('Minimum Bz (past times):\n')
    resultslog.write(str(round(float(np.nanmin(dis_sta['bz'][past_times])),1))+ ' nT at '+ \
          str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
    resultslog.write('\n \n')

    minbz_time=dis_sta['time'][future_times[0][0]+np.nanargmin(dis_sta['bz'][future_times])]
    resultslog.write('Minimum Bz (future times):\n')
    resultslog.write(str(round(float(np.nanmin(dis_sta['bz'][future_times])),1))+ ' nT at '+ \
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

    # DEFINE OUTPUT DIRECTORIES:
    outputdirectory='results'
    # Check if directory for output exists (for plots and txt files)
    if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
    # Make directory for savefiles
    if os.path.isdir(outputdirectory+'/savefiles') == False:
        os.mkdir(outputdirectory+'/savefiles')
    # Check if directory for beacon data exists
    if os.path.isdir('sta_beacon') == False: os.mkdir('sta_beacon')

    # INITIATE LOGGING:
    logger = ps.init_logging(verbose=verbose)

    # Closes all plots
    plt.close('all')

    main()


