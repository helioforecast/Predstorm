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
- add error bars for the Temerin/Li Dst model with 1 and 2 sigma
- fill data gaps from STEREO-A beacon data with reasonable Bz fluctuations etc.
  based on a assessment of errors with the ... stereob_errors program
- add timeshifts from L1 to Earth
- add approximate levels of Dst for each location to see the aurora (depends on season)
  taken from correlations of ovation prime, SuomiNPP data in NASA worldview and Dst
- check coordinate conversions again, GSE to GSM is ok
- deal with CMEs at STEREO, because systematically degrades prediction results
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
import copy
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
import urllib

# Local imports
import heliosat
import predstorm as ps
from predstorm.config.constants import AU, dist_to_L1
from predstorm.plot import plot_solarwind_and_dst_prediction
from predstorm.predict import dst_loss_function

# GET INPUT PARAMETERS
from predstorm_l5_input import *

#========================================================================================
#--------------------------------- MAIN SCRIPT ------------------------------------------
#========================================================================================

def main():
    """The main code."""

    timestamp = datetime.utcnow()
    lag_sta_L1, lag_sta_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
    est_timelag = lag_sta_L1 + lag_sta_r
    plot_future_days = est_timelag

    startread = timestamp - timedelta(days=plot_future_days+7)

    predstorm_header = ''
    predstorm_header += '\n------------------------------------------------------------------------\n'
    predstorm_header += '\n'
    predstorm_header += 'PREDSTORM L5 v1 method for geomagnetic storm and aurora forecasting. \n'
    predstorm_header += 'IWF-Helio Group, Space Research Institute Graz, last update August 2019.\n'
    predstorm_header += '\n'
    predstorm_header += 'Time shifting magnetic field and plasma data from STEREO-A, \n'
    predstorm_header += 'or from an L5 mission or interplanetary CubeSats, to predict\n'
    predstorm_header += 'the solar wind at Earth and the Dst index for magnetic storm strength.\n'
    predstorm_header += '\n'
    predstorm_header += '\n'
    predstorm_header += '------------------------------------------------------------------------'
    logger.info(predstorm_header)
    logger.info("Starting PREDSTORM_L5 script. Running in mode {} with timestamp".format(run_mode.upper(), 
        timestamp.strftime("%Y-%m-%d %H:%M")))

    #================================== (1) GET DATA ========================================

    logger.info("\n-------------------------\nDATA READS\n-------------------------")
    ps.get_sdo_realtime_image()

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
            dism = ps.get_dscovr_realtime_data()
            dism = dism.cut(endtime=timestamp)
            dis = dism.make_hourly_data()
        else:
            logger.info("Using historic mode with archive DSCOVR data")
            dism = ps.get_dscovr_data(starttime=timestamp-timedelta(days=plot_past_days+1),
                                      endtime=timestamp)
        timeutc = date2num(timestamp)
        timenow = timeutc
    dism.interp_nans()
    dis = dism.make_hourly_data()
    sw_past = dis

    timeutcstr = datetime.strftime(timestamp, tstr_format)
    timenowstr = datetime.strftime(num2date(timenow), tstr_format)

    logger.info('Current time UTC')
    logger.info('\t{}'.format(timestamp))
    logger.info('UTC Time of last datapoint in real time DSCOVR data')
    logger.info('\t{}'.format(num2date(dism['time'][-1])))
    logger.info('Time lag in minutes: {}'.format(int(round((timeutc-dism['time'][-1])*24*60))))

    #------------------------ (1b) Get real-time STEREO-A beacon data -----------------------

    logger.info("(2) Getting STEREO-A data...")
    if run_mode == 'normal':
        stam = ps.get_stereo_beacon_data(starttime=startread, endtime=timestamp-timedelta(minutes=1))
    elif run_mode == 'historic':
        lag_L1, lag_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
        est_timelag = lag_L1 + lag_r
        stam = ps.get_stereo_beacon_data(starttime=timestamp-timedelta(days=plot_future_days+est_timelag),
                                         endtime=timestamp+timedelta(days=2))

    logger.info('UTC time of last datapoint in STEREO-A beacon data:')
    logger.info('\t{}'.format(num2date(stam['time'][-1])))
    logger.info('Time lag in minutes: {}'.format(int(round((timeutc-stam['time'][-1])*24*60))))

    # Prepare data:
    stam.load_positions()
    sta_details = stam.return_position_details(timestamp)

    use_recurrence_model = False
    if stam.h['PlasmaDataIntegrity'] == 0: # very low quality data
        use_recurrence_model = True
        logger.info("STEREO-A plasma data is missing/corrupted, using 27-day recurrence model for plasma data instead.")
        rec_start = timestamp - timedelta(days=27)
        rec_end = timestamp - timedelta(days=27-plot_future_days)
        sw_future = ps.get_omni_data_new(starttime=rec_start, endtime=rec_end)
        sw_future['time'] = sw_future['time'] + 27.
        sw_future.h['DataSource'] += ' t+27days'
        sw_future.source += '+27days'
        # Remove OMNI indices, don't need them:
        sw_future['kp'], sw_future['ae'], sw_future['dst'] = 0., 0., 0.
        sw_future.vars = [v for v in sw_future.vars if v not in ['kp', 'ae', 'dst']]

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

    logger.info("\n-------------------------\nL5-to-L1 MAPPING\n-------------------------")
    #------------------------ (2a) Corrections to time-shifted STEREO-A data ----------------
    logger.info("Doing corrections to STEREO-A data...")

    logger.info("(1) Shift time at STEREO-A according to solar wind rotation")
    stam.shift_time_to_L1()
    logger.info("STA-to-L1 adjusted time of last datapoint in STEREO-A:")
    logger.info("\t{}".format(num2date(stam['time'][-1])))

    logger.info("(2) Make correction for difference in heliocentric distance")
    stam.shift_wind_to_L1()

    logger.info("(3) Conversion from RTN to GSE and then to GSM as if STEREO was on Sun-Earth line")
    stam['bx'], stam['by'], stam['bz'] = stam['br'], -stam['bt'], stam['bn']    # RTN to quasi-GSE
    stam.convert_GSE_to_GSM()

    if not use_recurrence_model:
        sw_future = stam.make_hourly_data()
        sw_future_min = stam
    else:
        sta = stam.interp_to_time(sw_future['time'])
        sw_future.vars += ['br', 'bt', 'bn']
        for bvar in ['bx', 'by', 'bz', 'btot', 'br', 'bt', 'bn']:
            sw_future[bvar] = sta[bvar] 
        sw_future.interp_nans()
        sw_future_min = copy.deepcopy(sw_future)

    #------------------- (2b) COMBINE DSCOVR and time-shifted STEREO-A data -----------------

    sw_merged = ps.merge_Data(sw_past, sw_future)
    try:
        sw_merged_min = ps.merge_Data(dism, stam)
        savemindata = True
    except:
        logger.warning("No minute data available.")
        savemindata = False

    #---------------------- (2c) calculate Dst for combined data ----------------------------

    logger.info("\n-------------------------\nINDEX PREDICTIONS\n-------------------------")
    logger.info('Making index predictions for L1 calculated from time-shifted STEREO-A beacon data.')

    # Predict Kp
    kp_newell = sw_merged.make_kp_prediction()
    # Predict Auroral Power
    aurora_power = sw_merged.make_aurora_power_prediction()
    # Calculate Newell coupling parameter
    newell_coupling = sw_merged.get_newell_coupling()

    # Predict Dst from L1 and STEREO-A:
    if dst_method == 'temerin_li':
        dst_pred = sw_merged.make_dst_prediction()
        dst_label = 'Dst Temerin & Li 2002'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'temerin_li_2006':
        dst_pred = sw_merged.make_dst_prediction(method='temerin_li_2006')
        dst_label = 'Dst Temerin & Li 2006'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'obrien':
        dst_pred = sw_merged.make_dst_prediction(method='obrien')
        dst_label = 'Dst OBrien & McPherron 2000'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method == 'burton':
        dst_pred = sw_merged.make_dst_prediction(method='burton')
        dst_label = 'Dst Burton et al. 1975'
        dst_pred['dst'] = dst_pred['dst'] + dst_offset
    elif dst_method.startswith('ml'):
        with open('dst_pred_model_final.pickle', 'rb') as f:
            model = pickle.load(f)
        dst_pred = sw_merged.make_dst_prediction_from_model(model)
        if dst_method == 'ml_dstdiff':
            dst_tl = sw_merged.make_dst_prediction(method='temerin_li_2006', t_correction=True)
            dst_pred['dst'] = dst_tl['dst'] + dst_pred['dst']
        dst_label = 'Dst predicted using ML (GBR)'
        #dst_pred['dst'] = dst_pred['dst'] + 10

    #========================== (3) PLOT RESULTS ============================================

    logger.info("\n-------------------------\nPLOTTING\n-------------------------")
    # ********************************************************************
    logger.info("Creating output plot...")
    plot_solarwind_and_dst_prediction([dism, sw_past], [sw_future_min, sw_future], 
                                      dst, dst_pred,
                                      newell_coupling=newell_coupling,
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
    ps.save_to_file(filename_save, wind=sw_merged, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    logger.info('Variables saved in TXT form: '+filename_save)

    # Realtime 1-hour data:
    ps.save_to_file('predstorm_real.txt', wind=sw_merged, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    # Realtime 1-min data:
    if savemindata:
        ps.save_to_file('predstorm_real_1m.txt', wind=sw_merged_min, 
                        dst=dst_pred.interp_to_time(sw_merged_min['time']),
                        kp=kp_newell.interp_to_time(sw_merged_min['time']), 
                        aurora=aurora_power.interp_to_time(sw_merged_min['time']), 
                        ec=newell_coupling.interp_to_time(sw_merged_min['time']))

    logger.info("PREDSTORM_L5 run complete!")

    #----------------------------- (4b) CALCULATE FORECAST RESULTS --------------------------

    logger.info("\n\nSATELLITE POSITION DETAILS\n--------------------------\n"+sta_details)

    future_times = np.where(sw_merged['time'] > date2num(timestamp))
    past_times = np.where(sw_merged['time'] < date2num(timestamp))

    min_cut = np.max((dst['time'][0], sw_merged['time'][0]))
    max_cut = np.min((dst['time'][-1], sw_merged['time'][-1]))
    dst_cut = dst['dst'][np.logical_and(dst['time'] > min_cut, dst['time'] < max_cut)]
    dst_pred_cut = dst_pred['dst'][np.logical_and(dst_pred['time'] > min_cut, dst_pred['time'] < max_cut)]
    results_str = ''

    if len(dst_cut) == len(dst_pred_cut):
        #----------------------------- (4c) GOODNESS METRICS --------------------------
        scores = ps.predict.get_scores(dst_cut, dst_pred_cut, dst['time'][np.logical_and(dst['time'] > min_cut, dst['time'] < max_cut)],
                                       printtext=verbose)
        results_str += 'SCORING\n-------\n'
        results_str += 'MAE of real Dst to Dst forecast:\t{:.2f} nT\n'.format(scores['mae'])
        results_str += 'Diff of real Dst to Dst forecast:\t{:.2f} +/- {:.2f} nT\n'.format(scores['diff_mean'], scores['diff_std'])
        results_str += 'Correlation of forecast in time:\t{:.2f}\n'.format(scores['ppmc'])
        results_str += 'Best correlation of forecast in time:\t{}\n'.format(scores['xcorr_max'])
        results_str += 'Best correlation time difference:\t{} hours\n'.format(scores['xcorr_offset'])
        results_str += '\n'
    else:   # TODO: handle mismatching Dst sizes
        logger.warning("Dst (past) sizes are mismatched with {} and {}".format(len(dst_cut), len(dst_pred_cut)))

    #----------------------------- (4d) RESULTS OF DST PREDICTION  --------------------------
    results_str += 'PREDSTORM L5 (STEREO-A) DST PREDICTION RESULTS\n----------------------------------------------\n'

    mindst_time = dst['time'][np.nanargmin(dst['dst'])]
    mindst_predp_time = dst_pred['time'][np.nanargmin(dst_pred['dst'][past_times])]
    results_str += 'Dst minimum:\n'
    results_str += "\tReal:    \t{:.1f} nT\t  at {}\n".format(np.nanmin(dst['dst']), str(num2date(mindst_time))[0:16])
    results_str += "\tForecast:\t{:.1f} nT\t  at {}\n".format(np.nanmin(dst_pred['dst'][past_times]), str(num2date(mindst_predp_time))[0:16])
    results_str += '\n'

    mindst_predf_time = dst_pred['time'][np.nanargmin(dst_pred['dst'][future_times])]
    results_str += 'Predicted Dst minimum:\n'
    results_str += "\tForecast:\t{:.1f} nT\tat {}\n".format(np.nanmin(dst_pred['dst'][future_times]), str(num2date(mindst_predf_time))[0:16])
    results_str += '\n'

    for dstlims in [[-50,-100], [-100,-200], [-200,-2000]]:
        results_str += 'Predicted times of moderate storm levels (-50 to -100 nT):\n'
        storm_times_ind = np.where(np.logical_and(dst_pred['dst'][future_times] < dstlims[0], dst_pred['dst'][future_times] > dstlims[1]))[0]
        if len(storm_times_ind) > 0:
            for i in np.arange(0,len(storm_times_ind),1):
                results_str += '\t{}\n'.format(str(num2date(sw_merged['time'][future_times][storm_times_ind][i]))[0:16])
        else:
            results_str += '\tNone\n'
        results_str += '\n'

    #----------------------------- (4e) DATA ON SOLAR WIND VALUES --------------------------
    results_str += 'SOLAR WIND PARAMETERS\n---------------------\n'

    for var, unit in zip(['speed', 'btot'], ['km/s', 'nT    ']):
        maxvar_time = sw_merged['time'][past_times[0][0]+np.nanargmax(sw_merged[var][past_times])]
        results_str += 'Maximum {}:\n'.format(var)
        results_str += "\tPast:    \t{:.1f} {} \t   at {}\n".format(np.nanmax(sw_merged[var][past_times]), unit, str(num2date(maxvar_time+1/(24*60)))[0:16])
        maxvar_time = sw_merged['time'][future_times[0][0]+np.nanargmax(sw_merged[var][future_times])]
        results_str += "\tForecast:\t{:.1f} {} \t   at {}\n".format(np.nanmax(sw_merged[var][future_times]), unit, str(num2date(maxvar_time+1/(24*60)))[0:16])
        results_str += '\n'

    var = 'bz'
    minvar_time = sw_merged['time'][past_times[0][0]+np.nanargmin(sw_merged[var][past_times])]
    results_str += 'Minimum {}:\n'.format(var)
    results_str += "\tPast:    \t{:.1f} {} \t   at {}\n".format(np.nanmin(sw_merged[var][past_times]), unit, str(num2date(minvar_time+1/(24*60)))[0:16])
    minvar_time = sw_merged['time'][future_times[0][0]+np.nanargmin(sw_merged[var][future_times])]
    results_str += "\tForecast:\t{:.1f} {} \t   at {}\n".format(np.nanmin(sw_merged[var][future_times]), unit, str(num2date(minvar_time+1/(24*60)))[0:16])
    results_str += '\n'

    logger.info("Final results:\n\n"+results_str)

    #-------------------------------------- (4f) TEST PLOT FOR -------------------------------------
    try:
        fig, (axes) = plt.subplots(1, 3, figsize=(15, 5))
        # Plot of correlation between values
        axes[0].plot(dst_pred_cut, dst_cut, 'bx')
        axes[0].set_title("Real vs. forecast Dst")
        axes[0].set_xlabel("Forecast Dst [nT]")
        axes[0].set_ylabel("Kyoto Dst [nT]")

        # Histogram of forecast
        n, bins, patches = axes[1].hist(dst_pred['dst'], 20)
        axes[1].set_title("Histogram of forecast Dst")
        axes[1].set_xlim([30, -200])
        axes[1].set_xlabel("Forecast Dst [nT]")

        plt.savefig('predstorm_stats.png')
    except:
        pass


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
            historic_date = historic_date.replace(tzinfo=None)
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
            print("--historic    : Run script with a historic data set.")
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
    if os.path.isdir('data') == False: 
        logger.info("Creating folder data for data downloads...")
        os.mkdir('data')

    # Closes all plots
    plt.close('all')

    main()


