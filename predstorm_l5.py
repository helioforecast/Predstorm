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
- deal with CMEs at STEREO, because systematically degrades prediction results

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
opts, args = getopt.getopt(argv,"hv=",["server", "help", "historic=", "verbose=", "use3DCORE=", "validation", "force-stereoa"])

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
import h5py
import logging
import logging.config
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num, DateFormatter
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
from predstorm.config import plotting as pltcfg
from predstorm.plot import plot_solarwind_and_dst_prediction, plot_solarwind_science
from predstorm.plot import plot_solarwind_pretty
from predstorm.predict import dst_loss_function

# GET INPUT PARAMETERS
import predstorm_l5_input as psl5

#========================================================================================
#--------------------------------- MAIN SCRIPT ------------------------------------------
#========================================================================================

def main():
    """The main code."""

    # General variables:
    tstr_format = "%Y-%m-%dT%H%M" #formerly "%Y-%m-%d-%H_%M"
    use_recurrence_model = False
    plot_past_days = psl5.plot_past_days

    # Define timing variables:
    timenow = datetime.utcnow()
    if run_mode == 'normal':
        timestamp = timenow
    timestampstr = datetime.strftime(timestamp, tstr_format) # timeutcstr
    timenowstr = datetime.strftime(timenow, tstr_format)
    use_realtime = (timenow - timestamp).days < (7.-plot_past_days)

    predstorm_header = ''
    predstorm_header += '\n------------------------------------------------------------------------\n'
    predstorm_header += '\n'
    predstorm_header += 'PREDSTORM L5 v1 method for geomagnetic storm and aurora forecasting. \n'
    predstorm_header += 'Created by Helio4Cast Group, Graz, last update September 2020.\n'
    predstorm_header += '\n'
    predstorm_header += 'Time shifting magnetic field and plasma data from STEREO-A, \n'
    predstorm_header += 'or from an L5 mission or interplanetary CubeSats, to predict\n'
    predstorm_header += 'the solar wind at Earth and the Dst index for magnetic storm strength.\n'
    predstorm_header += '\n'
    predstorm_header += '\n'
    predstorm_header += '------------------------------------------------------------------------'
    logger.info(predstorm_header)
    logger.info("Starting PREDSTORM_L5 script. Running in mode {} with timestamp {}".format(run_mode.upper(),
                timestampstr))

    #================================== (1) GET DATA ========================================

    logger.info("\n-------------------------\nDATA READS\n-------------------------")
    ps.get_sdo_realtime_image()

    #------------------------ (1a) Get real-time DSCOVR data --------------------------------

    logger.info("(1) Getting L1 data...")
    if use_realtime:
        # If recent, use real-time data:
        dism = ps.get_noaa_realtime_data()
        dism = dism.cut(endtime=timestamp)
    else:
        # If older timestamp, source from online archive:
        logger.info("Using archived DSCOVR data")
        dism = ps.get_dscovr_data(starttime=timestamp-timedelta(days=plot_past_days+1),
                                  endtime=timestamp)
    time_last_rtsw = num2date(dism['time'][-1]).replace(tzinfo=None) # original timenow

    # Linearly interpolate over NaNs and resample to hourly data:
    dism.interp_nans()
    sw_past_min = dism
    sw_past = dism.make_hourly_data()

    logger.info('Current time (UTC):')
    logger.info('\t{}'.format(timenow))
    logger.info('Time of last datapoint in NOAA real-time data (UTC):')
    logger.info('\t{}'.format(time_last_rtsw))
    logger.info('Time lag in minutes: {:.0f}'.format(np.round((timenow-time_last_rtsw).seconds/60., 0)))

    #------------------------ (1b) Get real-time STEREO-A beacon data -----------------------

    logger.info("(2) Getting STEREO-A data...")

    # Estimate lag between STEREO-A measurements and Earth:
    lag_sta_L1, lag_sta_r = ps.get_time_lag_wrt_earth(timestamp=timestamp, satname='STEREO-A')
    est_timelag = lag_sta_L1 + lag_sta_r
    # Find the number of days to plot into future (reduces as ST-A comes closer):
    plot_future_days = min([est_timelag, psl5.plot_future_days])
    stereo_start = timestamp - timedelta(days=plot_future_days+7)
    logger.info("From STEREO-A position, plotting {:.1f} days in the past and {:.1f} days into the future.".format(
        plot_past_days, plot_future_days))

    # Read data:
    try:
        stam = ps.get_stereo_beacon_data(starttime=stereo_start, endtime=timestamp+timedelta(minutes=1))
        if stam.h['PlasmaDataIntegrity'] == 0: # very low quality data
            if force_stereoa:
                logger.warning("STEREO-A data has low data quality but using data anyway.")
            else:
                raise Exception("Very low STEREO-A data quality!")
        if len(np.where(np.isnan(stam['speed']))[0]) > len(stam)/2:
            if force_stereoa:
                logger.warning("STEREO-A data is {:.1f}% nans but using data anyway.".format(len(np.where(np.isnan(stam['speed']))[0])/len(stam)*100.))
            else:
                raise Exception("STEREO-A data is {:.1f}% nans!".format(len(np.where(np.isnan(stam['speed']))[0])/len(stam)*100.))
        nan_periods = stam.find_nan_periods()
        stam = stam.interp_nans()
        stam.load_positions()
        sta_details = stam.return_position_details(timestamp)
    except Exception as e:
        logger.info("STEREO-A read failed for reason: {}".format(e))
        use_recurrence_model = True

    # If reading STEREO-A failed, take day from 27-day recurrence model instead:
    if run_mode == 'normal' and use_recurrence_model:
        logger.info("STEREO-A plasma data is missing/corrupted, using 27-day recurrence model for plasma data instead!")
        rec_start = timestamp - timedelta(days=27)
        rec_end = timestamp - timedelta(days=27-plot_future_days)
        pers27_path = "data/rtsw_min_last100days.h5"
        sw_future_min = ps.get_rtsw_archive_data(pers27_path)
        sw_future_min.cut(starttime=rec_start, endtime=rec_end)
        sw_future_min['time'] += 27. # correct by one Carrington rotation
        sw_future_min.h['DataSource'] += ' t+27days'
        sw_future_min.source += '+27days'
        shifted_nan_periods = sw_future_min.find_nan_periods()

    if not use_recurrence_model:
        time_last_sta = num2date(stam['time'][-1]).replace(tzinfo=None)
        logger.info('Time of last datapoint in STEREO-A data (UTC):')
        logger.info('\t{}'.format(time_last_sta))
        logger.info('Time lag in minutes: {:.0f}'.format(np.round((timenow - time_last_sta).seconds/60., 0)))

    #------------------------- (1c) Load NOAA Dst for comparison ----------------------------

    logger.info("(3) Getting Kyoto Dst data...")
    if use_realtime:
        dst = ps.get_noaa_dst()
    else:
        dst = ps.get_past_dst(filepath="dstarchive/WWW_dstae00010670.dat",
                              starttime=num2date(timestamp)-timedelta(days=plot_past_days+1),
                              endtime=num2date(timestamp))
        if len(dst) == 0.:
            raise Exception("Kyoto Dst data for historic mode is missing! Go to http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html")
    dst = dst.cut(endtime=timestamp)

    #------------------------- (1d) Load 3DCORE output if available -------------------------

    if use3DCORE:
        logger.info("(4) Reading 3DCORE flux rope output...")
        fr_t_m, fr_B_m, fr_t, fr_B = ps.get_3DCORE_output(path_3DCORE)
    else:
        fr_t_m = []
    dst['dst'] = dst['dst'] + psl5.dst_obs_offset

    #========================== (2) PREDICTION CALCULATIONS ==================================

    #------------------------ (2a) Corrections to time-shifted STEREO-A data ----------------

    if not use_recurrence_model:
        logger.info("\n-------------------------\nL5-to-L1 MAPPING\n-------------------------")
        logger.info("Applying corrections to STEREO-A data...")

        logger.info("(1) Shift time at STEREO-A according to solar wind rotation")
        stam.shift_time_to_L1()
        logger.info("STA-to-L1 adjusted time of last datapoint in STEREO-A:")
        logger.info("\t{}".format(num2date(stam['time'][-1])))

        logger.info("(2) Make correction for difference in heliocentric distance")
        stam.shift_wind_to_L1()

        logger.info("(3) Conversion from RTN to GSE and then to GSM as if STEREO was on Sun-Earth line")
        stam['bx'], stam['by'], stam['bz'] = stam['br'], -stam['bt'], stam['bn']    # RTN to quasi-GSE
        stam.convert_GSE_to_GSM()
        sw_future = stam.make_hourly_data()
        sw_future_min = stam

        # Calculate shifts for NaN periods in STA data:
        shifted_nan_periods = {}
        for key in nan_periods:
            shifted_nan_periods[key] = []
            for times in nan_periods[key]:
                starttime, endtime = times
                lag_sta_L1, lag_sta_r = ps.get_time_lag_wrt_earth(timestamp=num2date(starttime), satname='STEREO-A')
                shifted_start = starttime + lag_sta_L1 + lag_sta_r
                lag_sta_L1, lag_sta_r = ps.get_time_lag_wrt_earth(timestamp=num2date(endtime), satname='STEREO-A')
                shifted_end = endtime + lag_sta_L1 + lag_sta_r
                shifted_nan_periods[key].append([shifted_start, shifted_end])

    else:
        # Assign 27-day recurrence L1 data variables to new keys:
        sw_future_min.vars += ['br', 'bt', 'bn']
        sw_future_min['br'], sw_future_min['bt'], sw_future_min['bn'] = sw_future_min['bx'], sw_future_min['by'], sw_future_min['bz']
        sw_future_min = sw_future_min.interp_nans()
        sw_future = sw_future_min.make_hourly_data()

    #-------------------------- (2b) Take flux rope data from 3DCORE ------------------------

    if use3DCORE:
        logger.info("(!) Filling prediction with flux rope values from 3DCORE")
        # Interpolate minute values to match STEREO values:
        sw_f_times = sw_future_min['time'][np.logical_and(sw_future_min['time'] >= fr_t_m[0], sw_future_min['time'] <= fr_t_m[-1])]
        for ib, bn in enumerate(fr_B_m):
            fr_B_m[ib] = np.interp(sw_f_times, fr_t_m, fr_B_m[ib])
        # Minute:
        sw_future_min['bx'][np.logical_and(sw_future_min['time'] >= fr_t_m[0], sw_future_min['time'] <= fr_t_m[-1])] = fr_B_m[0]
        sw_future_min['by'][np.logical_and(sw_future_min['time'] >= fr_t_m[0], sw_future_min['time'] <= fr_t_m[-1])] = fr_B_m[1]
        sw_future_min['bz'][np.logical_and(sw_future_min['time'] >= fr_t_m[0], sw_future_min['time'] <= fr_t_m[-1])] = fr_B_m[2]
        sw_future_min['bn'][np.logical_and(sw_future_min['time'] >= fr_t_m[0], sw_future_min['time'] <= fr_t_m[-1])] = fr_B_m[2]
        # Hourly:
        sw_future['bx'][np.logical_and(sw_future['time'] >= fr_t[0], sw_future['time'] <= fr_t[-1])] = fr_B[0]
        sw_future['by'][np.logical_and(sw_future['time'] >= fr_t[0], sw_future['time'] <= fr_t[-1])] = fr_B[1]
        sw_future['bz'][np.logical_and(sw_future['time'] >= fr_t[0], sw_future['time'] <= fr_t[-1])] = fr_B[2]
        # Recalculate total B-field:
        sw_future_min['btot'] = np.sqrt(sw_future_min['bx']**2. + sw_future_min['by']**2. + sw_future_min['bz']**2.)
        sw_future['btot'] = np.sqrt(sw_future['bx']**2. + sw_future['by']**2. + sw_future['bz']**2.)

    #------------------- (2c) COMBINE DSCOVR and time-shifted L5/PERS data ------------------

    sw_merged = ps.merge_Data(sw_past, sw_future)
    try:
        sw_merged_min = ps.merge_Data(sw_past_min, sw_future_min)
        savemindata = True
    except:
        logger.warning("No minute data available.")
        savemindata = False

    #---------------------- (2d) calculate Dst for combined data ----------------------------

    logger.info("\n-------------------------\nINDEX PREDICTIONS\n-------------------------")
    logger.info('Making index predictions for L1')

    # Predict Kp
    kp_newell = sw_merged.make_kp_prediction()
    # Predict Auroral Power
    aurora_power = sw_merged.make_aurora_power_prediction()
    # Calculate Newell coupling parameter
    newell_coupling = sw_merged.get_newell_coupling()

    # Predict Dst from L1 and STEREO-A:
    if psl5.dst_method == 'temerin_li':
        dst_pred = sw_merged.make_dst_prediction()
        dst_label = 'Dst Temerin & Li 2002'
        dst_pred['dst'] = dst_pred['dst'] + psl5.dst_offset
    elif psl5.dst_method == 'temerin_li_2006':
        dst_pred = sw_merged.make_dst_prediction(method='temerin_li_2006', t_correction=True)
        dst_label = 'Dst Temerin & Li 2006'
        dst_pred['dst'] = dst_pred['dst'] + psl5.dst_offset
    elif psl5.dst_method == 'obrien':
        dst_pred = sw_merged.make_dst_prediction(method='obrien')
        dst_label = 'Dst OBrien & McPherron 2000'
        dst_pred['dst'] = dst_pred['dst'] + psl5.dst_offset
    elif psl5.dst_method == 'burton':
        dst_pred = sw_merged.make_dst_prediction(method='burton')
        dst_label = 'Dst Burton et al. 1975'
        dst_pred['dst'] = dst_pred['dst'] + psl5.dst_offset
    elif psl5.dst_method.startswith('ml'):
        with open('dst_pred_model_final.pickle', 'rb') as f:
            model = pickle.load(f)
        dst_pred = sw_merged.make_dst_prediction_from_model(model, old_method=True)
        if psl5.dst_method == 'ml_dstdiff':
            dst_tl = sw_merged.make_dst_prediction(method='temerin_li_2006', t_correction=True)
            dst_pred['dst'] = dst_tl['dst'] + dst_pred['dst'] + psl5.dst_offset
        dst_label = 'Dst predicted using ML (GBR)'

    # Combine in data object:
    sw_merged['dst'] = dst_pred['dst']
    sw_merged['kp'] = kp_newell['kp']
    sw_merged['aurora'] = aurora_power['aurora']
    sw_merged['ec'] = newell_coupling['ec']

    #========================== (3) PLOT RESULTS ============================================

    logger.info("\n-------------------------\nPLOTTING\n-------------------------")
    # ********************************************************************
    logger.info("Creating output plots...")
    plot_solarwind_and_dst_prediction([sw_past_min, sw_past], [sw_future_min, sw_future], 
                                      dst, dst_pred,
                                      newell_coupling=newell_coupling,
                                      past_days=plot_past_days,
                                      future_days=plot_future_days,
                                      dst_label=dst_label,
                                      timestamp=timestamp,
                                      times_3DCORE=fr_t_m,
                                      times_nans=shifted_nan_periods)
    plt.close()
    plot_solarwind_science([sw_past_min, sw_past], [sw_future_min, sw_future], 
                                      timestamp=timestamp,
                                      past_days=plot_past_days,
                                      future_days=plot_future_days)

    try:
        plot_solarwind_pretty(sw_past, sw_future, dst_pred, newell_coupling, timestamp)
    except Exception as e:
        logger.warning("Could not run plot_solarwind_pretty() due to error: {}".format(e))
    # ********************************************************************


    #========================== (4) WRITE OUT RESULTS AND VARIABLES =========================

    #-------------- (4a) Write prediction variables (plot) to pickle and txt ASCII file -----

    logger.info("\n-------------------------\nWRITING RESULTS\n-------------------------")

    # Realtime 1-hour data:
    ps.save_to_file('predstorm_real.txt', wind=sw_merged, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    # Realtime 1-min data:
    if savemindata:
        ps.save_to_file('predstorm_real_1m.txt', wind=sw_merged_min, 
                        dst=dst_pred.interp_to_time(sw_merged_min['time']),
                        kp=kp_newell.interp_to_time(sw_merged_min['time']), 
                        aurora=aurora_power.interp_to_time(sw_merged_min['time']), 
                        ec=newell_coupling.interp_to_time(sw_merged_min['time']))

    past_100days = timenow - timedelta(days=100)

    # Data for archiving (past 100 days):
    if run_mode == 'normal':
        pickled_forecasts = "data/past_100days_running_forecasts.p"
        if not os.path.exists(pickled_forecasts):
            forecasts = sw_merged.data
            forecasts = forecasts[:, forecasts[0] > date2num(timestamp)]
            logger.info("Creating new 100-day file for saving forecasts under {}".format(pickled_forecasts))
            recurr = np.full((1, forecasts.shape[1]), use_recurrence_model)   # column defining which input was used
            forecasts = np.vstack((forecasts, recurr))
        else:
            with open(pickled_forecasts, 'rb') as f:
                past_forecasts = pickle.load(f)
            new_forecasts = sw_merged.data
            new_forecasts = new_forecasts[:, new_forecasts[0] > past_forecasts[0][-1]]
            recurr = np.full((1, new_forecasts.shape[1]), use_recurrence_model)
            new_forecasts = np.vstack((new_forecasts, recurr))
            forecasts = np.hstack((past_forecasts, new_forecasts))
            logger.info("Last {} value at {}, adding {} new value(s).".format(pickled_forecasts, num2date(past_forecasts[0][-1]), new_forecasts.shape[1]))
        with open(pickled_forecasts, "wb") as f:
            pickle.dump(forecasts, f)

    # Standard data:
    filename_save = outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_{}.txt'.format(timenowstr)
    ps.save_to_file(filename_save, wind=sw_merged, dst=dst_pred, kp=kp_newell, aurora=aurora_power, ec=newell_coupling)
    logger.info('Variables saved in TXT form: '+filename_save)

    logger.info("PREDSTORM_L5 run complete!")


    #========================== (5) CARRY OUT VALIDATION ====================================

    #----------------------------- (4b) CALCULATE FORECAST RESULTS --------------------------

    if not use_recurrence_model:
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


def validation(look_back=40):
    """Carries out validation for past month of data."""

    import seaborn as sns
    sns.set_style('darkgrid')

    now = datetime.utcnow()

    lw = pltcfg.lw

    pickled_forecasts = "data/past_100days_running_forecasts.p"  # correct without leo later
    if not os.path.exists(pickled_forecasts):
        raise Exception("Cannot carry out validation without past saved values! \
                        Make sure the file {} is being written.".format(pickled_forecasts))

    with open(pickled_forecasts, 'rb') as f:
        past_forecasts = pickle.load(f)

    sw_validation = ps.SatData({'time': past_forecasts[0]})
    sw_validation.data = past_forecasts[:-1]
    sw_validation.vars = ['bz', 'btot', 'speed', 'density', 'dst', 'aurora', 'kp', 'ec', 'ae']
    sw_validation['ae'] = past_forecasts[-1]
    sw_validation = sw_validation.make_hourly_data()
    sw_validation = sw_validation.cut(starttime=now-timedelta(days=look_back), endtime=now)

    # Read 27-day recurrence data:
    pers27_path = "data/rtsw_hour_last100days.h5"
    kyoto_dst = ps.get_rtsw_archive_data(pers27_path, add_dst=True)
    kyoto_dst = kyoto_dst.interp_to_time(sw_validation['time'])
    pers27_path_min = "data/rtsw_min_last100days.h5"
    sw_recurrence = ps.get_rtsw_archive_data(pers27_path_min)
    sw_recurrence = sw_recurrence.make_hourly_data()
    sw_recurrence['time'] += 27. # correct by one Carrington rotation
    sw_recurrence.h['DataSource'] += ' t+27days'
    sw_recurrence = sw_recurrence.interp_to_time(sw_validation['time'])

    pltvars = ['bz', 'btot', 'speed', 'density', 'dst']
    ylabels = {'bz': '$B_z$ [nT]', 'btot': '$B_{tot}$ [nT]', 'speed': 'Solar wind speed\n[km/s]',
               'density': 'Density [ccm-3]', 'dst': '$Dst$ [nT]'}
    l5_true = sw_validation['ae'] != 1.
    l5_inds = np.where(l5_true)
    fig, axes = plt.subplots(len(pltvars)+1, sharex=True, figsize=pltcfg.figsize)

    for i_var, pltvar in enumerate(pltvars):
        if pltvar != 'dst':
            axes[i_var].plot_date(sw_recurrence['time'], sw_recurrence[pltvar], 
                                  'k-', lw=lw, alpha=0.2, label="27-day rec (not used)")
        elif pltvar == 'dst':
            axes[i_var].plot_date(kyoto_dst['time'], kyoto_dst['dst'], '--', c='green', lw=lw, label="Kyoto Dst")
        sta_only = np.full((len(sw_validation)), np.nan)
        sta_only[l5_true] = sw_validation[pltvar][l5_true]
        rec_only = np.full((len(sw_validation)), np.nan)
        rec_only[~l5_true] = sw_validation[pltvar][~l5_true]
        axes[i_var].plot_date(sw_validation['time'], rec_only, 'k-', lw=lw, label="27-day rec ({:.0f}%)".format(100.*(len(sw_validation)-len(l5_inds[0]))/len(sw_validation)))
        axes[i_var].plot_date(sw_validation['time'], sta_only, 'r-', lw=lw, label="STEREO-A ({:.0f}%)".format(100.*len(l5_inds[0])/len(sw_validation)))
        axes[i_var].set_ylabel(ylabels[pltvar])
    axes[-1].plot_date(sw_validation['time'], sw_validation['dst']-kyoto_dst['dst'], 'k-', lw=lw)
    axes[-1].set_ylabel("$\Delta Dst$ [nT]")

    # Formatting:
    axes[0].set_title("Validation plot for solar wind forecasting between {} and {}".format((now - timedelta(days=look_back)).strftime('%Y-%m-%d'), now.strftime('%Y-%m-%d')))
    axes[-1].set_xlim([now - timedelta(days=look_back), now])
    axes[0].legend(loc='upper left', ncol=3)
    axes[-2].legend(loc='upper left', ncol=3)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig('predstorm_validation.png')


#========================================================================================
#--------------------------------- RUN SCRIPT -------------------------------------------
#========================================================================================

if __name__ == '__main__':

    run_mode = 'normal'
    verbose, use3DCORE, force_stereoa = True, False, False
    run_validation = False
    for opt, arg in opts:
        if opt == "--server":
            server = True
        if opt == '-v' or opt == "--verbose":
            if arg == 'False':
                verbose = False
        elif opt == '--historic':
            run_mode = 'historic'
            timestamp = datetime.strptime(arg, "%Y-%m-%dT%H:%M")
            timestamp = timestamp.replace(tzinfo=None)
        elif opt == '--use3DCORE':
            use3DCORE = True
            path_3DCORE = arg
        elif opt == '--force-stereoa':
            force_stereoa = True
        elif opt == '--validation':
            run_validation = True
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
            print("--use3DCORE   : Run script with 3DCORE flux rope input.")
            print("                python predstorm_l5.py --historic='2017-09-07T23:00' --use3DCORE='dst.pickle'")
            print("--validation  : Run script in validation mode.")
            print("                python predstorm_l5.py --validation")
            print("--force-stereoa : Force STEREO-A usage (if the data exists), even if the data is bad.")
            print("                python predstorm_l5.py --force-stereoa")
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
    if os.path.isdir(outputdirectory) == False:
        os.mkdir(outputdirectory)
    # Make directory for savefiles
    if os.path.isdir(outputdirectory+'/savefiles') == False:
        os.mkdir(outputdirectory+'/savefiles')
    # Check if directory for beacon data exists:
    if os.path.isdir('data') == False: 
        logger.info("Creating folder data for data downloads...")
        os.mkdir('data')

    # Closes all plots
    plt.close('all')

    # Run validation:
    if run_validation:
        validation()
        sys.exit()

    main()


