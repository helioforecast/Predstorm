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
opts, args = getopt.getopt(argv,"h",["server", "help"])

server = False
if "--server" in [o for o, v in opts]:
    server = True
    print("In server mode!")

import matplotlib
if server:
    matplotlib.use('Agg') # important for server version, otherwise error when making figures
else:
    matplotlib.use('Qt5Agg') # figures are shown on mac
#    try:
#        import IPython
#    except:
#        print("IPython import for testing failed!")

from datetime import datetime, timedelta
import json
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


import predstorm_module
from predstorm_module import get_dscovr_data_real
from predstorm_module import get_stereoa_data_beacon
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

# GET INPUT PARAMETERS
from predstorm_l5_input import *

for opt, arg in opts:
    if opt == "--server":
        server = True
    elif opt == '-h' or opt == '--help':
        print("This is help text.")

# DEFINE OUTPUT DIRECTORIES:
outputdirectory='results'
# Check if directory for output exists (for plots and txt files)
if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
# Make directory for savefiles
if os.path.isdir(outputdirectory+'/savefiles') == False:
    os.mkdir(outputdirectory+'/savefiles')
# Check if directory for beacon data exists
if os.path.isdir('sta_beacon') == False: os.mkdir('sta_beacon')

#========================================================================================
#--------------------------------- FUNCTIONS --------------------------------------------
#========================================================================================

def plot_solarwind_and_dst_prediction(DSCOVR_data, STEREOA_data, Dst, dst_label='Dst Temerin & Li 2002', past_days=3.5, future_days=7., lw=1, fs=11, ms=5, figsize=(14,12), verification_mode=False):
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
    Dst : array
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

    plotstart = dism.time[-1] - past_days
    plotend = dis.time[-1] + future_days

    # SUBPLOT 1: Total B-field and Bz
    # -------------------------------
    ax1 = fig.add_subplot(411)
    axes.append(ax1)

    # Total B-field and Bz (DSCOVR)
    plt.plot_date(dism.time, dism.btot,'-k', label='B total L1', linewidth=lw)
    plt.plot_date(dism.time, dism.bzgsm,'-g', label='Bz GSM L1',linewidth=lw)

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

    plt.title('L1 DSCOVR real time solar wind from NOAA SWPC for '+ str(mdates.num2date(timeutc))[0:16]+ ' UT   STEREO-A beacon', fontsize=16)

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
    plt.ylim([np.nanmin(Dst)-50,np.nanmax(Dst)+20])
        
    if not verification_mode:
        plt.plot_date(com_time, Dst,'-r', label=dst_label,markersize=3, linewidth=1)
        # Add generic error bars of +/-15 nT:
        error=15
        plt.fill_between(com_time, Dst-error, Dst+error, alpha=0.2, 
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
        ax.legend(loc=2,ncol=3,fontsize=fs-2)
        
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
        
    filename = os.path.join(outputdirectory,'predstorm_v1_{}_stereo_a_plot_{}-{}_{}.png'.format(
                            plot_label, timeutcstr[0:10], timeutcstr[11:13], timeutcstr[14:16]))
    filename_eps = filename.replace('png', 'eps')
        
    if not verification_mode:
        plt.savefig('predstorm_real.png')
        print('Real-time plot saved as predstorm_real.png!')
        
    #if not server: # Just plot and exit
    #    plt.show()
    #    sys.exit()
    plt.savefig(filename)
    print('Plot saved as png:\n', filename)
    log.write('\n')
    log.write('Plot saved as png:\n'+ filename)
    

def return_stereoa_details(positions, DSCOVR_lasttime):
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
    timelag_sta_l1=abs(sta_long_heeq)/(360./sun_syn)
    arrival_time_l1_sta=DSCOVR_lasttime + timelag_sta_l1
    arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))
    
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

# Closes all plots
plt.close('all')

print('------------------------------------------------------------------------')
print()
print('PREDSTORM L5 v1 method for geomagnetic storm and aurora forecasting. ')
print('Christian Moestl, IWF Graz, last update November 2018.')
print()
print('Time shifting magnetic field and plasma data from STEREO-A, ')
print('or from an L5 mission or interplanetary CubeSats, to predict')
print('the solar wind at Earth and the Dst index for magnetic storm strength.')
print()
print()
print('------------------------------------------------------------------------')

#================================== (1) GET DATA ========================================


#------------------------ (1a) real time SDO image
#not PFSS
sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg'
#PFSS
#sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193pfss.jpg'
try: urllib.request.urlretrieve(sdo_latest,'latest_1024_0193.jpg')
except urllib.error.URLError as e:
    print('Failed downloading ', sdo.latest,' ',e.reason)
#convert to png    
#check if ffmpeg is available locally in the folder or systemwide  
if os.path.isfile('ffmpeg'): 
    os.system('./ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
    ffmpeg_avail=True
    print('downloaded SDO latest_1024_0193.jpg converted to png')
    os.system('rm latest_1024_0193.jpg')
else:
    os.system('ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
    os.system('rm latest_1024_0193.jpg')

#TO DO: at some point make own PFSS model with heliopy
 
#------------------------ (1b) Get real-time DSCOVR data --------------------------------

# Get real time DSCOVR data with minute/hourly time resolution as recarray
[dism,dis]=get_dscovr_data_real()

# Get time of the last entry in the DSCOVR data
timenow=dism.time[-1]
timenowstr=str(mdates.num2date(timenow))[0:16]

# Get UTC time now
timeutc=mdates.date2num(datetime.utcnow())
timeutcstr=str(datetime.utcnow())[0:16]

# Open file for logging results:        # TODO use logging module
logfile=outputdirectory+'/predstorm_v1_realtime_stereo_a_results_'+timeutcstr[0:10]+'-' \
         +timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
print('Logfile for results is: ',logfile)
print()

print()
print()
print('Current time UTC')
print(timeutcstr)
print('UTC Time of last datapoint in real time DSCOVR data')
print(timenowstr)
print('Time lag in minutes:', int(round((timeutc-timenow)*24*60)))
print()

log=open(logfile,'wt')
log.write('')
log.write('PREDSTORM L5 v1 results \n')
log.write('For UT time: \n')
log.write(timenowstr)
log.write('\n')

#------------------------ (1c) Get real-time STEREO-A beacon data -----------------------

#get real time STEREO-A data with minute/hourly time resolution as recarray
[stam,sta]=get_stereoa_data_beacon()
#use hourly interpolated data - the 'sta' recarray for further calculations, 
#'stam' for plotting

#get spacecraft position
print('load spacecraft and planetary positions')
pos=getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)
#take position of STEREO-A for time now from position file
pos_time_now_ind=np.where(timenow < pos_time_num)[0][0]
sta_r=pos.sta[0][pos_time_now_ind]


print()
laststa=stam.time[-1]
laststa_time_str=str(mdates.num2date(laststa))[0:16]
print('UTC Time of last datapoint in STEREO-A beacon data')
print(laststa_time_str)
print('Time lag in hours:', int(round((timeutc-laststa)*24)))
print()

#========================== (2) PREDICTION CALCULATIONS ==================================

#------------------------ (2a)  Time lag for solar rotation ------------------------------

# Get longitude and latitude
sta_long_heeq = pos.sta[1][pos_time_now_ind]*180./np.pi
sta_lat_heeq = pos.sta[2][pos_time_now_ind]*180./np.pi

# define time lag from STEREO-A to Earth 
timelag_sta_l1=abs(sta_long_heeq)/(360/sun_syn) #days
arrival_time_l1_sta=dis.time[-1]+timelag_sta_l1
arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))

#feature_sta=mdates.date2num(sunpy.time.parse_time('2018-04-27T01:00:00'))
#arrival_feature_sta_str=str(mdates.num2date(feature_sta+timelag_sta_l1))

#print a few important numbers for current prediction

stereostr = return_stereoa_details(pos, dis.time[-1])
print(stereostr)

log.write('\n')
log.write('\n')
log.write(stereostr)
log.write('\n')
log.write('\n')

#------------------------ (2b) Corrections to time-shifted STEREO-A data ----------------

# (1) make correction for heliocentric distance of STEREO-A to L1 position
# take position of Earth and STEREO-A from positions file 
# for B and N, makes a difference of about -5 nT in Dst
earth_r=pos.earth_l1[0][pos_time_now_ind]
sta.btot=sta.btot*(earth_r/sta_r)**-2
sta.br=sta.br*(earth_r/sta_r)**-2
sta.bt=sta.bt*(earth_r/sta_r)**-2
sta.bn=sta.bn*(earth_r/sta_r)**-2
sta.den=sta.den*(earth_r/sta_r)**-2
print()
print('corrections to STEREO-A data:')
print('1: decline of B and N by factor ',round(((earth_r/sta_r)**-2),3))

log.write('corrections to STEREO-A data:')
log.write('\n')
log.write('1: decline of B and N by factor '+str(round(((earth_r/sta_r)**-2),3)))
log.write('\n')

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
print('2: time lag due to Parker spiral in hours: ', round(time_lag_diff_r*24,1))
log.write('2: time lag due to Parker spiral in hours: '+ str(round(time_lag_diff_r*24,1)))
log.write('\n')


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

print('3: coordinate conversion of magnetic field components RTN > HEEQ > GSE > GSM.')
print()
print()
log.write('3: coordinate conversion of magnetic field components RTN > HEEQ > GSE > GSM.')
log.write('\n')
log.write('\n')


#interpolate one more time after time shifts, so that the time is in full hours
#and the STEREO-A data now start with the end of the dscovr data +1 hour
sta_time=np.arange(dis.time[-1]+1.000/24,sta.time[-1],1.0000/(24))
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

print('Make Dst prediction for L1 calculated from time-shifted STEREO-A beacon data.')
log.write('\n')
log.write('Make Dst prediction for L1 calculated from time-shifted STEREO-A beacon data.')
log.write('\n')

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
print('Loaded Kyoto Dst from NOAA for last 7 days.')
log.write('\n')
log.write('Loaded Kyoto Dst from NOAA for last 7 days.')
log.write('\n')


#========================== (3) PLOT RESULTS  ===========================================

#for the minute data, check which are the intervals to show for STEREO-A until end of plot
sta_index_future=np.where(np.logical_and(stam.time > dism.time[-1], \
                          stam.time < dism.time[-1]+plot_future_days))

# Prediction Dst from L1 and STEREO-A:
if dst_method == 'temerin_li':      # Can compare methods later to see which is most accurate
    Dst = dst_temerin_li
    dst_label = 'Dst Temerin & Li 2002'
elif dst_method == 'obrien':
    Dst = dst_obrien
    dst_label = 'Dst OBrien & McPherron 2000'
elif dst_method == 'burton':
    Dst = dst_burton
    dst_label = 'Dst Burton et al. 1975'
Dst = Dst + dst_offset

# **************************************************************
plot_solarwind_and_dst_prediction([dism, dis], [stam, sta], Dst,
                                  past_days=plot_past_days,
                                  future_days=plot_future_days,
                                  dst_label=dst_label)
# **************************************************************


###################################### (4) WRITE OUT RESULTS AND VARIABLES ###############


############# (4a) write prediction variables (plot) to pickle and txt ASCII file

filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
              timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.p'

#make recarrays
combined=np.rec.array([com_time,com_btot,com_bx,com_by,com_bz,com_den,com_vr,dst_temerin_li,kp_newell,aurora_power,], \
dtype=[('time','f8'),('btot','f8'),('bx','f8'),('by','f8'),('bz','f8'),('den','f8'),\
       ('vr','f8'),('dst_temerin_li','f8'),('kp_newell','f8'),('aurora_power','f8')])
        
pickle.dump(combined, open(filename_save, 'wb') )

print('PICKLE: Variables saved in: \n', filename_save, ' \n')
log.write('\n')
log.write('PICKLE: Variables saved in: \n'+ filename_save+ '\n')

filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
              timenowstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'


########## ASCII file

vartxtout=np.zeros([np.size(com_time),16])

#create array of time strings
#com_time_str= [''  for com_time_str in np.arange(np.size(com_time))]

#get date in ascii
for i in np.arange(np.size(com_time)):
   #for format 2019-03-13 23:59
   #com_time_str[i]=str(mdates.num2date(com_time[i]))[0:16]
   #com_time_str[i]=time_dummy.strftime("%Y %m %d %H %M %M")

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




print('TXT: Variables saved in: \n', filename_save, ' \n ')
log.write('TXT: Variables saved in: \n'+ filename_save+ '\n')




################################# VERIFICATION MODE BRANCH ###############################
#######**********************
if verification_mode:
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





############################### (4b) CALCULATE FORECAST RESULTS ########################################
# WRITE PREDICTION RESULTS TO COMMAND LINE AND LOGFILE


print()
print()
print('-------------------------------------------------')
print()


log.write('\n')
log.write('\n')
log.write('-------------------------------------------------')
log.write('\n')


#check future times in combined Dst with respect to the end of DSCOVR data
future_times=np.where(com_time > timenow)
#past times in combined dst
past_times=np.where(com_time < timenow)


print('PREDSTORM L5 (STEREO-A) prediction results: \n')
print('Current time: ', timeutcstr, ' UT')
print()

log.write('PREDSTORM L5 (STEREO-A) prediction results:')
log.write('\n')
log.write('Current time: '+ timeutcstr+ ' UT')


print('Minimum of Dst (past times):')
print(int(round(np.nanmin(dst_temerin_li[past_times]))), 'nT')
mindst_time=com_time[past_times[0][0]+np.nanargmin(dst_temerin_li[past_times])]
print('at time: ', str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
#added 1 minute manually because of rounding errors in time 19:59:9999 etc.

log.write('\n')
log.write('Minimum of Dst (past times):\n')
log.write(str(int(round(np.nanmin(dst_temerin_li[past_times])))) + ' nT \n')
log.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
log.write('\n')


print()
print('Predicted minimum of Dst (future times):')
print(int(round(np.nanmin(dst_temerin_li[future_times]))), 'nT')
mindst_time=com_time[future_times[0][0]+np.nanargmin(dst_temerin_li[future_times])]
print('at time: ', str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
#added 1 minute manually because of rounding errors in time 19:59:9999 etc.

log.write('\n')
log.write('Predicted minimum of Dst (future times):\n')
log.write(str(int(round(np.nanmin(dst_temerin_li[future_times])))) + ' nT \n')
log.write('at time: '+str(mdates.num2date(mindst_time+1/(24*60)))[0:16])
log.write('\n')

print()
print('Predicted times of moderate storm levels (-50 to -100 nT):')
log.write('\n')
log.write('Predicted times of moderate storm levels (-50 to -100 nT):\n')
storm_times_ind=np.where(np.logical_and(dst_temerin_li[future_times] < -50, dst_temerin_li[future_times] > -100))[0]
#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
 for i in np.arange(0,len(storm_times_ind),1):
  print(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
  log.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
else: 
  print('None')
  log.write('None')
print()
log.write('\n')

print()
print('Predicted times of intense storm levels (-100 to -200 nT):')
log.write('\n')
log.write('Predicted times of intense storm levels (-100 to -200 nT):\n')
storm_times_ind=np.where(np.logical_and(dst_temerin_li[future_times] < -100, dst_temerin_li[future_times] > -200))[0]
#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
   log.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
else: 
  print('None')
  log.write('None')
print()
log.write('\n')

print()
print('Predicted times of super storm levels (< -200 nT):')
log.write('\n')
log.write('Predicted times of super storm levels (< -200 nT):\n')
storm_times_ind=np.where(dst_temerin_li[future_times] < -200)[0]
#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
   log.write(str(mdates.num2date(com_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16]+'\n')
else: 
  print('None')
  log.write('None')

print()
log.write('\n')




print()
print('------ Other parameters')
print()

log.write('\n')
log.write('------ Other parameters')
log.write('\n \n')


### speed

print('Maximum speed (past times):')
maxvr_time=com_time[past_times[0][0]+np.nanargmax(com_vr[past_times])]
print(int(round(np.nanmax(com_vr[past_times]))), 'km/s at', \
      str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
print()
log.write('Maximum speed (past times):\n')
log.write(str(int(round(np.nanmax(com_vr[past_times]))))+ ' km/s at '+ \
      str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
log.write('\n \n')


print('Maximum speed (future times):')
maxvr_time=com_time[future_times[0][0]+np.nanargmax(com_vr[future_times])]
print(int(round(np.nanmax(com_vr[future_times]))), 'km/s at', \
      str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
print()
log.write('Maximum speed (future times):\n')
log.write(str(int(round(np.nanmax(com_vr[future_times]))))+ ' km/s at '+ \
      str(mdates.num2date(maxvr_time+1/(24*60)))[0:16])
log.write('\n \n')


### btot

print('Maximum Btot (past times):')
maxb_time=com_time[past_times[0][0]+np.nanargmax(com_btot[past_times])]
print(round(np.nanmax(com_btot[past_times]),1), 'nT at', \
      str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
print()
log.write('Maximum Btot (past times):\n')
log.write(str(round(np.nanmax(com_btot[past_times]),1))+ ' nT at '+ \
      str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
log.write('\n \n')


print('Maximum Btot (future times):')
maxb_time=com_time[future_times[0][0]+np.nanargmax(com_btot[future_times])]
print(round(np.nanmax(com_btot[future_times]),1), 'nT at', \
      str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
print()
log.write('Maximum Btot (future times):\n')
log.write(str(round(np.nanmax(com_btot[future_times]),1))+ ' nT at '+ \
      str(mdates.num2date(maxb_time+1/(24*60)))[0:16])
log.write('\n \n')


### bz

print('Minimum Bz (past times):')
minbz_time=com_time[past_times[0][0]+np.nanargmin(com_bz[past_times])]
print(round(np.nanmin(com_bz[past_times]),1), 'nT at', \
      str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
print()
log.write('Minimum Bz (past times):\n')
log.write(str(round(np.nanmin(com_bz[past_times]),1))+ ' nT at '+ \
      str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
log.write('\n \n')


print('Minimum Bz (future times):')
minbz_time=com_time[future_times[0][0]+np.nanargmin(com_bz[future_times])]
print(round(np.nanmin(com_bz[future_times]),1), 'nT at', \
      str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
print()
log.write('Minimum Bz (future times):\n')
log.write(str(round(np.nanmin(com_bz[future_times]),1))+ ' nT at '+ \
      str(mdates.num2date(minbz_time+1/(24*60)))[0:16])
log.write('\n \n')

log.close() 



##########################################################################################
################################# CODE STOP ##############################################
##########################################################################################


  
