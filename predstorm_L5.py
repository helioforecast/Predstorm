# PREDSTORM real time solar wind and magnetic storm forecasting with 
# time-shifted data from a spacecraft east of the Sun-Earth line 
# here STEREO-A is used, also suited for data from a 
# possible future L5 mission or interplanetary CubeSats

#Author: C. Moestl, IWF Graz, Austria
#twitter @chrisoutofspace, https://github.com/cmoestl
#started April 2018, last update November 2018

#python 3.5.5 with sunpy and seaborn, ipython 4.2.0

#current status:
# The code works with STEREO-A beacon and DSCOVR data and downloads STEREO-A beacon files 
# into the sta_beacon directory 14 days prior to current time 
# tested for correctly handling missing PLASTIC files

# things to add: 

# - make verification mode new, add manual dst offset
# - add error bars for the Temerin/Li Dst model with 1 and 2 sigma
# - fill data gaps from STEREO-A beacon data with reasonable Bz fluctuations etc.
#   based on a thorough assessment of errors with the ... stereob_errors program
# - add timeshifts from L1 to Earth
# - add approximate levels of Dst for each location to see the aurora (depends on season)
#   taken from correlations of ovation prime, SuomiNPP data in NASA worldview and Dst 
# - check coordinate conversions again, GSE to GSM is ok
# - deal with CMEs at STEREO, because systematically degrades prediction results
# - add metrics ROC for verification etc.
# - DSCOVR data archive:
# https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/1543017600000;1543363199999
# https://www.ngdc.noaa.gov/next-web/docs/guide/catalog.html#dscovrCatalog


# future larger steps:
# (1) add the semi-supervised learning algorithm from the predstorm_L1 program; e.g. with older
# 	STEREO data additionally, so a #combined L1/L5 forecast
# 	most important: implement pattern recognition for STEREO-A streams, 
# 	and link this to the most probably outcome days later at L1
# 	train with STB data around the location where STA is at the moment
# (2) fundamental issue: by what amount is Bz stable for HSS from L5 to L1? are there big changes?
# is Bz higher for specific locations with respect to the HCS and the solar equator? 
# temporal and spatial coherence of Bz
# (3) probabilities for magnetic storm magnitude, probabilities for aurora for many locations


## MIT LICENSE
## Copyright 2018, Christian Moestl 
## Permission is hereby granted, free of charge, to any person obtaining a copy of this 
## software and associated documentation files (the "Software"), to deal in the Software
## without restriction, including without limitation the rights to use, copy, modify, 
## merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
## permit persons to whom the Software is furnished to do so, subject to the following 
## conditions:
## The above copyright notice and this permission notice shall be included in all copies 
## or substantial portions of the Software.
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
## INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
## PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
## HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
## CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
## OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.







##########################################################################################
######################################### CODE START #####################################
##########################################################################################


import scipy.io
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pdb
import urllib
import json
import pickle
import sunpy.time
import seaborn as sns


import predstorm_module
from predstorm_module import get_dscovr_data_real
from predstorm_module import get_stereoa_data_beacon
from predstorm_module import time_to_num_cat
from predstorm_module import converttime
from predstorm_module import make_dst_from_wind
from predstorm_module import make_kp_from_wind
from predstorm_module import getpositions
from predstorm_module import convert_GSE_to_GSM
from predstorm_module import sphere2cart
from predstorm_module import convert_RTN_to_GSE_sta_l1
from predstorm_module import get_noaa_dst

#ignore warnings
#import warnings
#warnings.filterwarnings('ignore')









############################## INPUT PARAMETERS ######################################


inputfilename='predstorm_L5_input.txt'

#reads all lines as strings
lines = open(inputfilename).read().splitlines()

#whether to show interpolated data points on the DSCOVR input plot
showinterpolated=int(lines[3])


#the time interval for both the observed and predicted wind 
#Delta T in hours, start with 24 hours here (covers 1 night of aurora)
deltat=int(lines[9])

#take 4 solar minimum years as training data for 2018
trainstart=lines[12]
trainend=lines[13]

#synodic solar rotation sun_syn=26.24 #days
#carrington rotation 26 deg latitude: 27.28 days
#use other values for equatorial coronal holes?
sun_syn=float(lines[16]) #days

#how far to see in the future with STEREO-A data to the right of the current time
realtime_plot_timeadd=int(lines[19])

#to shift the left beginning of the plot
realtime_plot_leftadd=int(lines[22])


#to get older data for plotting Burton/OBrien Dst for verification
verification_mode=int(lines[25])
#verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-05-04-10_00.p'
#verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-05-29-12_32.p'
#verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-06-16-07_13.p'
verify_filename=lines[27]

#intervals for verification
verify_int_start=lines[30]
verify_int_end=lines[31]

##################*******************
dst_offset=-15


outputdirectory='results'

#check if directory for output exists (for plots and txt files)
#if not make new directory
if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
#also make directory for savefiles
if os.path.isdir(outputdirectory+'/savefiles') == False: \
   os.mkdir(outputdirectory+'/savefiles')

#check if directory for beacon data exists, if not make new directory
if os.path.isdir('sta_beacon') == False: os.mkdir('sta_beacon')









######################################## MAIN PROGRAM ####################################

#get current directory
os.system('pwd')
#closes all plots
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




#SDO image
#download latest 193 PFSS to current directory
#maybe make your own at some point: https://github.com/antyeates1983/pfss
sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193pfss.jpg'
try: urllib.request.urlretrieve(sdo_latest,'latest_1024_0193pfss.jpg')
except urllib.error.URLError as e:
       print(' ', sdo.latest,' ',e.reason)
#convert to png       
os.system('/Users/chris/bin/ffmpeg -i latest_1024_0193pfss.jpg latest_1024_0193pfss.png -loglevel quiet -y')
print('downloaded SDO latest_1024_0193pfss.jpg converted to png')
#delete jpg
os.system('rm latest_1024_0193pfss.jpg')

################################ (1) GET DATA ############################################


######################### (1a) get real time DSCOVR data #################################

#get real time DSCOVR data with minute/hourly time resolution as recarray
[dism,dis]=get_dscovr_data_real()

#get time of the last entry in the DSCOVR data
timenow=dism.time[-1]
timenowstr=str(mdates.num2date(timenow))[0:16]

#get UTC time now
timeutc=mdates.date2num(datetime.datetime.utcnow())
timeutcstr=str(datetime.datetime.utcnow())[0:16]

print()
print()
print('Current time UTC')
print(timeutcstr)
print('UTC Time of last datapoint in real time DSCOVR data')
print(timenowstr)
print('Time lag in minutes:', int(round((timeutc-timenow)*24*60)))
print()



########################## (1b) open file for logging results
logfile=outputdirectory+'/predstorm_v1_realtime_stereo_a_results_'+timeutcstr[0:10]+'-' \
         +timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
print('Logfile for results is: ',logfile)
print()

log=open(logfile,'wt')
log.write('')
log.write('PREDSTORM L5 v1 results \n')
log.write('For UT time: \n')
log.write(timenowstr)
log.write('\n')


########################### (1c) get real time STEREO-A beacon data

#get real time STEREO-A data with minute/hourly time resolution as recarray
[stam,sta]=get_stereoa_data_beacon()
#use hourly interpolated data - the 'sta' recarray for further calculations, 
#'stam' for plotting

#get spacecraft position
print('load spacecraft and planetary positions')
pos=getpositions('cats/positions_2007_2023_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)
#take position of STEREO-A for time now from position file
pos_time_now_ind=np.where(timenow < pos_time_num)[0][0]
sta_r=pos.sta[0][pos_time_now_ind]
#get longitude and latitude
sta_long_heeq=pos.sta[1][pos_time_now_ind]*180/np.pi
sta_lat_heeq=pos.sta[2][pos_time_now_ind]*180/np.pi


print()
laststa=stam.time[-1]
laststa_time_str=str(mdates.num2date(laststa))[0:16]
print('UTC Time of last datapoint in STEREO-A beacon data')
print(laststa_time_str)
print('Time lag in hours:', int(round((timeutc-laststa)*24)))
print()











########################### (2) PREDICTION CALCULATIONS ##################################


########################### (2a)  Time lag for solar rotation

# define time lag from STEREO-A to Earth 
timelag_sta_l1=abs(sta_long_heeq)/(360/sun_syn) #days
arrival_time_l1_sta=dis.time[-1]+timelag_sta_l1
arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))


#feature_sta=mdates.date2num(sunpy.time.parse_time('2018-04-27T01:00:00'))
#arrival_feature_sta_str=str(mdates.num2date(feature_sta+timelag_sta_l1))

#print a few important numbers for current prediction
print('STEREO-A HEEQ longitude to Earth is ', round(sta_long_heeq,1),' degree.') 
print('This is ', round(abs(sta_long_heeq)/60,2),' times the location of L5.') 
print('STEREO-A HEEQ latitude is ', round(sta_lat_heeq,1),' degree.') 
print('Earth L1 HEEQ latitude is ', \
       round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1),' degree')
print('Difference HEEQ latitude is ', \
       abs(round(sta_lat_heeq-pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1)),' degree')
print('STEREO-A heliocentric distance is ', round(sta_r,3),' AU.') 
print('The Sun rotation period with respect to Earth is chosen as ', sun_syn,' days') 
print('This is a time lag of ', round(timelag_sta_l1,2), ' days.') 
print('Arrival time of now STEREO-A wind at L1:',arrival_time_l1_sta_str[0:16])

log.write('\n')
log.write('\n')
log.write('STEREO-A HEEQ longitude to Earth is '+ str(round(sta_long_heeq,1))+' degree.   \
           \nThis is '+ str(round(abs(sta_long_heeq)/60,2))+' times the location of L5.   \
           \nSTEREO-A HEEQ latitude is '+str( round(sta_lat_heeq,1))+' degree.                 \
           \nEarth L1 HEEQ latitude is '+str(round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1))+' degree. \
           \nDifference HEEQ latitude is '+str(abs(round(sta_lat_heeq-pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1)))+' degree. \
           \nSTEREO-A heliocentric distance is '+ str(round(sta_r,3))+' AU. \
           \nThe Sun rotation period with respect to Earth is chosen as '+ str(sun_syn)+' days. \
           \nThis is a time lag of '+str( round(timelag_sta_l1,2))+ ' days. \
           \nArrival time of now STEREO-A wind at L1: '+str(arrival_time_l1_sta_str[0:16]))
log.write('\n')
log.write('\n')



################################# (2b) Further corrections to time-shifted STEREO-A data 

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



#################### (2c) COMBINE DSCOVR and time-shifted STEREO-A data ################## 

# make combined array of DSCOVR and STEREO-A data
com_time=np.concatenate((dis.time, sta_time))
com_btot=np.concatenate((dis.btot, sta_btot))
com_bx=np.concatenate((dis.bxgsm, sta_br))
com_by=np.concatenate((dis.bygsm, sta_bt))
com_bz=np.concatenate((dis.bzgsm, sta_bn))
com_vr=np.concatenate((dis.speed, sta_speedr))
com_den=np.concatenate((dis.den, sta_den))


#if there are nans interpolate them (important for Temerin/Li method)
if sum(np.isnan(com_den)) >0: 
 good = np.where(np.isfinite(com_den)) 
 com_den=np.interp(com_time,com_time[good],com_den[good])

if sum(np.isnan(com_vr)) >0: 
 good = np.where(np.isfinite(com_vr)) 
 com_vr=np.interp(com_time,com_time[good],com_vr[good])


##################### (2d) calculate Dst/Kp for combined data ###############################

print('Make Dst/Kp prediction for L1 calculated from time-shifted STEREO-A beacon data.')
log.write('\n')
log.write('Make Dst/Kp prediction for L1 calculated from time-shifted STEREO-A beacon data.')
log.write('\n')

#This function works as result=make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):
# ******* PROBLEM: USES vr twice (should be V and Vr in Temerin/Li 2002), take V from STEREO-A data too
[dst_burton, dst_obrien, dst_temerin_li]=make_dst_from_wind(com_btot, com_bx,com_by, \
                                         com_bz, com_vr,com_vr, com_den, com_time)

#make_kp_from_wind(btot_in,by_in,bz_in,v_in,density_in) and round to 1 decimal
kp_newell=np.round(make_kp_from_wind(com_btot,com_by,com_bz,com_vr, com_den),1)


#get NOAA Dst for comparison 
[dst_time,dst]=get_noaa_dst()
print('Loaded Kyoto Dst from NOAA for last 7 days.')
log.write('\n')
log.write('Loaded Kyoto Dst from NOAA for last 7 days.')
log.write('\n')













################################### (3) PLOT RESULTS  ####################################

#for the minute data, check which are the intervals to show for STEREO-A until end of plot
sta_index_future=np.where(np.logical_and(stam.time > dism.time[-1], \
                          stam.time < dism.time[-1]+realtime_plot_timeadd))

#initiate figure
sns.set_context("talk")     
sns.set_style("darkgrid")  
fig=plt.figure(1,figsize=(14,12)) #fig=plt.figure(1,figsize=(14,14))
wide=1
fsize=11
msize=5

plotstart=np.floor(dism.time[-1]-realtime_plot_leftadd)
plotend=np.floor(dis.time[-1]+realtime_plot_timeadd)



################################# panel 1
ax1 = fig.add_subplot(411)

#plot DSCOVR with minute resolution
plt.plot_date(dism.time, dism.btot,'-k', label='B total L1', linewidth=wide)
plt.plot_date(dism.time, dism.bzgsm,'-g', label='Bz GSM L1',linewidth=wide)

#plot STEREO-A minute resolution data with timeshift	
plt.plot_date(stam.time[sta_index_future], stam.btot[sta_index_future], \
              '-r', linewidth=wide, label='B STEREO-Ahead')
plt.plot_date(stam.time[sta_index_future], stam.bn[sta_index_future], markersize=0, \
       linestyle='-', color='darkolivegreen', linewidth=wide, label='Bn RTN STEREO-Ahead')

#indicate 0 level for Bz
plt.plot_date([plotstart,plotend], [0,0],'--k', alpha=0.5, linewidth=1)

#vertical line and indicator for prediction and observation
plt.plot_date([timeutc,timeutc],[-100,100],'-k', linewidth=2)


#if showinterpolated > 0: plt.plot_date(rbtimes24, rbtot24,'ro', label='B total interpolated last 24 hours',linewidth=wide,markersize=msize)
#if showinterpolated > 0: plt.plot_date(rbtimes24, rbzgsm24,'go', label='Bz GSM interpolated last 24 hours',linewidth=wide,markersize=msize)
#test hourly interpolation 
#plt.plot_date(rtimes7, rbtot7,'-ko', label='B7',linewidth=wide,markersize=5)
#plt.plot_date(rtimes7, rbzgsm7,'-go', label='Bz7',linewidth=wide,markersize=5)
#plt.plot_date(sta_time7, sta_btot7,'-ko', label='B7',linewidth=wide, markersize=5)
#plt.plot_date(sta_time7, sta_bn7,'-go', label='Bz GSM STEREO-Ahead',linewidth=wide,markersize=5)
#plt.plot_date(sta_time7, dbn,'-bo', label='Bg',linewidth=wide,markersize=5)


plt.ylabel('Magnetic field [nT]',  fontsize=fsize+2)
myformat = mdates.DateFormatter('%b %d %Hh') #myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax1.xaxis.set_major_formatter(myformat)
ax1.legend(loc='upper left', fontsize=fsize-2,ncol=4)

#for y limits check where the maximum and minimum are for DSCOVR and STEREO in the minute data taken together
#negative is surely in bz for this plot, positive in btot
bplotmax=np.nanmax(np.concatenate((dism.btot,stam.btot[sta_index_future])))+5
bplotmin=np.nanmin(np.concatenate((dism.bzgsm,stam.bn[sta_index_future]))-5)

plt.ylim(bplotmin, bplotmax)
plt.xlim([plotstart,plotend])

plt.title('L1 DSCOVR real time solar wind from NOAA SWPC for '+ str(mdates.num2date(timeutc))[0:16]+ ' UT   STEREO-A beacon', fontsize=16)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)



###################################### panel 2
ax2 = fig.add_subplot(412)

#DSCOVR
plt.plot_date(dism.time, dism.speed,'-k', label='speed L1',linewidth=wide)
#plot STEREO-A data with timeshift	and savgol filter
from scipy.signal import savgol_filter
plt.plot_date(stam.time[sta_index_future], savgol_filter(stam.speedr[sta_index_future],5,1),'-r', linewidth=wide, label='speed STEREO-Ahead')

#add speed levels
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [400,400],'--k', alpha=0.3, linewidth=1)
plt.annotate('slow',xy=(dis.time[0]+realtime_plot_leftadd,400),xytext=(dis.time[0]+realtime_plot_leftadd,400),color='k', fontsize=10)
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [800,800],'--k', alpha=0.3, linewidth=1)
plt.annotate('fast',xy=(dis.time[0]+realtime_plot_leftadd,800),xytext=(dis.time[0]+realtime_plot_leftadd,800),color='k', fontsize=10	)


#for y limits check where the maximum and minimum are for DSCOVR and STEREO taken together
vplotmax=np.nanmax(np.concatenate((dism.speed,savgol_filter(stam.speedr[sta_index_future],15,1))))+100
vplotmin=np.nanmin(np.concatenate((dism.speed,savgol_filter(stam.speedr[sta_index_future],15,1)))-50)
plt.ylim(vplotmin, vplotmax)
plt.xlim([plotstart,plotend])

#now vertical line for current time
plt.plot_date([timeutc,timeutc],[0,vplotmax],'-k', linewidth=2)
plt.annotate('now',xy=(timeutc,vplotmax-100),xytext=(timeutc+0.05,vplotmax-100),color='k', fontsize=14)
#plt.annotate('observation',xy=(timenowb,bplotmax-3),xytext=(timenowb-0.55,bplotmax-3),color='k', fontsize=15)
#plt.annotate('prediction',xy=(timenowb,bplotmax-3),xytext=(timenowb+0.45,bplotmax-3),color='b', fontsize=15)

#if showinterpolated > 0: plt.plot_date(rptimes24, rpv24,'ro', label='V interpolated last 24 hours',linewidth=wide,markersize=msize)
#test interpolation
#plt.plot_date(dis.time, rpv7,'-ko', label='v7',linewidth=wide,markersize=5)
#plt.plot_date(sta_time7, sta_vr7,'-go', label='Vr7',linewidth=wide,markersize=5)


#plt.ylim([np.nanmin(rpv)-50,np.nanmax(rpv)+100])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fsize+2)
ax2.xaxis.set_major_formatter(myformat)
ax2.legend(loc=2,fontsize=fsize-2,ncol=2)



########################################### panel 3 density
ax3 = fig.add_subplot(413)
plt.plot_date(dism.time, dism.den,'-k', label='density L1',linewidth=wide)

#plot STEREO-A data with timeshift and savgol filter
plt.plot_date(stam.time[sta_index_future], savgol_filter(stam.den[sta_index_future],5,1),'-r', linewidth=wide, label='density STEREO-Ahead')

#now vertical line
plt.plot_date([timeutc,timeutc],[0,500],'-k', linewidth=2)

#if showinterpolated > 0:  plt.plot_date(rptimes24, rpn24,'ro', label='N interpolated last 24 hours',linewidth=wide,markersize=msize)
#test interpolation
#plt.plot_date(dis.time, rpn7,'-ko', label='n7',linewidth=wide,markersize=5)
#plt.plot_date(sta_time7, sta_den7,'-go', label='den7',linewidth=wide,markersize=5)

plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fsize+2)
ax3.xaxis.set_major_formatter(myformat)
ax3.legend(loc=2,ncol=2,fontsize=fsize-2)

#for y limits check where the maximum and minimum are for DSCOVR and STEREO taken together
plt.ylim([0,np.nanmax(np.nanmax(np.concatenate((dism.den,stam.den[sta_index_future])))+10)])

plt.xlim([plotstart,plotend])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)




########################################### panel 4 Dst
ax4 = fig.add_subplot(414)

#observed Dst Kyoto
plt.plot_date(dst_time, dst,'ko', label='Dst observed',markersize=4)

dst_temerin_li=dst_temerin_li+dst_offset

#now vertical line
plt.plot_date([timeutc,timeutc],[-2000,200],'-k', linewidth=2)

plt.ylabel('Dst [nT]', fontsize=fsize+2)
ax4.xaxis.set_major_formatter(myformat)
plt.xlim([plotstart,plotend])
plt.ylim([np.nanmin(dst_temerin_li)-50,np.nanmax(dst_temerin_li)+20])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

#plot Dst made from L1 and STEREO-A
if verification_mode == 0:
  plt.plot_date(com_time, dst_temerin_li,'-r', label='Dst Temerin & Li 2002',markersize=3, linewidth=1)
  #generic errors of +/-15 nT from test program with STEREO-B data
  error=15
  plt.fill_between(com_time, dst_temerin_li-error, dst_temerin_li+error, alpha=0.2, label='Error for high speed streams')
  #other methods
  #plt.plot_date(com_time+1/24, dst_burton,'-b', label='Dst Burton et al. 1975',markersize=3, linewidth=1)
  #plt.plot_date(com_time+1/24, dst_obrien,'-r', label='Dst OBrien & McPherron 2000',markersize=3, linewidth=1)
  #plt.fill_between(com_time+1/24, dst_burton-error, dst_burton+error, alpha=0.2)
  #plt.fill_between(com_time+1/24, dst_obrien-error, dst_obrien+error, alpha=0.2)



#####**********************
if verification_mode > 0:
  #load saved data l prefix is for loaded 
  [timenowb, sta_ptime, sta_vr, sta_btime, sta_btot, sta_br,sta_bt, sta_bn, rbtime_num, rbtot, rbzgsm, rptime_num, rpv, rpn, lrdst_time, lrdst, lcom_time, ldst_burton, ldst_obrien,ldst_temerin_li]=pickle.load(open(verify_filename,'rb') )  
  plt.plot_date(lcom_time, ldst_burton,'-b', label='Forecast Dst Burton et al. 1975',markersize=3, linewidth=1)
  plt.plot_date(lcom_time, ldst_obrien,'-r', label='Forecast Dst OBrien & McPherron 2000',markersize=3, linewidth=1)
 
ax4.legend(loc=2,ncol=3,fontsize=fsize-2)


#add geomagnetic storm levels
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [0,0],'--k', alpha=0.3, linewidth=1)
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [-50,-50],'--k', alpha=0.3, linewidth=1)
plt.annotate('moderate',xy=(dis.time[0]+realtime_plot_leftadd,-50+2),xytext=(dis.time[0]+realtime_plot_leftadd,-50+2),color='k', fontsize=10)
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [-100,-100],'--k', alpha=0.3, linewidth=1)
plt.annotate('intense',xy=(dis.time[0]+realtime_plot_leftadd,-100+2),xytext=(dis.time[0]+realtime_plot_leftadd,-100+2),color='k', fontsize=10)
plt.plot_date([dis.time[0], dis.time[-1]+realtime_plot_timeadd], [-250,-250],'--k', alpha=0.3, linewidth=1)
plt.annotate('super-storm',xy=(dis.time[0]+realtime_plot_leftadd,-250+2),xytext=(dis.time[0]+realtime_plot_leftadd,-250+2),color='k', fontsize=10)

#plt.tight_layout()
#plt.tight_layout()




plt.figtext(0.99,0.05,'C. Moestl, IWF Graz, Austria', fontsize=12, ha='right')
plt.figtext(0.99,0.025,'https://twitter.com/chrisoutofspace', fontsize=12, ha='right')

plt.figtext(0.01,0.03,'We take no responsibility or liability for the frequency of provision and accuracy of this forecast.' , fontsize=8, ha='left')
plt.figtext(0.01,0.01,'We will not be liable for any losses and damages in connection with using the provided information.' , fontsize=8, ha='left')



print()

#save plot 
if verification_mode == 0:
 filename=outputdirectory+'/predstorm_v1_realtime_stereo_a_plot_'+timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.png'
 filenameeps=outputdirectory+'/predstorm_v1_realtime_stereo_a_plot_'+timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.eps'

 print('Plot saved as png and eps:\n', filename)
 log.write('\n')
 log.write('Plot saved as png and eps:\n'+ filename)
 
 #plt.savefig(filenameeps)
 print('real time plot saved as predstorm_real.png')
 plt.savefig('predstorm_real.png')


#flag if verification_mode is used
if verification_mode > 0:
 filename=outputdirectory+'/predstorm_v1_verify_stereo_a_plot_'+timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.png'
 filenameeps=outputdirectory+'/predstorm_v1_verify_stereo_a_plot_'+timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.eps'

 print('Plot saved as png and eps:\n', filename)
 log.write('\n')
 log.write('Plot saved as png and eps:\n'+ filename)



plt.savefig(filename)
#plt.savefig(filenameeps)












###################################### (4) WRITE OUT RESULTS AND VARIABLES ###############


############# (4a) write prediction variables (plot) to pickle and txt file

filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
              timeutcstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.p'

#make recarrays
combined=np.rec.array([com_time,dst_temerin_li,kp_newell,com_btot,com_bx,com_by,com_bz,com_den,com_vr], \
dtype=[('time','f8'),('dst_temerin_li','f8'),('kp_newell','f8'),('btot','f8'),\
        ('bx','f8'),('by','f8'),('bz','f8'),('den','f8'),('vr','f8')])
        
pickle.dump(combined, open(filename_save, 'wb') )

print('PICKLE: Variables saved in: \n', filename_save, ' \n')
log.write('\n')
log.write('PICKLE: Variables saved in: \n'+ filename_save+ '\n')

filename_save=outputdirectory+'/savefiles/predstorm_v1_realtime_stereo_a_save_'+ \
              timenowstr[0:10]+'-'+timeutcstr[11:13]+'_'+timeutcstr[14:16]+'.txt'
vartxtout=np.zeros([np.size(com_time),9])

#com_time_str= [''  for com_time_str in np.arange(10)]
#for i in np.arange(size(com_time)):
#   com_time_str[i]=str(mdates.num2date(com_time[i]))[0:16]

#vartxtout[:,0]=com_time_str
vartxtout[:,0]=com_time
vartxtout[:,1]=dst_temerin_li
vartxtout[:,2]=kp_newell
vartxtout[:,3]=com_btot
vartxtout[:,4]=com_bx
vartxtout[:,5]=com_by
vartxtout[:,6]=com_bz
vartxtout[:,7]=com_den
vartxtout[:,8]=com_vr
np.savetxt(filename_save, vartxtout, delimiter='',fmt='%8.6f %5.0i %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.0i') 

print('TXT: Variables saved in: \n', filename_save, ' \n ')
log.write('TXT: Variables saved in: \n'+ filename_save+ '\n')




################################# VERIFICATION MODE BRANCH ###############################
#######**********************
if verification_mode > 0:
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


  
