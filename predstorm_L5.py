#PREDSTORM real time solar wind and magnetic storm forecasting with 
# time-shifted data from a spacecraft east of the Sun-Earth line 
# here STEREO-A is used, also suited for data from a 
# possible future L5 mission or interplanetary CubeSats

#Author: C. Moestl, IWF Graz, Austria
#twitter @chrisoutofspace, https://github.com/cmoestl
#started April 2018, last update October 2018

#python 3.5.2 with sunpy and seaborn, ipython 4.2.0

#current status:
#The code works with STEREO-A data and downloads STEREO-A beacon files 
#14 days prior to current time, tested for correctly handling missing PLASTIC files

# things to add: 

# - write prediction variables as txt and pickle file and variables as pickle
# - make a logfile for the predictions with the main results 

# - add error bars for the Temerin/Li Dst model with 1 and 2 sigma, 
#   based on a thorough assessment of errors with the ... stereob_errors program
# - add timeshifts from L1 to Earth
# - add approximate levels of Dst for each location to see the aurora (depends on season)
#   taken from correlations of ovation prime, SuomiNPP data in NASA worldview and Dst 
# - check coordinate conversions, GSE to GSM is ok
# - deal with CMEs at STEREO, because systematically degrades prediction results
# - add metrics ROC etc.

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


##############################################################################
############################# CODE START
##############################################################################

import scipy
import scipy.io
import sys
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import copy
import pdb
import urllib
import json
import ephem
import pickle
import sunpy.time
import seaborn as sns
from pycdf import pycdf

from predstorm_module import time_to_num_cat
from predstorm_module import converttime
from predstorm_module import make_dst_from_wind
from predstorm_module import getpositions
from predstorm_module import convert_GSE_to_GSM
from predstorm_module import sphere2cart
from predstorm_module import convert_RTN_to_GSE_sta_l1

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


############################## INPUT PARAMETERS ######################################


inputfilename='predstorm_L5_input.txt'

#reads all lines as strings
lines = open(inputfilename).read().splitlines()

#whether to show interpolated data points on the DSCOVR input plot
showinterpolated=int(lines[3])

#read in data from omni file -> 1 , from save_file -> 0
data_from_omni_file = int(lines[6]) #

#the time interval for both the observed and predicted wind (** could be longer for predicted wind)
#Delta T in hours, start with 24 hours here (covers 1 night of aurora)
deltat=int(lines[9])

#take 4 solar minimum years as training data for 2018
trainstart=lines[12]
trainend=lines[13]

#synodic solar rotation sun_syn=26.24 #days
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


outputdirectory='real'

#check if directory for output exists
os.path.isdir(outputdirectory)
#if not make new directory
if os.path.isdir(outputdirectory) == False: os.mkdir(outputdirectory)
#also make directory for movie
if os.path.isdir(outputdirectory+'/savefiles') == False: os.mkdir(outputdirectory+'/savefiles')

#check if directory for beacon data exists
os.path.isdir('beacon')
#if not make new directory
if os.path.isdir('beacon') == False: os.mkdir(outputdirectory)




###################################################################################
## VARIABLES

#initialize
#define global variables from OMNI2 hourly dataset
#see http://omniweb.gsfc.nasa.gov/html/ow_data.html
#dataset=473376; # for save file july 2016 
#use this to check on size of OMNI2 hourly data min(np.where(times1==0))
dataset=482136;

#global Variables
spot=np.zeros(dataset) 
btot=np.zeros(dataset) #floating points
bx=np.zeros(dataset) #floating points
by=np.zeros(dataset) #floating points
bz=np.zeros(dataset) #floating points
bzgsm=np.zeros(dataset) #floating points
bygsm=np.zeros(dataset) #floating points

speed=np.zeros(dataset) #floating points
speedx=np.zeros(dataset) #floating points
speed_phi=np.zeros(dataset) #floating points
speed_theta=np.zeros(dataset) #floating points

dst=np.zeros(dataset) #float
kp=np.zeros(dataset) #float

den=np.zeros(dataset) #float
pdyn=np.zeros(dataset) #float
year=np.zeros(dataset)
day=np.zeros(dataset)
hour=np.zeros(dataset)
t=np.zeros(dataset) #index time
times1=np.zeros(dataset) #datetime time

  
  
  
  

########################################################################################## 
######################################## MAIN PROGRAM ####################################
########################################################################################## 

#get current directory
os.system('pwd')
#closes all plots
plt.close('all')

print('-------------------------------------------------')
print()
print('PREDSTORM L5 v1 method for geomagnetic storm and aurora forecasting. ')
print('Christian Moestl, IWF Graz, last update October 2018.')
print()
print('Time shifting magnetic field and plasma data from STEREO-A, ')
print('or from an L5 mission or interplanetary CubeSats, to predict')
print('the solar wind at Earth and the Dst index for magnetic storm strength.')
print()
print()
print('-------------------------------------------------')





######################### (1) get real time DSCOVR data ##################################

#data from http://services.swpc.noaa.gov/products/solar-wind/
#if needed replace with ACE
#http://legacy-www.swpc.noaa.gov/ftpdir/lists/ace/
#get 3 or 7 day data
#url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json'
#url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json'


url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

#download, see URLLIB https://docs.python.org/3/howto/urllib2.html
with urllib.request.urlopen(url_plasma) as url:
    pr = json.loads	(url.read().decode())
with urllib.request.urlopen(url_mag) as url:
    mr = json.loads(url.read().decode())
print('DSCOVR plasma data available')
print(pr[0])
print('DSCOVR MAG data available')
print(mr[0])
#kill first row which stems from the description part
pr=pr[1:]
mr=mr[1:]

#define variables 
#plasma
rptime_str=['']*len(pr)
rptime_num=np.zeros(len(pr))
rpv=np.zeros(len(pr))
rpn=np.zeros(len(pr))
rpt=np.zeros(len(pr))

#mag
rbtime_str=['']*len(mr)
rbtime_num=np.zeros(len(mr))
rbtot=np.zeros(len(mr))
rbzgsm=np.zeros(len(mr))
rbygsm=np.zeros(len(mr))
rbxgsm=np.zeros(len(mr))

#convert variables to numpy arrays
#mag
for k in np.arange(0,len(mr),1):

 #handle missing data, they show up as None from the JSON data file
 if mr[k][6] is None: mr[k][6]=np.nan
 if mr[k][3] is None: mr[k][3]=np.nan
 if mr[k][2] is None: mr[k][2]=np.nan
 if mr[k][1] is None: mr[k][1]=np.nan

 rbtot[k]=float(mr[k][6])
 rbzgsm[k]=float(mr[k][3])
 rbygsm[k]=float(mr[k][2])
 rbxgsm[k]=float(mr[k][1])

 #convert time from string to datenumber
 rbtime_str[k]=mr[k][0][0:16]
 rbtime_num[k]=mdates.date2num(sunpy.time.parse_time(rbtime_str[k]))
 
#plasma
for k in np.arange(0,len(pr),1):
 if pr[k][2] is None: pr[k][2]=np.nan
 rpv[k]=float(pr[k][2]) #speed
 rptime_str[k]=pr[k][0][0:16]
 rptime_num[k]=mdates.date2num(sunpy.time.parse_time(rptime_str[k]))
 if pr[k][1] is None: pr[k][1]=np.nan
 rpn[k]=float(pr[k][1]) #density
 if pr[k][3] is None: pr[k][3]=np.nan
 rpt[k]=float(pr[k][3]) #temperature

#interpolate to 1 hour steps: make an array from last time in hour steps backwards for 24 hours, then interpolate

#this is the last 24 hours in 1 hour timesteps, 25 data points
#for field
rbtimes24=np.arange(rbtime_num[-1]-1,rbtime_num[-1]+1/24,1/24) 
rbtot24=np.interp(rbtimes24,rbtime_num,rbtot)
rbzgsm24=np.interp(rbtimes24,rbtime_num,rbzgsm)
rbygsm24=np.interp(rbtimes24,rbtime_num,rbygsm)
rbxgsm24=np.interp(rbtimes24,rbtime_num,rbxgsm)

#for plasma
rptimes24=np.arange(rptime_num[-1]-1,rptime_num[-1]+1/24,1/24) 
rpv24=np.interp(rptimes24,rptime_num,rpv)
rpn24=np.interp(rptimes24,rptime_num,rpn)

#define times of the future wind, deltat hours after current time
timesfp=np.arange(rptimes24[-1],rptimes24[-1]+1+1/24,1/24)
timesfb=np.arange(rbtimes24[-1],rbtimes24[-1]+1+1/24,1/24)

#set time now 
#for plasma current time
timenowp=rptime_num[-1]
#for B field current time
timenowb=rbtime_num[-1]
timenowstr=str(mdates.num2date(timenowb))[0:16]

#for Dst calculation, interpolate to hourly data
#this is the last 24 hours in 1 hour timesteps, 25 data points
#start on next day 0 UT, so rbtimes7 contains values at every full hour like the real Dst
rtimes7=np.arange(np.ceil(rbtime_num)[0],rbtime_num[-1],1.0000/24)
rbtot7=np.interp(rtimes7,rbtime_num,rbtot)
rbzgsm7=np.interp(rtimes7,rbtime_num,rbzgsm)
rbygsm7=np.interp(rtimes7,rbtime_num,rbygsm)
rbxgsm7=np.interp(rtimes7,rbtime_num,rbxgsm)
rpv7=np.interp(rtimes7,rptime_num,rpv)
rpn7=np.interp(rtimes7,rptime_num,rpn)

#******interpolate NaN values in the hourly interpolated data ******* to add 



#########################################################################################
######################## open file for logging results
logfile='real/results_predstorm_l5_save_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.txt'




log=open(logfile,'wt')

log.write('')
log.write('PREDSTORM L5 v1 results \n')
log.write('For UT time: \n')
log.write(timenowstr)
log.write('\n')




###################### (1b) get real time STEREO-A beacon data
print()

print('load spacecraft and planetary positions')
pos=getpositions('cats/positions_2007_2023_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)

#take position of STEREO-A now
pos_time_now_ind=np.where(timenowb < pos_time_num)[0][0]

sta_r=pos.sta[0][pos_time_now_ind]
#get longitude and latitude
sta_long_heeq=pos.sta[1][pos_time_now_ind]*180/np.pi
sta_lat_heeq=pos.sta[2][pos_time_now_ind]*180/np.pi


timelag_sta_l1=abs(sta_long_heeq)/(360/sun_syn) #days
arrival_time_l1_sta=rtimes7[-1]+timelag_sta_l1
arrival_time_l1_sta_str=str(mdates.num2date(arrival_time_l1_sta))

#feature_sta=mdates.date2num(sunpy.time.parse_time('2018-04-27T01:00:00'))
#arrival_feature_sta_str=str(mdates.num2date(feature_sta+timelag_sta_l1))





print('STEREO-A HEEQ longitude to Earth is ', round(sta_long_heeq,1),' degree.   \
        \nThis is ', round(abs(sta_long_heeq)/60,2),' times the location of L5.')
        
log.write('STEREO-A HEEQ longitude to Earth is '+ str(round(sta_long_heeq,1))+' degree.   \
        \nThis is '+ str(round(abs(sta_long_heeq)/60,2))+' times the location of L5.')




print('STEREO-A HEEQ longitude to Earth is ', round(sta_long_heeq,1),' degree.') 
print('This is ', round(abs(sta_long_heeq)/60,2),' times the location of L5.') 
print('STEREO-A HEEQ latitude is ', round(sta_lat_heeq,1),' degree.') 
print('Earth L1 HEEQ latitude is ',round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1),' degree')
print('Difference HEEQ latitude is ',abs(round(sta_lat_heeq,1)-round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1)),' degree')
print('STEREO-A heliocentric distance is ', round(sta_r,3),' AU.') 
print('The Sun rotation period with respect to Earth is ', sun_syn,' days') 
print('This is a time lag of ', round(timelag_sta_l1,2), ' days.') 
print('Arrival time of now STEREO-A wind at L1:',arrival_time_l1_sta_str[0:16])

#log.write('STEREO-A HEEQ longitude to Earth is ', round(sta_long_heeq,1),' degree.\n') 
#log.write('This is ', round(abs(sta_long_heeq)/60,2),' times the location of L5.\n') 
#log.write('STEREO-A HEEQ latitude is ', round(sta_lat_heeq,1),' degree.\n') 
#log.write('Earth L1 HEEQ latitude is ',round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1),' degree'\n)
#log.write('Difference HEEQ latitude is ',abs(round(sta_lat_heeq,1)-round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1)),' degree'\n)
#log.write('STEREO-A heliocentric distance is ', round(sta_r,3),' AU.') 
#log.write('The Sun rotation period with respect to Earth is ', sun_syn,' days') 
#log.write('This is a time lag of ', round(timelag_sta_l1,2), ' days.') 
#log.write('Arrival time of now STEREO-A wind at L1:',arrival_time_l1_sta_str[0:16])






print()
print('get STEREO-A beacon data from STEREO SCIENCE CENTER')

#only last 2 hours here at NOAA
#http://legacy-www.swpc.noaa.gov/ftpdir/lists/stereo/
#http://legacy-www.swpc.noaa.gov/stereo/STEREO_data.html

#at the STEREO SCIENCE CENTER these are the cdf files for the beacon data, daily
#browse data, ~ 200kb
#https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic/2018/05/STA_LB_PLA_BROWSE_20180502_V12.cdf	
#original data, ~1 MB
#https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic/2018/05/STA_LB_PLA_20180502_V12.cdf	


#make file lists for the last 14 days and download data if not already here

daynowstr=['']*14
sta_pla_file_str=['']*14
sta_mag_file_str=['']*14
http_sta_pla_file_str=['']*14
http_sta_mag_file_str=['']*14

plastic_location='https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic'
impact_location='https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/impact'

#download cdf files if needed
for p in np.arange(0,14):
 stayear=str(mdates.num2date(rtimes7[-1]-14+p))[0:4]
 stamonth=str(mdates.num2date(rtimes7[-1]-14+p))[5:7]
 staday=str(mdates.num2date(rtimes7[-1]-14+p))[8:10]
 daynowstr[p]=stayear+stamonth+staday
 

 #filename convention is
 #https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/impact/2018/05/STA_LB_IMPACT_20180502_V02.cdf

 #filenames
 #plastic
 sta_pla_file_str[p]='STA_LB_PLASTIC_'+daynowstr[p]+'_V12.cdf'
 #impact
 sta_mag_file_str[p]='STA_LB_IMPACT_'+daynowstr[p]+'_V02.cdf' 
 
 #check if file is already there, otherwise download
  
 if not os.path.exists('beacon/'+sta_pla_file_str[p]):
  #download files if they are not here
 
  http_sta_pla_file_str[p]=plastic_location+'/'+stayear+'/'+stamonth+'/'+sta_pla_file_str[p]
  #check if url exists, if not state reason  
  try: urllib.request.urlretrieve(http_sta_pla_file_str[p], 'beacon/'+sta_pla_file_str[p])
  except urllib.error.URLError as e:
   print(' ', http_sta_pla_file_str[p],' ',e.reason)
  
 if not os.path.exists('beacon/'+sta_mag_file_str[p]):
  http_sta_mag_file_str[p]=impact_location+'/'+stayear+'/'+stamonth+'/'+sta_mag_file_str[p]
  try: urllib.request.urlretrieve(http_sta_mag_file_str[p], 'beacon/'+sta_mag_file_str[p])
  except urllib.error.URLError as e:
   print(' ', http_sta_pla_file_str[p],' ',e.reason)
 
################################### 
#now read in all CDF files and stitch to one array
#access cdfs in python works as:
#https://pythonhosted.org/SpacePy/pycdf.html#read-a-cdf
  
#define stereo-a variables with open size, cut 0 later
sta_ptime=np.zeros(0)  
sta_vr=np.zeros(0)  
sta_den=np.zeros(0)  


for p in np.arange(0,14):

  if os.path.exists('beacon/'+sta_pla_file_str[p]):
     sta =  pycdf.CDF('beacon/'+sta_pla_file_str[p])
  #variables Epoch_MAG: Epoch1: CDF_EPOCH [1875]
  #MAGBField: CDF_REAL4 [8640, 3]
  sta_time=mdates.date2num(sta['Epoch1'][...])
  
  sta_dvr=sta['Velocity_RTN'][...][:,0]
  #sta_dvt=sta['Velocity_RTN'][...][:,1]
  #sta_dvn=sta['Velocity_RTN'][...][:,2]
  
  sta_dden=sta['Density'][...]
  
  #missing data are < -1e30
  mis=np.where(sta_time < -1e30)
  sta_time[mis]=np.nan
  mis=np.where(sta_dvr < -1e30)
  sta_dvr[mis]=np.nan
  #mis=np.where(sta_dvt < -1e30)
  #sta_dvt[mis]=np.nan
  #mis=np.where(sta_dvn < -1e30)
  #sta_dvn[mis]=np.nan
  
  mis=np.where(sta_dden < -1e30)
  sta_dden[mis]=np.nan


  sta_ptime=np.append(sta_ptime, sta_time)
  sta_vr=np.append(sta_vr,sta_dvr)
  sta_den=np.append(sta_den,sta_dden)


#sum of nan data points
#sum(np.isnan(sta_ptime))
#same for magnetic field
sta_btime=np.zeros(0)  
sta_br=np.zeros(0)  
sta_bt=np.zeros(0)  
sta_bn=np.zeros(0)  
for p in np.arange(0,14):
  sta =  pycdf.CDF('/Users/chris/python/predstorm/beacon/'+sta_mag_file_str[p])
  #variables Epoch_MAG: CDF_EPOCH [8640]
  #MAGBField: CDF_REAL4 [8640, 3]
  sta_time=mdates.date2num(sta['Epoch_MAG'][...])
  #d stands for dummy
  sta_dbr=sta['MAGBField'][...][:,0]
  sta_dbt=sta['MAGBField'][...][:,1]
  sta_dbn=sta['MAGBField'][...][:,2]

  #append data to array
  sta_btime=np.append(sta_btime, sta_time)
  sta_br=np.append(sta_br,sta_dbr)
  sta_bt=np.append(sta_bt,sta_dbt)
  sta_bn=np.append(sta_bn,sta_dbn)
##check
#plt.plot_date(sta_btime,sta_bn)
#plt.plot_date(sta_ptime,sta_vr)

#make total field variable
sta_btot=np.sqrt(sta_br**2+sta_bt**2+sta_bn**2)

print('STEREO-A data loaded: speed, density, and magnetic field in RTN. ')
print(' ')





####################################### APPLY CORRECTIONS TO STEREO-A data 
#(1) make correction for heliocentric distance of 0.95 AU to L1 position - take position of Earth and STEREO-A from file 
#for B and N, makes a difference of about -5 nT in Dst
earth_r=pos.earth_l1[0][pos_time_now_ind]
sta_btot=sta_btot*(earth_r/sta_r)**-2
sta_br=sta_br*(earth_r/sta_r)**-2
sta_bt=sta_bt*(earth_r/sta_r)**-2
sta_bn=sta_bn*(earth_r/sta_r)**-2
sta_den=sta_den*(earth_r/sta_r)**-2
print()
print('correction 1 to STEREO-A data: decline of B and N by factor ',round(((earth_r/sta_r)**-2),3))


#(2) correction for timing for the Parker spiral 
#1st approximation - because parker spiral at 1 AU is at a 45deg angle, the shift in distance in longitude
#is similar to the shift in radial distance
#*** this may be calculated more exactly with the Parker spiral equations, but will give little difference
#difference in heliocentric distance STEREO-A to Earth
diff_r=earth_r-sta_r
#difference in degree along 1 AU circle
diff_r_deg=diff_r/(2*np.pi*1)*360
#time lag due to the parker spiral near 1 AU	- this is positive because the spiral leads 
#to a later arrival at larger heliocentric distances
time_lag_diff_r=round(diff_r_deg/(360/sun_syn),2)
print('correction 2 to STEREO-A data: approximate Parker spiral time lag in hours: ', round(time_lag_diff_r*24,1))






############################ interpolate STEREO-A to 1 hour times for Dst prediction starting with the last full hour of observations + 1 hour
#so there is a seamless connection from L1 to STEREO-A data
#until the end of the time lagged STEREO-A data
#add timelag to stereo-a original times so everything is shifted
sta_time7=np.arange(rtimes7[-1]+1/24.000,sta_btime[-1]+timelag_sta_l1+time_lag_diff_r,1.000/24.000)
sta_btot7=np.interp(sta_time7,sta_btime+timelag_sta_l1+time_lag_diff_r,sta_btot)
sta_br7=np.interp(sta_time7,sta_btime+timelag_sta_l1+time_lag_diff_r,sta_br)
sta_bt7=np.interp(sta_time7,sta_btime+timelag_sta_l1+time_lag_diff_r,sta_bt)
sta_bn7=np.interp(sta_time7,sta_btime+timelag_sta_l1+time_lag_diff_r,sta_bn)
sta_vr7=np.interp(sta_time7,sta_ptime+timelag_sta_l1+time_lag_diff_r,sta_vr)
sta_den7=np.interp(sta_time7,sta_ptime+timelag_sta_l1+time_lag_diff_r,sta_den)



#(3) conversion from RTN to HEEQ to GSE to GSM - but as if STA was along the Sun-Earth line
print('correction 3 to STEREO-A hourly interpolated data: B RTN to HEEQ to GSE to GSM, as if STEREO-A along the Sun-Earth line.')

#convert STEREO-A RTN data to GSE as if STEREO-A was along the Sun-Earth line
[dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta, pos_time_num)
#GSE to GSM
[sta_br7,sta_bt7,sta_bn7]=convert_GSE_to_GSM(dbr,dbt,dbn,sta_time7)

sta_btot7=np.sqrt(sta_br**2+sta_bt**2+sta_bn**2)


print()
print()



############### calculate Dst for DSCOVR and STEREO-A for last 7 day data with Burton and OBrien

#first try
#the Dst from AER does not seem to be reliable
#http://swe.aer.com/static/DMSPgc/Dst_10day.txt
#dsturl='http://swe.aer.com/static/DMSPgc/Dst_10day.txt'
#rdst_str = urllib.request.urlopen(dsturl).read().decode()
#rdst_size=int(np.round(len(rdst_str)/33))-4
#rdst_time=np.zeros(rdst_size)
#rdst=np.zeros(rdst_size)
#go through each line of the txt file and extract time and Dst value
#for i in np.arange(0,rdst_size):
#  rdst_slice=rdst_str[146+33*i:176+33*i]
  #print(rdst_slice)
  #make a usable string of the time
#  rdst_time_str=rdst_slice[0:10]+' '+rdst_slice[12:17]
  #convert to mdates number
#  rdst_time[i]=mdates.date2num(sunpy.time.parse_time(rdst_time_str))
#  rdst[i]=float(rdst_slice[22:30])
#interpolate to hourly data
#rdst7=np.interp(rtimes7,rdst_time,rdst)
#---------


print('load real time Dst from Kyoto via NOAA')
url_dst='http://services.swpc.noaa.gov/products/kyoto-dst.json'
with urllib.request.urlopen(url_dst) as url:
    dr = json.loads	(url.read().decode())
dr=dr[1:]
#define variables 
#plasma
rdst_time_str=['']*len(dr)
rdst_time=np.zeros(len(dr))
rdst=np.zeros(len(dr))
#convert variables to numpy arrays
#mag
for k in np.arange(0,len(dr),1):
 #handle missing data, they show up as None from the JSON data file
 if dr[k][1] is None: dr[k][1]=np.nan
 rdst[k]=float(dr[k][1])
 #convert time from string to datenumber
 rdst_time_str[k]=dr[k][0][0:16]
 rdst_time[k]=mdates.date2num(sunpy.time.parse_time(rdst_time_str[k]))
#interpolate to rtimes 7 not needed
#rdst7=np.interp(rtimes7,rdst_time,rdst)




#combined array of rtimes7 and sta_time7 times and values rbtot7 sta_btot7 rbzgsm7 sta_bn7 rpv7 sta_vr7  rpn7 sta_den7


#combined dst time
cdst_time=np.concatenate((rtimes7, sta_time7))
cdst_btot=np.concatenate((rbtot7, sta_btot7))
cdst_bx=np.concatenate((rbxgsm7, sta_br7))
cdst_by=np.concatenate((rbygsm7, sta_bt7))
cdst_bz=np.concatenate((rbzgsm7, sta_bn7))
cdst_vr=np.concatenate((rpv7, sta_vr7))
cdst_den=np.concatenate((rpn7, sta_den7))


#if there are nans interpolate them again
if sum(np.isnan(cdst_den)) >0: 
 good= np.where(np.isfinite(cdst_den)) 
 cdst_den=np.interp(cdst_time,cdst_time[good],cdst_den[good])

if sum(np.isnan(cdst_vr)) >0: 
 good= np.where(np.isfinite(cdst_vr)) 
 cdst_vr=np.interp(cdst_time,cdst_time[good],cdst_vr[good])
 






#make Dst index from L1 and STEREO-A solar wind data
#[rdst_burton, rdst_obrien]=make_predstorm_dst(rbtot7, rbzgsm7, rpv7, rpn7, rtimes7)
#[dst_burton, dst_obrien]=make_predstorm_dst(cdst_btot, cdst_bz, cdst_vr, cdst_den, cdst_time)


#make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):#
[dst_burton, dst_obrien, dst_temerin_li]=make_dst_from_wind(cdst_btot, cdst_bx,cdst_by,cdst_bz, cdst_vr,cdst_vr, cdst_den, cdst_time)



print('calculate Dst prediction from L1 and STEREO-A beacon data')


#not used currently
#################################  get OMNI training data ##############################
#download from  ftp://nssdcftp.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat

data_from_omni_file=0
if data_from_omni_file == 1:
 getdata()
 converttime()
 pickle.dump([spot,btot,bx,by,bz,bygsm,bzgsm,speed,speedx, dst,kp, den,pdyn,year,day,hour,times1], open( "cats/omni2save_april2018.p", "wb" ) ) 
else: [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "cats/omni2save_april2018.p", "rb" ) )



################################################## plot DSCOVR and STEREO-A data ##################################





#check which parts of the array begin after time now rbtime_num[-1] and rptime_num[-1] and end with plot end
#for plasma add both timeshifts for longitude and parker spiral
sta_ptime_lag=sta_ptime+timelag_sta_l1+time_lag_diff_r
sta_pindex_future=np.where(np.logical_and(sta_ptime_lag > rptime_num[-1],sta_ptime_lag <rptime_num[-1]+realtime_plot_timeadd))
#for field
sta_btime_lag=sta_btime+timelag_sta_l1+time_lag_diff_r
sta_bindex_future=np.where(np.logical_and(sta_btime_lag > rbtime_num[-1],sta_btime_lag <rbtime_num[-1]+realtime_plot_timeadd))



sns.set_context("talk")     
sns.set_style("darkgrid")  
fig=plt.figure(1,figsize=(12,10)) #fig=plt.figure(1,figsize=(14,14))
weite=1
fsize=11
msize=5

################################# panel 1
ax4 = fig.add_subplot(411)
plt.plot_date(rbtime_num, rbtot,'-k', label='B total L1', linewidth=weite)
if showinterpolated > 0: plt.plot_date(rbtimes24, rbtot24,'ro', label='B total interpolated last 24 hours',linewidth=weite,markersize=msize)
plt.plot_date(rbtime_num, rbzgsm,'-g', label='Bz GSM L1',linewidth=weite)
if showinterpolated > 0: plt.plot_date(rbtimes24, rbzgsm24,'go', label='Bz GSM interpolated last 24 hours',linewidth=weite,markersize=msize)

#indicate 0 level for Bz
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [0,0],'--k', alpha=0.5, linewidth=1)


#plot STEREO-A data with timeshift	
plt.plot_date(sta_btime_lag[sta_bindex_future], sta_btot[sta_bindex_future],'-r', linewidth=weite, label='B STEREO-Ahead')
plt.plot_date(sta_btime_lag[sta_bindex_future], sta_bn[sta_bindex_future],markersize=0,linestyle='-', color='darkolivegreen', linewidth=weite, label='Bn RTN STEREO-Ahead')


#test hourly interpolation 
#plt.plot_date(rtimes7, rbtot7,'-ko', label='B7',linewidth=weite,markersize=5)
#plt.plot_date(rtimes7, rbzgsm7,'-go', label='Bz7',linewidth=weite,markersize=5)
#plt.plot_date(sta_time7, sta_btot7,'-ko', label='B7',linewidth=weite, markersize=5)
plt.plot_date(sta_time7, sta_bn7,'-go', label='Bz GSM STEREO-Ahead',linewidth=weite,markersize=5)

#plt.plot_date(sta_time7, dbn,'-bo', label='Bg',linewidth=weite,markersize=5)




plt.ylabel('Magnetic field [nT]',  fontsize=fsize+2)
#myformat = mdates.DateFormatter('%Y %b %d %Hh')
myformat = mdates.DateFormatter('%b %d %Hh')

ax4.xaxis.set_major_formatter(myformat)
ax4.legend(loc='upper left', fontsize=fsize-2,ncol=4)
plt.xlim([np.ceil(rbtime_num)[0]+realtime_plot_leftadd,rbtime_num[-1]+realtime_plot_timeadd])

#for y limits check where the maximum and minimum are for DSCOVR and STEREO taken together
#negative is surely in bz, positive in btot

bplotmax=np.nanmax(np.concatenate((rbtot,sta_btot[sta_bindex_future])))+5
plt.ylim(np.nanmin(np.concatenate((rbzgsm,sta_bn[sta_bindex_future]))-5), bplotmax)

plt.title('L1 DSCOVR real time solar wind from NOAA SWPC for '+ str(mdates.num2date(timenowb))[0:16]+ ' UT   STEREO-A beacon', fontsize=16)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


#vertical line and indicator for prediction and observation
plt.plot_date([timenowb,timenowb],[-100,100],'-k', linewidth=2)



###################################### panel 2
ax5 = fig.add_subplot(412)
#add speed levels
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [400,400],'--k', alpha=0.3, linewidth=1)
plt.annotate('slow',xy=(rtimes7[0]+realtime_plot_leftadd,400),xytext=(rtimes7[0]+realtime_plot_leftadd,400),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [800,800],'--k', alpha=0.3, linewidth=1)
plt.annotate('fast',xy=(rtimes7[0]+realtime_plot_leftadd,800),xytext=(rtimes7[0]+realtime_plot_leftadd,800),color='k', fontsize=10	)

plt.plot_date(rptime_num, rpv,'-k', label='speed L1',linewidth=weite)
if showinterpolated > 0: plt.plot_date(rptimes24, rpv24,'ro', label='V interpolated last 24 hours',linewidth=weite,markersize=msize)
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]+realtime_plot_timeadd])
#plt.plot_date(rtimes7, rpv7,'-ko', label='B7',linewidth=weite)


#plot STEREO-A data with timeshift	and savgol filter
from scipy.signal import savgol_filter
plt.plot_date(sta_ptime_lag[sta_pindex_future], savgol_filter(sta_vr[sta_pindex_future],5,1),'-r', linewidth=weite, label='speed STEREO-Ahead')

#now vertical line
plt.plot_date([timenowb,timenowb],[0,4000],'-k', linewidth=2)

#test interpolation
#plt.plot_date(rtimes7, rpv7,'-ko', label='v7',linewidth=weite,markersize=5)
#plt.plot_date(sta_time7, sta_vr7,'-go', label='Vr7',linewidth=weite,markersize=5)


plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fsize+2)
ax5.xaxis.set_major_formatter(myformat)
ax5.legend(loc=2,fontsize=fsize-2,ncol=2)
plt.xlim([np.ceil(rbtime_num)[0]+realtime_plot_leftadd,rbtime_num[-1]+realtime_plot_timeadd])

#for y limits check where the maximum and minimum are for DSCOVR and STEREO taken together
vplotmax=np.nanmax(np.concatenate((rpv,savgol_filter(sta_vr[sta_pindex_future],15,1))))+100
plt.ylim(np.nanmin(np.concatenate((rpv,savgol_filter(sta_vr[sta_pindex_future],15,1)))-50), vplotmax)

plt.annotate('now',xy=(timenowb,vplotmax-100),xytext=(timenowb+0.05,vplotmax-100),color='k', fontsize=14)
#plt.annotate('observation',xy=(timenowb,bplotmax-3),xytext=(timenowb-0.55,bplotmax-3),color='k', fontsize=15)
#plt.annotate('prediction',xy=(timenowb,bplotmax-3),xytext=(timenowb+0.45,bplotmax-3),color='b', fontsize=15)


#plt.ylim([np.nanmin(rpv)-50,np.nanmax(rpv)+100])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


########################################### panel 3 density
ax6 = fig.add_subplot(413)
plt.plot_date(rptime_num, rpn,'-k', label='density L1',linewidth=weite)
if showinterpolated > 0:  plt.plot_date(rptimes24, rpn24,'ro', label='N interpolated last 24 hours',linewidth=weite,markersize=msize)


#plot STEREO-A data with timeshift	
plt.plot_date(sta_ptime_lag[sta_pindex_future], savgol_filter(sta_den[sta_pindex_future],5,1),'-r', linewidth=weite, label='density STEREO-Ahead')

#now vertical line
plt.plot_date([timenowb,timenowb],[0,500],'-k', linewidth=2)


#test interpolation
#plt.plot_date(rtimes7, rpn7,'-ko', label='n7',linewidth=weite,markersize=5)
#plt.plot_date(sta_time7, sta_den7,'-go', label='den7',linewidth=weite,markersize=5)

plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fsize+2)
ax6.xaxis.set_major_formatter(myformat)
ax6.legend(loc=2,ncol=2,fontsize=fsize-2)


#for y limits check where the maximum and minimum are for DSCOVR and STEREO taken together
plt.ylim([0,np.nanmax(np.nanmax(np.concatenate((rpn,sta_den[sta_pindex_future])))+10)])


plt.xlim([np.ceil(rbtime_num)[0]+realtime_plot_leftadd,rbtime_num[-1]+realtime_plot_timeadd])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

########################################### panel 4 Dst
ax6 = fig.add_subplot(414)

#model Dst for L1 only
#added timeshift of 1 hour for L1 to Earth! This should be different for each timestep to be exact
#plt.plot_date(rtimes7+1/24, rdst_burton,'-b', label='Dst Burton et al. 1975',markersize=3, linewidth=1)
#plt.plot_date(rtimes7+1/24, rdst_obrien,'-r', label='Dst OBrien & McPherron 2000',markersize=3, linewidth=1)

#plot Dst made from L1 and STEREO-A
if verification_mode == 0:
  #plt.plot_date(cdst_time+1/24, dst_burton,'-b', label='Dst Burton et al. 1975',markersize=3, linewidth=1)
  #plt.plot_date(cdst_time+1/24, dst_obrien,'-r', label='Dst OBrien & McPherron 2000',markersize=3, linewidth=1)
  plt.plot_date(cdst_time+1/24, dst_temerin_li,'-r', label='Dst Temerin & Li 2002',markersize=3, linewidth=1)
  #generic errors of +/-15 nT from test program with STEREO-B data **update with TL
  error=15.5
  #plt.fill_between(cdst_time+1/24, dst_burton-error, dst_burton+error, alpha=0.2)
  #plt.fill_between(cdst_time+1/24, dst_obrien-error, dst_obrien+error, alpha=0.2)
  plt.fill_between(cdst_time+1/24, dst_temerin_li-error, dst_temerin_li+error, alpha=0.2, label='Error for high speed streams')


#real Dst
#for Kyoto
plt.plot_date(rdst_time, rdst,'ko', label='Dst observed',markersize=4)
#for AER
#plt.plot_date(rtimes7, rdst7,'ko', label='Dst observed',markersize=4)

#now vertical line
plt.plot_date([timenowb,timenowb],[-2000,200],'-k', linewidth=2)

plt.ylabel('Dst [nT]', fontsize=fsize+2)
ax6.xaxis.set_major_formatter(myformat)
plt.xlim([np.ceil(rbtime_num)[0]+realtime_plot_leftadd,rbtime_num[-1]+realtime_plot_timeadd])
plt.ylim([np.nanmin(dst_temerin_li)-50,50])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)




#for verification

if verification_mode > 0:
  
  #load saved data l prefix is for loaded 
  [timenowb, sta_ptime, sta_vr, sta_btime, sta_btot, sta_br,sta_bt, sta_bn, rbtime_num, rbtot, rbzgsm, rptime_num, rpv, rpn, lrdst_time, lrdst, lcdst_time, ldst_burton, ldst_obrien,ldst_temerin_li]=pickle.load(open(verify_filename,'rb') )  
  plt.plot_date(lcdst_time+1/24, ldst_burton,'-b', label='Forecast Dst Burton et al. 1975',markersize=3, linewidth=1)
  plt.plot_date(lcdst_time+1/24, ldst_obrien,'-r', label='Forecast Dst OBrien & McPherron 2000',markersize=3, linewidth=1)
 
ax6.legend(loc=2,ncol=3,fontsize=fsize-2)


#add geomagnetic storm levels
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [0,0],'--k', alpha=0.3, linewidth=1)
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [-50,-50],'--k', alpha=0.3, linewidth=1)
plt.annotate('moderate',xy=(rtimes7[0]+realtime_plot_leftadd,-50+2),xytext=(rtimes7[0]+realtime_plot_leftadd,-50+2),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [-100,-100],'--k', alpha=0.3, linewidth=1)
plt.annotate('intense',xy=(rtimes7[0]+realtime_plot_leftadd,-100+2),xytext=(rtimes7[0]+realtime_plot_leftadd,-100+2),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]+realtime_plot_timeadd], [-250,-250],'--k', alpha=0.3, linewidth=1)
plt.annotate('super-storm',xy=(rtimes7[0]+realtime_plot_leftadd,-250+2),xytext=(rtimes7[0]+realtime_plot_leftadd,-250+2),color='k', fontsize=10)



plt.tight_layout()

#save plot 

if verification_mode == 0:
 filename='real/predstorm_realtime_stereo_l1_plot_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.jpg'

#flag if verification_mode is used
if verification_mode > 0:
 filename='real/verify_predstorm_realtime_stereo_l1_plot_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.jpg'

 
plt.savefig(filename)
#filename='real/predstorm_realtime_input_1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.eps'
#plt.savefig(filename)

filename_save='real/savefiles/variables_predstorm_l5_save_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.p'
print('All variables for plot saved in ', filename_save, ' for later verification usage.')

pickle.dump([timenowb, sta_ptime, sta_vr, sta_btime, sta_btot, sta_br,sta_bt, sta_bn, rbtime_num, rbtot, rbzgsm, rptime_num, rpv, rpn, rdst_time, rdst, cdst_time, dst_burton, dst_obrien,dst_temerin_li], open(filename_save, "wb" ) )
#load with
#[sta_ptime, sta_vr, rdst_time, rdst, cdst_time, dst_burton, dst_obrien]=pickle.load(open(f,'rb') )











print()
print()
print('-------------------------------------------------')
print()

print()
#print('Predicted maximum of B total in next 24 hours:')
#print(np.nanmax(sta_btot),' nT')
#print('Predicted minimum of Bz GSM in next 24 hours:')
#print(np.nanmin(bzp),' nT')
#print('Predicted maximum V in next 24 hours:')
#print(int(round(np.nanmax(speedp,0))),' km/s')

#check future times in combined Dst 
future_times=np.where(cdst_time > timenowb)



if verification_mode > 0:
  print('Verification results for interval:')
  

  #rdst_time rdst includes the observed Dst of the event
  verify_ind_obs=np.where(np.logical_and(rdst_time > verify_int_start,rdst_time < verify_int_end))
  #lcdst_time ldst_burton ldst_obrien are the forecasted indices
  verify_ind_for=np.where(np.logical_and(lcdst_time > verify_int_start,lcdst_time < verify_int_end-1/24))
 
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
  print('Time of Dst minimum Burton:', str(mdates.num2date(lcdst_time[verify_ind_for][np.argmin(ldst_burton[verify_ind_for])]+1/3600))[0:16] )
  print('Time of Dst minimum OBrien:', str(mdates.num2date(lcdst_time[verify_ind_for][np.argmin(ldst_obrien[verify_ind_for])]+1/3600))[0:16] )

  print('Time difference of Dst minimum Burton:', int( (lcdst_time[verify_ind_for][np.argmin(ldst_burton[verify_ind_for])]-rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])])*24), ' hours' )
  print('Time difference of Dst minimum OBrien:', int( (lcdst_time[verify_ind_for][np.argmin(ldst_obrien[verify_ind_for])]-rdst_time[verify_ind_obs][np.argmin(rdst[verify_ind_obs])])*24), ' hours' )


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




print('PREDSTORM L5 (STEREO-A) prediction results:')
print()
print('Current time: ', rbtime_str[-1], ' UT')

print()
print('Predicted minimum of Dst Burton/OBrien:')
print(int(round(np.nanmin(dst_burton[future_times]))), ' / ', int(round(np.nanmin(dst_obrien[future_times]))),'  nT')

mindst_time=cdst_time[future_times[0][0]+np.nanargmin(dst_burton[future_times])]
print('at time:')


#add 1 minute manually because of rounding errors in time 19:59:9999 etc.
print(str(mdates.num2date(mindst_time+1/(24*60)))[0:16])

print()
print()

#write out times of storm levels
print('times of moderate storm level in prediction')
print()

storm_times_ind=np.where(dst_burton[future_times] < -50)[0]
#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
 for i in np.arange(0,len(storm_times_ind),1):
  print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
 print('Burton model:')

print()
storm_times_ind=np.where(dst_obrien[future_times] < -50)[0]
if len(storm_times_ind) >0:   
 for i in np.arange(0,len(storm_times_ind),1):
  print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
 print('OBrien model:')



print()
print()

print('times of intense storm level')
storm_times_ind=np.where(dst_burton[future_times] < -100)[0]

#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
  print('Burton model:')

storm_times_ind=np.where(dst_obrien[future_times] < -100)[0]
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]))[0:16])
  print('OBrien model:')
 


print()
print()
print('times of super storm level')
storm_times_ind=np.where(dst_burton[future_times] < -200)[0]

#when there are storm times above this level, indicate:
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]+1/(24*60)))[0:16])
  print('Burton model:')

storm_times_ind=np.where(dst_obrien[future_times] < -200)[0]
if len(storm_times_ind) >0:   
  for i in np.arange(0,len(storm_times_ind),1):
   print(str(mdates.num2date(cdst_time[future_times][storm_times_ind][i]))[0:16])
  print('OBrien model:')

print()
print()



log.write('Current time: '+ rbtime_str[-1]+ ' UT')
log.write('')
log.write('Predicted minimum of Dst Burton/OBrien:')
log.write(str(int(round(np.nanmin(dst_burton[future_times])))) + '/' + str(int(round(np.nanmin(dst_obrien[future_times]))))+'  nT')
log.write('at time:')



log.close() 

filename_save='real/savefiles/variables_realtime_stereo_l1_save_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.p'
print('All results saved in ', filename_save)
sys.exit()


##########################################################################################
################################# CODE STOP ##############################################
##########################################################################################



