##predstorm real time solar wind forecasting

#predicting the L1 solar wind and Dst index with unsupervised pattern recognition
#algorithms Riley et al. 2017, Owens et al. 2017
#soon Möstl et al. 2018 3DCORE, Reiss et al. background wind
#Author: C. Moestl 
#started April 2018

#python 3.5 with sunpy and seaborn

#Things to do:

#method
#semi-supervised learning: add known intervals of ICMEs, MFRs and CIRs in the training data
#helcats lists for ICMEs at Wind since 2007
#HSS e.g. https://link.springer.com/article/10.1007%2Fs11207-013-0355-z
#add 3DCORE profiles
#see https://www.quantamagazine.org/machine-learnings-amazing-ability-to-predict-chaos-20180418/
#reservoir computing
#https://en.wikipedia.org/wiki/Pattern_recognition


#training data:
#use stereo one hour data as training data set, corrected for 1 AU
#use VEX and MESSENGER as tests for HelioRing like forecasts, use STEREO at L5 for training data of the last few days

#forecast plot:
#add approximate levels of Dst for each location to see aurora, taken from ovation prime/worldview and Dst 
#add Temerin and Li method and kick out Burton/OBrien; make error bars for Dst
#take mean of ensemble forecast for final blue line forecast or only best match?


#combined L1/L5 forecast
#most important: implement pattern recognition for STEREO-A streams, and link this to the most probably outcome days later at L1
#train with STB data around the location where STA is at the moment
#das ist sofort ein paper!

# - prediction als txt file und variables rauschreiben
# - add prediction time series of solar wind in files
# - make a logfile for the predictions
# - coordinate conversions checken (GSE to GSM ok)
# - problem sind cmes bei stereo, mit listen ausschliessen aus training data
# - eigene tests mit semi-supervised, also sodass known intervals benutzt werden)
# - eigenes programm zum hindsight testen mit stereo-a und omni (da ist dst auch drin)
# (das ist alles ein proposal... zb h2020, erc)
# irgendwann forecast wie wetter app machen auf website und verschiedene locations und ovation (aber eben nur für background wond)
# CCMC Bz berücksichtigen
# fundamental: how much is Bz stable for HSS from L5 to L1? are there big changes?
# is Bz higher for specific locations with respect to the HCS and the solar equator? 
# probabilities for magnetic storm magnitude, probabilities for aurora for many locations
# link with weather forcecast, wie am Handy in der app 
# am wichtigsten: validation sodass die % stimmen!!

#temporal and spatial coherence
#(1) STEREO-A and Earth are within 0.1° heliospheric latitude, so they could see similar parts of the stream + (2) the coronal hole hasn't changed much in the last few days.



#IMPORT 
from scipy import stats
import scipy
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sunpy.time
import time
import pickle
import seaborn as sns
import os
import copy
import pdb
import urllib
import json
import ephem
from pycdf import pycdf
import scipy.io
import pdb

import warnings
warnings.filterwarnings('ignore')
################################## INPUT PARAMETERS ######################################

#whether to show interpolated data points on the DSCOVR input plot
showinterpolated=0

#read in data from omni file -> 1 , from save_file -> 0
data_from_omni_file = 0 #

#the time interval for both the observed and predicted wind (** could be longer for predicted wind)
#Delta T in hours, start with 24 hours here (covers 1 night of aurora)
deltat=24

#take 4 solar minimum years as training data for 2018
trainstart='2006-Jan-01 00:00'
trainend='2010-Jan-01 00:00'


#synodic solar rotation 
sun_syn=26.24 #days
#use other values for equatorial coronal holes?
sun_syn=25.5 #days



#how far to see in the future with STEREO-A data to the right of the current time
realtime_plot_timeadd=7

#to shift the left beginning of the plot
realtime_plot_leftadd=2


#to get older data for plotting Burton/OBrien Dst for verification
verification_mode=0
#verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-05-04-10_00.p'
#verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-05-29-12_32.p'
verify_filename='real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-06-16-07_13.p'
#intervals for verification
verify_int_start=mdates.date2num(sunpy.time.parse_time('2018-05-31 12:00:00'))
verify_int_end=mdates.date2num(sunpy.time.parse_time('2018-06-02 23:00:00'))

#######################################################






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

  

def getdata():

 #statt NaN waere besser linear interpolieren
 #lese file ein:
 
 #FORMAT(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2, F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1)
 #1963   1  0 1771 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9  7  23    -6  119 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 999.9 999.9 99999 99999 99.9

 
 j=0
 print('start reading variables from file')
 with open('/Users/chris/python/data/omni_data/omni2_all_years.dat') as f:
  for line in f:
   line = line.split() # to deal with blank 
   #print line #41 is Dst index, in nT
   dst[j]=line[40]
   kp[j]=line[38]
   
   if dst[j] == 99999: dst[j]=np.NaN
   #40 is sunspot number
   spot[j]=line[39]
   #if spot[j] == 999: spot[j]=NaN

   #25 is bulkspeed F6.0, in km/s
   speed[j]=line[24]
   if speed[j] == 9999: speed[j]=np.NaN
 
   #get speed angles F6.1
   speed_phi[j]=line[25]
   if speed_phi[j] == 999.9: speed_phi[j]=np.NaN

   speed_theta[j]=line[26]
   if speed_theta[j] == 999.9: speed_theta[j]=np.NaN
   #convert speed to GSE x see OMNI website footnote
   speedx[j] = - speed[j] * np.cos(np.radians(speed_theta[j])) * np.cos(np.radians(speed_phi[j]))



   #9 is total B  F6.1 also fill ist 999.9, in nT
   btot[j]=line[9]
   if btot[j] == 999.9: btot[j]=np.NaN

   #GSE components from 13 to 15, so 12 to 14 index, in nT
   bx[j]=line[12]
   if bx[j] == 999.9: bx[j]=np.NaN
   by[j]=line[13]
   if by[j] == 999.9: by[j]=np.NaN
   bz[j]=line[14]
   if bz[j] == 999.9: bz[j]=np.NaN
 
   #GSM
   bygsm[j]=line[15]
   if bygsm[j] == 999.9: bygsm[j]=np.NaN
 
   bzgsm[j]=line[16]
   if bzgsm[j] == 999.9: bzgsm[j]=np.NaN 	
 
 
   #24 in file, index 23 proton density /ccm
   den[j]=line[23]
   if den[j] == 999.9: den[j]=np.NaN
 
   #29 in file, index 28 Pdyn, F6.2, fill values sind 99.99, in nPa
   pdyn[j]=line[28]
   if pdyn[j] == 99.99: pdyn[j]=np.NaN 		
 
   year[j]=line[0]
   day[j]=line[1]
   hour[j]=line[2]
   j=j+1     

 print('done reading OMNI2 variables from file')
 print(j, ' datapoints')   #for reading data from OMNI file
 
############################################################# 


def time_to_num_cat(time_in):  
  #for time conversion  
  #for all catalogues
  #time_in is the time in format: 2007-11-17T07:20:00
  #for times help see: 
  #http://docs.sunpy.org/en/latest/guide/time.html
  #http://matplotlib.org/examples/pylab_examples/date_demo2.html
  
  j=0
  #time_str=np.empty(np.size(time_in),dtype='S19')
  time_str= ['' for x in range(len(time_in))]
  #=np.chararray(np.size(time_in),itemsize=19)
  time_num=np.zeros(np.size(time_in))
 
  for i in time_in:
   #convert from bytes (output of scipy.readsav) to string
   time_str[j]=time_in[j][0:16].decode()+':00'
   year=int(time_str[j][0:4])
   #convert time to sunpy friendly time and to matplotlibdatetime
   #only for valid times so 9999 in year is not converted
   #pdb.set_trace()
   if year < 2100:
    	  time_num[j]=mdates.date2num(sunpy.time.parse_time(time_str[j]))
   j=j+1  
   #the date format in matplotlib is e.g. 735202.67569444
   #this is time in days since 0001-01-01 UTC, plus 1.
  return time_num


def converttime():

 #http://docs.sunpy.org/en/latest/guide/time.html
 #http://matplotlib.org/examples/pylab_examples/date_demo2.html

 print('convert time start')
 for index in range(0,dataset):
      #first to datetimeobject 
      timedum=datetime.datetime(int(year[index]), 1, 1) + datetime.timedelta(day[index] - 1) +datetime.timedelta(hours=hour[index])
      #then to matlibplot dateformat:
      times1[index] = matplotlib.dates.date2num(timedum)
      #print time
      #print year[index], day[index], hour[index]
 print('convert time done')   #for time conversion

############################################################
  



def getpositions(filename):  
    pos=scipy.io.readsav(filename)  
    print
    print('positions file:', filename) 
    return pos

  
  
  
  
  
  
  
def make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):

 #this makes from synthetic or observed solar wind the Dst index	
 #all nans in the input data must be removed prior to function call
 #3 models are calculated: Burton et al., OBrien/McPherron, and Temerin/Li

 #btot_in IMF total field, in nT, GSE or GSM (they are the same)
 #bx_in - the IMF Bx field in nT, GSE or GSM (they are the same)
 #by_in - the IMF By field in nT, GSM
 #bz_in - the IMF Bz field in nT, GSM
 #v_in - the speed in km/s
 #vx_in - the solar wind speed x component (GSE is similar to GSM) in km/s
 #time_in - the time in matplotlib date format

 #define variables
 Ey=np.zeros(len(bz_in))
 #dynamic pressure
 pdyn1=np.zeros(len(bz_in))
 protonmass=1.6726219*1e-27  #kg
 #assume pdyn is only due to protons
 pdyn1=density_in*1e6*protonmass*(v_in*1e3)**2*1e9  #in nanoPascal
 dststar1=np.zeros(len(bz_in))
 dstcalc1=np.zeros(len(bz_in))
 dststar2=np.zeros(len(bz_in))
 dstcalc2=np.zeros(len(bz_in))
 
 #array with all Bz fields > 0 to 0 
 bz_in_negind=np.where(bz_in > 0)  
  #important: make a deepcopy because you manipulate the input variable
 bzneg=copy.deepcopy(bz_in)
 bzneg[bz_in_negind]=0

 #define interplanetary electric field 
 Ey=v_in*abs(bzneg)*1e-3; #now Ey is in mV/m
 
 ######################## model 1: Burton et al. 1975 
 Ec=0.5  
 a=3.6*1e-5
 b=0.2*100 #*100 wegen anderer dynamic pressure einheit in Burton
 c=20  
 d=-1.5/1000 
 for i in range(len(bz_in)-1):
  if Ey[i] > Ec:
   F=d*(Ey[i]-Ec) 
  else: F=0
  #Burton 1975 seite 4208: Dst=Dst0+bP^1/2-c   / und b und c positiv  
  #this is the ring current Dst
  deltat_sec=(time_in[i+1]-time_in[i])*86400 #timesyn is in days - convert to seconds
  dststar1[i+1]=(F-a*dststar1[i])*deltat_sec+dststar1[i];  #deltat must be in seconds
  #this is the Dst of ring current and magnetopause currents 
  dstcalc1[i+1]=dststar1[i+1]+b*np.sqrt(pdyn1[i+1])-c; 

 ###################### model 2: OBrien and McPherron 2000 
 #constants
 Ec=0.49
 b=7.26  
 c=11  #nT
 for i in range(len(bz_in)-1):
  if Ey[i] > Ec:            #Ey in mV m
   Q=-4.4*(Ey[i]-Ec) 
  else: Q=0
  tau=2.4*np.exp(9.74/(4.69+Ey[i])) #tau in hours
  #this is the ring current Dst
  deltat_hours=(time_in[i+1]-time_in[i])*24 #time_in is in days - convert to hours
  dststar2[i+1]=((Q-dststar2[i]/tau))*deltat_hours+dststar2[i] #t is pro stunde, time intervall ist auch 1h
  #this is the Dst of ring current and magnetopause currents 
  dstcalc2[i+1]=dststar2[i+1]+b*np.sqrt(pdyn1[i+1])-c; 
  
 
 
 ######## model 3: Xinlin Li LASP Colorado and Mike Temerin
 
  
 #2002 version 
 
 #define all terms
 dst1=np.zeros(len(bz_in))
 dst2=np.zeros(len(bz_in))
 dst3=np.zeros(len(bz_in))
 pressureterm=np.zeros(len(bz_in))
 directterm=np.zeros(len(bz_in))
 offset=np.zeros(len(bz_in))
 dst_temerin_li_out=np.zeros(len(bz_in))
 bp=np.zeros(len(bz_in))
 bt=np.zeros(len(bz_in))
 
 
 #define inital values (needed for convergence, see Temerin and Li 2002 note)
 dst1[0:10]=-15
 dst2[0:10]=-13
 dst3[0:10]=-2
 
 

 #define all constants
 p1=0.9
 p2=2.18e-4
 p3=14.7
 
 # these need to be found with a fit for 1-2 years before calculation
 # taken from the TL code:    offset_term_s1 = 6.70       ;formerly named dsto
 #   offset_term_s2 = 0.158       ;formerly hard-coded     2.27 for 1995-1999
 #   offset_term_s3 = -0.94       ;formerly named phasea  -1.11 for 1995-1999
 #   offset_term_s4 = -0.00954    ;formerly hard-coded
 #   offset_term_s5 = 8.159e-6    ;formerly hard-coded
 
 
 #s1=6.7
 #s2=0.158
 #s3=-0.94
 #set by myself as a constant in the offset term
 #s4=-3
 
 
 #s1=-2.788
 #s2=1.44
 #s3=-0.92
 #set by myself as a constant in the offset term
 #s4=-3
 #s4 and s5 as in the TL 2002 paper are not used due to problems with the time
 #s4=-1.054*1e-2
 #s5=8.6e-6


 #found by own offset optimization for 2015
 s1=4.29
 s2=5.94
 s3=-3.97
 
 a1=6.51e-2
 a2=1.37
 a3=8.4e-3  
 a4=6.053e-3
 a5=1.12e-3
 a6=1.55e-3
 
 tau1=0.14 #days
 tau2=0.18 #days
 tau3=9e-2 #days
 
 b1=0.792
 b2=1.326
 b3=1.29e-2
 
 c1=-24.3
 c2=5.2e-2

 #Note: vx has to be used with a positive sign throughout the calculation
 
 
 
 #----------------------------------------- loop over each timestep
 for i in np.arange(1,len(bz_in)-1):

      
  #t time in days since beginning of 1995   #1 Jan 1995 in Julian days
#  t1=sunpy.time.julian_day(mdates.num2date(time_in[i]))-sunpy.time.julian_day('1995-1-1 00:00')
  t1=sunpy.time.julian_day(mdates.num2date(time_in[i]))-sunpy.time.julian_day('2015-1-1 00:00')
  
 
  yearli=365.24 
  tt=2*np.pi*t1/yearli
  ttt=2*np.pi*t1
  alpha=0.078
  beta=1.22
  cosphi=np.sin(tt+alpha)*np.sin(ttt-tt-beta)*(9.58589*1e-2)+np.cos(tt+alpha)*(0.39+0.104528*np.cos(ttt-tt-beta))
 
  #equation 1 use phi from equation 2
  sinphi=(1-cosphi**2)**0.5
  
  pressureterm[i]=(p1*(btot_in[i]**2)+density_in[i]*((p2*((v_in[i])**2)/(sinphi**2.52))+p3))**0.5
  
  #2 directbzterm 
  directterm[i]=0.478*bz_in[i]*(sinphi**11.0)

  #3 offset term - the last two terms were cut because don't make sense as t1 rises extremely for later years
  offset[i]=s1+s2*np.sin(2*np.pi*t1/yearli+s3)
  #or just set it constant
  #offset[i]=-5
  bt[i]=(by_in[i]**2+bz_in[i]**2)**0.5  
  #mistake in 2002 paper - bt is similarly defined as bp (with by bz); but in Temerin and Li's code (dst.pro) bp depends on by and bx
  bp[i]=(by_in[i]**2+bx_in[i]**2)**0.5  
  #contains t1, but in cos and sin 
  dh=bp[i]*np.cos(np.arctan2(bx_in[i],by_in[i])+6.10) * ((3.59e-2)*np.cos(2*np.pi*t1/yearli+0.04)-2.18e-2*np.sin(2*np.pi*t1-1.60))
  theta_li=-(np.arccos(-bz_in[i]/bt[i])-np.pi)/2
  exx=1e-3*abs(vx_in[i])*bt[i]*np.sin(theta_li)**6.1
  #t1 and dt are in days
  dttl=sunpy.time.julian_day(mdates.num2date(time_in[i+1]))-sunpy.time.julian_day(mdates.num2date(time_in[i]))

 
  #4 dst1 
  #find value of dst1(t-tau1) 
  #time_in is in matplotlib format in days: 
  #im time_in den index suchen wo time_in-tau1 am nächsten ist
  #und dann bei dst1 den wert mit dem index nehmen der am nächsten ist, das ist dann dst(t-tau1)
  #wenn index nicht existiert (am anfang) einfach index 0 nehmen
  #check for index where timesi is greater than t minus tau
  
  indtau1=np.where(time_in > (time_in[i]-tau1))
  dst1tau1=dst1[indtau1[0][0]]
  #similar search for others  
  dst2tau1=dst2[indtau1[0][0]]
  th1=0.725*(sinphi**-1.46)
  th2=1.83*(sinphi**-1.46)
  fe1=(-4.96e-3)*  (1+0.28*dh)*  (2*exx+abs(exx-th1)+abs(exx-th2)-th1-th2)*  (abs(vx_in[i])**1.11)*((density_in[i])**0.49)*(sinphi**6.0)
  dst1[i+1]=dst1[i]+  (a1*(-dst1[i])**a2   +fe1*   (1+(a3*dst1tau1+a4*dst2tau1)/(1-a5*dst1tau1-a6*dst2tau1)))  *dttl
  
  #5 dst2    
  indtau2=np.where(time_in > (time_in[i]-tau2))
  dst1tau2=dst1[indtau2[0][0]]
  df2=(-3.85e-8)*(abs(vx_in[i])**1.97)*(btot_in[i]**1.16)*(np.sin(theta_li)**5.7)*((density_in[i])**0.41)*(1+dh)
  fe2=(2.02*1e3)*(sinphi**3.13)*df2/(1-df2)
  dst2[i+1]=dst2[i]+(b1*(-dst2[i])**b2+fe2*(1+(b3*dst1tau2)/(1-b3*dst1tau2)))*dttl
  
  #6 dst3  
  indtau3=np.where(time_in > (time_in[i]-tau3))
  dst3tau3=dst3[indtau3[0][0]]
  df3=-4.75e-6*(abs(vx_in[i])**1.22)*(bt[i]**1.11)*np.sin(theta_li)**5.5*((density_in[i])**0.24)*(1+dh)
  fe3=3.45e3*(sinphi**0.9)*df3/(1-df3)
  dst3[i+1]=dst3[i]+  (c1*dst3[i]   + fe3*(1+(c2*dst3tau3)/(1-c2*dst3tau3)))*dttl
  
   
  #print(dst1[i], dst2[i], dst3[i], pressureterm[i], directterm[i], offset[i])
  #debugging
  #if i == 30: pdb.set_trace()
  #for debugging
  #print()
  #print(dst1[i])
  #print(dst2[i])
  #print(dst3[i])
  #print(pressureterm[i])
  #print(directterm[i])


  #add time delays: ** to do
  
  #The dst1, dst2, dst3, (pressure term), (direct IMF bz term), and (offset terms) are added (after interpolations) with time delays of 7.1, 21.0, 43.4, 2.0, 23.1 and 7.1 min, respectively, for comparison with the ‘‘Kyoto Dst.’’ 

  #dst1
  
  dst_temerin_li_out[i]=dst1[i]+dst2[i]+dst3[i]+pressureterm[i]+directterm[i]+offset[i]
  
  

  


  #print(dst_temerin_li_out[i])
 
  #---------------- loop over
 

 return (dstcalc1,dstcalc2, dst_temerin_li_out)   
   

def sunriseset(location_name):

 location = ephem.Observer()

 if location_name == 'iceland':
    location.lat = '64.128' #+(N)
    location.long = '-21.82'  #+E, so negative is west
    location.elevation = 22 #meters

 if location_name == 'edmonton':
    location.lat = '53.631611' #+(N)
    location.long = '-113.3239'  #+E, so negative is west
    location.elevation = 623 #meters
 
 if location_name == 'dunedin': 
    location.lat = '45.87416' #+(N)
    location.long = '170.50361'  #+E, so negative is west
    location.elevation = 94 #meters


 sun = ephem.Sun()
 #get sun ephemerides for location	
 sun.compute(location)
 nextrise = location.next_rising(sun).datetime()
 nextset = location.next_setting(sun).datetime()
 prevrise=location.previous_rising(sun).datetime()
 prevset=location.previous_setting(sun).datetime()

 return (nextrise,nextset,prevrise,prevset)







#for testing purposes and the chrispy package
def convert_GSE_to_GSM(bxgse,bygse,bzgse,timegse):
 #GSE to GSM conversion
 #main issue: need to get angle psigsm after Hapgood 1992/1997, section 4.3
 #for debugging pdb.set_trace()
 #for testing OMNI DATA use
 #[bxc,byc,bzc]=convert_GSE_to_GSM(bx[90000:90000+20],by[90000:90000+20],bz[90000:90000+20],times1[90000:90000+20])
 
 mjd=np.zeros(len(timegse))
 
 #output variables
 bxgsm=np.zeros(len(timegse))
 bygsm=np.zeros(len(timegse))
 bzgsm=np.zeros(len(timegse))
  
 for i in np.arange(0,len(timegse)):
			#get all dates right
			jd=sunpy.time.julian_day(sunpy.time.break_time(mdates.num2date(timegse[i])))
			mjd[i]=float(int(jd-2400000.5)) #use modified julian date    
			T00=(mjd[i]-51544.5)/36525.0
			dobj=mdates.num2date(timegse[i])
			UT=dobj.hour + dobj.minute / 60. + dobj.second / 3600. #time in UT in hours    
			#define position of geomagnetic pole in GEO coordinates
			pgeo=78.8+4.283*((mjd[i]-46066)/365.25)*0.01 #in degrees
			lgeo=289.1-1.413*((mjd[i]-46066)/365.25)*0.01 #in degrees
			#GEO vector
			Qg=[np.cos(pgeo*np.pi/180)*np.cos(lgeo*np.pi/180), np.cos(pgeo*np.pi/180)*np.sin(lgeo*np.pi/180), np.sin(pgeo*np.pi/180)]
			#now move to equation at the end of the section, which goes back to equations 2 and 4:
			#CREATE T1, T00, UT is known from above
			zeta=(100.461+36000.770*T00+15.04107*UT)*np.pi/180
			################### theta und z
			T1=np.matrix([[np.cos(zeta), np.sin(zeta),  0], [-np.sin(zeta) , np.cos(zeta) , 0], [0,  0,  1]]) #angle for transpose
			LAMBDA=280.460+36000.772*T00+0.04107*UT
			M=357.528+35999.050*T00+0.04107*UT
			lt2=(LAMBDA+(1.915-0.0048*T00)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
			#CREATE T2, LAMBDA, M, lt2 known from above
			##################### lamdbda und Z
			t2z=np.matrix([[np.cos(lt2), np.sin(lt2),  0], [-np.sin(lt2) , np.cos(lt2) , 0], [0,  0,  1]])
			et2=(23.439-0.013*T00)*np.pi/180
			###################### epsilon und x
			t2x=np.matrix([[1,0,0],[0,np.cos(et2), np.sin(et2)], [0, -np.sin(et2), np.cos(et2)]])
			T2=np.dot(t2z,t2x)  #equation 4 in Hapgood 1992
			#matrix multiplications   
			T2T1t=np.dot(T2,np.matrix.transpose(T1))
			################
			Qe=np.dot(T2T1t,Qg) #Q=T2*T1^-1*Qq
			psigsm=np.arctan(Qe.item(1)/Qe.item(2)) #arctan(ye/ze) in between -pi/2 to +pi/2
			
			T3=np.matrix([[1,0,0],[0,np.cos(-psigsm), np.sin(-psigsm)], [0, -np.sin(-psigsm), np.cos(-psigsm)]])
			GSE=np.matrix([[bxgse[i]],[bygse[i]],[bzgse[i]]]) 
			GSM=np.dot(T3,GSE)   #equation 6 in Hapgood
			bxgsm[i]=GSM.item(0)
			bygsm[i]=GSM.item(1)
			bzgsm[i]=GSM.item(2)
 #-------------- loop over

 return (bxgsm,bygsm,bzgsm)


def sphere2cart(r, phi, theta):
    #convert spherical to cartesian coordinates
    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
    return (x, y, z) 



def convert_RTN_to_GSE_sta_l1(cbr,cbt,cbn,ctime,pos_stereo_heeq):

 #function call [dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta)

 #pdb.set_trace()	for debugging
 #convert STEREO A magnetic field from RTN to GSE
 #for prediction of structures seen at STEREO-A later at Earth L1
 #so we do not include a rotation of the field to the Earth position

 #output variables
 heeq_bx=np.zeros(len(ctime))
 heeq_by=np.zeros(len(ctime))
 heeq_bz=np.zeros(len(ctime))
 
 bxgse=np.zeros(len(ctime))
 bygse=np.zeros(len(ctime))
 bzgse=np.zeros(len(ctime))
 
  
 ########## first RTN to HEEQ 
 
 #go through all data points
 for i in np.arange(0,len(ctime)):
    time_ind_pos=(np.where(pos_time_num < ctime[i])[-1][-1])
    #make RTN vectors, HEEQ vectors, and project 
    #r, long, lat in HEEQ to x y z
    [xa,ya,za]=sphere2cart(pos_stereo_heeq[0][time_ind_pos],pos_stereo_heeq[1][time_ind_pos],pos_stereo_heeq[2][time_ind_pos])
    

    #HEEQ vectors
    X_heeq=[1,0,0]
    Y_heeq=[0,1,0]
    Z_heeq=[0,0,1]

    #normalized X RTN vector
    Xrtn=[xa, ya,za]/np.linalg.norm([xa,ya,za])
    #solar rotation axis at 0, 0, 1 in HEEQ
    Yrtn=np.cross(Z_heeq,Xrtn)/np.linalg.norm(np.cross(Z_heeq,Xrtn))
    Zrtn=np.cross(Xrtn, Yrtn)/np.linalg.norm(np.cross(Xrtn, Yrtn))
    

    #project into new system
    heeq_bx[i]=np.dot(np.dot(cbr[i],Xrtn)+np.dot(cbt[i],Yrtn)+np.dot(cbn[i],Zrtn),X_heeq)
    heeq_by[i]=np.dot(np.dot(cbr[i],Xrtn)+np.dot(cbt[i],Yrtn)+np.dot(cbn[i],Zrtn),Y_heeq)
    heeq_bz[i]=np.dot(np.dot(cbr[i],Xrtn)+np.dot(cbt[i],Yrtn)+np.dot(cbn[i],Zrtn),Z_heeq)

  

 #get modified Julian Date for conversion as in Hapgood 1992
 jd=np.zeros(len(ctime))
 mjd=np.zeros(len(ctime))
 
 #then HEEQ to GSM
 #-------------- loop go through each date
 for i in np.arange(0,len(ctime)):
  sunpy_time=sunpy.time.break_time(mdates.num2date(ctime[i]))
  jd[i]=sunpy.time.julian_day(sunpy_time)
  mjd[i]=float(int(jd[i]-2400000.5)) #use modified julian date    
  #then lambda_sun
  T00=(mjd[i]-51544.5)/36525.0
  dobj=mdates.num2date(ctime[i])
  UT=dobj.hour + dobj.minute / 60. + dobj.second / 3600. #time in UT in hours   
  LAMBDA=280.460+36000.772*T00+0.04107*UT
  M=357.528+35999.050*T00+0.04107*UT
  #lt2 is lambdasun in Hapgood, equation 5, here in rad
  lt2=(LAMBDA+(1.915-0.0048*T00)*np.sin(M*np.pi/180)+0.020*np.sin(2*M*np.pi/180))*np.pi/180
  #note that some of these equations are repeated later for the GSE to GSM conversion
  S1=np.matrix([[np.cos(lt2+np.pi), np.sin(lt2+np.pi),  0], [-np.sin(lt2+np.pi) , np.cos(lt2+np.pi) , 0], [0,  0,  1]])
  #create S2 matrix with angles with reversed sign for transformation HEEQ to HAE
  omega_node=(73.6667+0.013958*((mjd[i]+3242)/365.25))*np.pi/180 #in rad
  S2_omega=np.matrix([[np.cos(-omega_node), np.sin(-omega_node),  0], [-np.sin(-omega_node) , np.cos(-omega_node) , 0], [0,  0,  1]])
  inclination_ecl=7.25*np.pi/180
  S2_incl=np.matrix([[1,0,0],[0,np.cos(-inclination_ecl), np.sin(-inclination_ecl)], [0, -np.sin(-inclination_ecl), np.cos(-inclination_ecl)]])
  #calculate theta
  theta_node=np.arctan(np.cos(inclination_ecl)*np.tan(lt2-omega_node)) 

  #quadrant of theta must be opposite lt2 - omega_node Hapgood 1992 end of section 5   
  #get lambda-omega angle in degree mod 360   
  lambda_omega_deg=np.mod(lt2-omega_node,2*np.pi)*180/np.pi
  #get theta_node in deg
  theta_node_deg=theta_node*180/np.pi
  #if in same quadrant, then theta_node = theta_node +pi   
  if abs(lambda_omega_deg-theta_node_deg) < 180: theta_node=theta_node+np.pi
  S2_theta=np.matrix([[np.cos(-theta_node), np.sin(-theta_node),  0], [-np.sin(-theta_node) , np.cos(-theta_node) , 0], [0,  0,  1]])

  #make S2 matrix
  S2=np.dot(np.dot(S2_omega,S2_incl),S2_theta)
  #this is the matrix S2^-1 x S1
  HEEQ_to_HEE_matrix=np.dot(S1, S2)
  #convert HEEQ components to HEE
  HEEQ=np.matrix([[heeq_bx[i]],[heeq_by[i]],[heeq_bz[i]]]) 
  HEE=np.dot(HEEQ_to_HEE_matrix,HEEQ)
  #change of sign HEE X / Y to GSE is needed
  bxgse[i]=-HEE.item(0)
  bygse[i]=-HEE.item(1)
  bzgse[i]=HEE.item(2)
  
 #-------------- loop over


 return (bxgse,bygse,bzgse)

























######################################## MAIN PROGRAM ####################################

#get current directory
#os.system('pwd')
#closes all plots
plt.close('all')

print('Christian Moestl, IWF Graz, last update May 2018.')
print()
print('PREDSTORM -method for geomagnetic storm and aurora forecasting. ')
print('Pattern recognition is based on Riley et al. 2017 Space Weather, and')
print('Owens, Riley and Horbury 2017 Solar Physics. ')
print()
print('Method extended for use of magnetic field and plasma data from STEREO, ')
print('or from an L5 mission or interplanetary CubeSats.')
print()
print('This is a pattern recognition technique that searches ')
print('for similar intervals in historic data as the current solar wind.')
print()
print('It is currently an unsupervised learning method.')
print()
print()
print('-------------------------------------------------')






######################### (1) get real time DSCOVR data ##################################

#see https://docs.python.org/3/howto/urllib2.html
#data from http://services.swpc.noaa.gov/products/solar-wind/

#if needed replace with ACE
#http://legacy-www.swpc.noaa.gov/ftpdir/lists/ace/



#get 3 or 7 day data
#url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json'
#url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json'


url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'



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

#interpolate NaN values in the hourly interpolated data ******* to add 











###################### (1b) get real time STEREO-beacon data


print()

#position of STEREO from sunpy?



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


print('STEREO-A HEEQ longitude to Earth is ', round(sta_long_heeq,1),' degree.') 
print('This is ', round(abs(sta_long_heeq)/60,2),' times the location of L5.') 
print('STEREO-A HEEQ latitude is ', round(sta_lat_heeq,1),' degree.') 
print('Earth L1 HEEQ latitude is ',round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1),' degree')
print('Difference HEEQ latitude is ',abs(round(sta_lat_heeq,1)-round(pos.earth_l1[2][pos_time_now_ind]*180/np.pi,1)),' degree')
print('STEREO-A heliocentric distance is ', round(sta_r,3),' AU.') 
print('The Sun rotation period with respect to Earth is ', sun_syn,' days') 
print('This is a time lag of ', round(timelag_sta_l1,2), ' days.') 
print('Arrival time of now STEREO-A wind at L1:',arrival_time_l1_sta_str[0:16])


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
  
 if not os.path.exists('/Users/chris/python/predstorm/beacon/'+sta_pla_file_str[p]):
  #download files if they are not here
  http_sta_pla_file_str[p]=plastic_location+'/'+stayear+'/'+stamonth+'/'+sta_pla_file_str[p]
  urllib.request.urlretrieve(http_sta_pla_file_str[p], '/Users/chris/python/predstorm/beacon/'+sta_pla_file_str[p])
  
 if not os.path.exists('/Users/chris/python/predstorm/beacon/'+sta_mag_file_str[p]):
  http_sta_mag_file_str[p]=impact_location+'/'+stayear+'/'+stamonth+'/'+sta_mag_file_str[p]
  urllib.request.urlretrieve(http_sta_mag_file_str[p], '/Users/chris/python/predstorm/beacon/'+sta_mag_file_str[p])
  

#read in all CDF files and stitch to one array
#access cdf
#https://pythonhosted.org/SpacePy/pycdf.html#read-a-cdf
 
 
#define stereo-a variables with open size, cut 0 later
sta_ptime=np.zeros(0)  
sta_vr=np.zeros(0)  

#sta_vt=np.zeros(2000*14)  
#sta_vn=np.zeros(2000*14)  

sta_den=np.zeros(0)  


for p in np.arange(0,14):
  sta =  pycdf.CDF('/Users/chris/python/predstorm/beacon/'+sta_pla_file_str[p])
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
[dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta)
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
 pickle.dump([spot,btot,bx,by,bz,bygsm,bzgsm,speed,speedx, dst,kp, den,pdyn,year,day,hour,times1], open( "/Users/chris/python/savefiles/omni2save_april2018.p", "wb" ) ) 
else: [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "/Users/chris/python/savefiles/omni2save_april2018.p", "rb" ) )



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

filename_save='real/savefiles/predstorm_realtime_stereo_l1_save_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.p'
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

print('PREDSTORM L1 + STEREO-A prediction results:')
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









sys.exit()
































































































################################# (2) get OMNI training data ##############################

#download from  ftp://nssdcftp.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat


if data_from_omni_file == 1:
 getdata()
 converttime()
 pickle.dump([spot,btot,bx,by,bz,bygsm,bzgsm,speed,speedx, dst,kp, den,pdyn,year,day,hour,times1], open( "/Users/chris/python/savefiles/omni2save_april2018.p", "wb" ) ) 
else: [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "/Users/chris/python/savefiles/omni2save_april2018.p", "rb" ) )


### slice data for comparison of solar wind to Dst conversion

print()
print()

print('OMNI2 1 hour training data, number of points available: ', np.size(speed))
print('start date:',str(mdates.num2date(np.min(times1))))
print('end date:',str(mdates.num2date(np.max(times1))))

trainstartnum=mdates.date2num(sunpy.time.parse_time(trainstart))-deltat/24
trainendnum=mdates.date2num(sunpy.time.parse_time(trainend))-deltat/24

print('Training data start and end interval: ', trainstart, '  ', trainend)


####### "now-wind" is 24 hour data ist rptimes24, rpv24, rbtimes24, rbtot24
#rename for plotting and analysis:
timesnp=rptimes24
speedn=rpv24
timesnb=rbtimes24
btotn=rbtot24
bzgsmn=rbzgsm24
bygsmn=rbygsm24
denn=rpn24

print()
print()
print('Number of data points in now-wind:', np.size(btotn))
print('Observing and forecasting window delta-T: ',deltat,' hours')
print('Time now: ', str(mdates.num2date(timenowb)))
print()
print('-------------------------------------------------')
print()

########################### (3)SLIDING window pattern recognition ##########################
# search for matches of the now wind with the training data

calculation_start=time.time()

#---------- sliding window analysis start

#select array from OMNI data as defined by training start and end time
startindex=np.max(np.where(trainstartnum > times1))+1
endindex=np.max(np.where(trainendnum > times1))+1

trainsize=endindex-startindex
print('Data points in training data set: ', trainsize)

#these are the arrays for the correlations between now wind and training data
corr_count_b=np.zeros(trainsize)
corr_count_bz=np.zeros(trainsize)
corr_count_by=np.zeros(trainsize)
corr_count_v=np.zeros(trainsize)
corr_count_n=np.zeros(trainsize)

#these are the arrays for the squared distances between now wind and training data
dist_count_b=np.zeros(trainsize)
dist_count_bz=np.zeros(trainsize)
dist_count_by=np.zeros(trainsize)
dist_count_v=np.zeros(trainsize)
dist_count_n=np.zeros(trainsize)

##  sliding window analysis
for i in np.arange(0,trainsize): 

  #go forward in time from start of training set in 1 hour increments
  #timeslidenum=trainstartnum+i/24
  #print(str(mdates.num2date(timeslidenum)))

  #*** this can be optimized with the startindex from above (so where is not necessary)
  #look this time up in the omni data and extract the next deltat hours
  #inds=np.where(timeslidenum==times1)[0][0]
  
  #simpler method:
  inds=startindex+i
  
  #for btotal field
  btots=btot[inds:inds+deltat+1]
  #get correlation of training data btots with now-wind btotn
  #corr_count_b[i]=np.corrcoef(btotn,btots)[0][1]
  dist_count_b[i]=np.sqrt(np.sum((btotn-btots)**2))/np.size(btotn)

  #same for bzgsm
  bzgsms=bzgsm[inds:inds+deltat+1]
  #corr_count_bz[i]=np.corrcoef(bzgsmn,bzgsms)[0][1]
  dist_count_bz[i]=np.sqrt(np.sum((bzgsmn-bzgsms)**2))/np.size(bzgsmn)

  #same for bygsm
  bygsms=bygsm[inds:inds+deltat+1]
  

  #corr_count_by[i]=np.corrcoef(bygsmn,bygsms)[0][1]
  dist_count_by[i]=np.sqrt(np.sum((bygsmn-bygsms)**2))/np.size(bygsmn)

  #same for speed
  speeds=speed[inds:inds+deltat+1]
  
  #when there is no nan:
  #if np.sum(np.isnan(speeds)) == 0:
  dist_count_v[i]=np.sqrt(np.sum((speedn-speeds)**2))/np.size(speedn)
  #corr_count_v[i]=np.corrcoef(speedn,speeds)[0][1]
  #see Riley et al. 2017 equation 1 but divided by size 
  #so this measure is the average rms error
  
  
  #same for density
  dens=den[inds:inds+deltat+1]
  #corr_count_n[i]=np.corrcoef(denn,dens)[0][1]
  dist_count_n[i]=np.sqrt(np.sum((denn-dens)**2))/np.size(denn)
  
### done


#for Btot
#maxval=np.max(corr_count_b)
#maxpos=np.argmax(corr_count_b)
#get top 50 of all correlations, they are at the end of the array
#top50_b=np.argsort(corr_count_b)[-50:-1]
#go forward in time from training data set start to the position of the best match + deltat hours 
#(so you take the future part coming after wind where the best match is seen)

#method with minimum rms distance
maxval_b=np.min(dist_count_b)
maxpos_b=np.argmin(dist_count_b)
top50_b=np.argsort(dist_count_b)[0:49]

print('find minimum of B distance at index:')
print(round(maxval_b,1), ' nT   index: ',maxpos_b)

indp_b=startindex+maxpos_b+deltat
#select array from OMNI data for predicted wind - all with p at the end
btotp=btot[indp_b:indp_b+deltat+1]



#for BzGSM
#maxval=np.max(corr_count_bz)
#maxpos=np.argmax(corr_count_bz)
#get top 50 of all correlations, they are at the end of the array
#top50_bz=np.argsort(corr_count_bz)[-50:-1]

#method with minimum rms distance
maxval_bz=np.nanmin(dist_count_bz)
maxpos_bz=np.argmin(dist_count_bz)
top50_bz=np.argsort(dist_count_bz)[0:49]

print('find minimum of BzGSM distance at index:')
print(round(maxval_bz,1), ' nT   index: ',maxpos_bz)
#go forward in time from training data set start to the position of the best match + deltat hours 
#(so you take the future part coming after wind where the best match is seen)
indp_bz=startindex+maxpos_bz+deltat
#select array from OMNI data for predicted wind - predictions all have a p at the end
bzp=bzgsm[indp_bz:indp_bz+deltat+1]

#for V
#method with correlation
#maxval_v=np.max(corr_count_v)
#maxpos_v=np.argmax(corr_count_v)
#top50_v=np.argsort(corr_count_v)[-50:-1]

#use nanmin because nan's might show up in dist_count
#method with minimum rms distance
maxval_v=np.nanmin(dist_count_v)
maxpos_v=np.argmin(dist_count_v)
top50_v=np.argsort(dist_count_v)[0:49]

print('find minimum of V distance at index:')
print(round(maxval_v), ' km/s   index: ',maxpos_v)

#select array from OMNI data for predicted wind - all with p at the end
indp_v=startindex+maxpos_v+deltat
speedp=speed[indp_v:indp_v+deltat+1]
    
    
#for N
#maxval_n=np.max(corr_count_n)
#maxpos_n=np.argmax(corr_count_n)
#top50_n=np.argsort(corr_count_n)[-50:-1]

#use nanmin because nan's might show up in dist_count_n
maxval_n=np.nanmin(dist_count_n)
maxpos_n=np.argmin(dist_count_n)
top50_n=np.argsort(dist_count_n)[0:49]

print('find minimum of N distance at index:')
print(round(maxval_n,1), ' ccm-3     index: ',maxpos_n)

#select array from OMNI data for predicted wind - all with p at the end
indp_n=startindex+maxpos_n+deltat
denp=den[indp_n:indp_n+deltat+1]
    
    

#---------- sliding window analysis end
calculation_time=round(time.time()-calculation_start,2)

print('Calculation Time in seconds: ', calculation_time)



















######################--------------------------------------- plot FORECAST results

sns.set_context("talk")     
sns.set_style("darkgrid")  
#fig=plt.figure(3,figsize=(15,13))

#for testing
fig=plt.figure(3,figsize=(13,11))

weite=1
fsize=11


#------------------- Panel 1 Btotal 

ax1 = fig.add_subplot(411)


#for previous plot best 50 correlations 
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries
 indp_b50=startindex+top50_b[j]
 btot50=btot[indp_b50:indp_b50+deltat+1]
 #plot for previous times
 plt.plot_date(timesnb,btot50, 'lightgrey', linewidth=weite, alpha=0.9)

#plot the now wind
plt.plot_date(timesnb,btotn, 'k', linewidth=weite, label='observation')

#for legend
plt.plot_date(0,0, 'lightgrey', linewidth=weite, alpha=0.8)#,label='50 best B matches')
plt.plot_date(0,0, 'g', linewidth=weite, alpha=0.8)#,label='B predictions from 50 matches')

#for future plot best 50 correlations
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries, 
 #add a deltat for selecting the deltat after the data
 indp_b50=startindex+top50_b[j]+deltat
 btot50=btot[indp_b50:indp_b50+deltat+1]
 #plot for future time
 plt.plot_date(timesfb,btot50, 'g', linewidth=weite, alpha=0.4)
 
#predicted wind best match
plt.plot_date(timesfb,btotp, 'b', linewidth=weite+1, label='prediction')
 
plt.ylabel('Magnetic field B [nT]', fontsize=fsize+2)
plt.xlim((timesnb[0], timesfb[-1]))

#indicate average level of training data btot
btraining_mean=np.nanmean(btot[startindex:endindex]) 
plt.plot_date([timesnp[0], timesfp[-1]], [btraining_mean,btraining_mean],'--k', alpha=0.5, linewidth=1) 
plt.annotate('average',xy=(timesnp[0],btraining_mean),xytext=(timesnp[0],btraining_mean),color='k', fontsize=10)

#add *** make ticks in 6h distances starting with 0, 6, 12 UT


myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax1.xaxis.set_major_formatter(myformat)
plt.plot_date([timesnb[-1],timesnb[-1]],[0,100],'-r', linewidth=3)
plt.ylim(0,max(btotp)+12)
#ax1.legend(loc=2, fontsize=fsize-2, ncol=2)

plt.annotate('now',xy=(timenowb,max(btotp)+12-3),xytext=(timenowb+0.01,max(btotp)+12-3),color='r', fontsize=15)
plt.annotate('observation',xy=(timenowb,max(btotp)+12-3),xytext=(timenowb-0.55,max(btotp)+12-3),color='k', fontsize=15)
plt.annotate('prediction',xy=(timenowb,max(btotp)+12-3),xytext=(timenowb+0.45,max(btotp)+12-3),color='b', fontsize=15)

plt.yticks(fontsize=fsize) 
plt.xticks(fontsize=fsize) 

plt.title('PREDSTORM L1 solar wind and magnetic storm prediction with unsupervised pattern recognition for '+ str(mdates.num2date(timenowb))[0:16]+ ' UT', fontsize=15)





#------------------------ Panel 2 BZ
ax2 = fig.add_subplot(412)

#plot best 50 correlations for now wind
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries
 indp_bz50=startindex+top50_bz[j]
 bz50=bzgsm[indp_bz50:indp_bz50+deltat+1]
 #plot for previous times
 plt.plot_date(timesnb,bz50, 'lightgrey', linewidth=weite, alpha=0.9)

#this is the observed now wind
plt.plot_date(timesnb,bzgsmn, 'k', linewidth=weite, label='Bz observed by DSCOVR')

#for legend
plt.plot_date(0,0, 'lightgrey', linewidth=weite, alpha=0.8,label='50 best Bz matches')
plt.plot_date(0,0, 'g', linewidth=weite, alpha=0.8,label='Bz predictions from 50 matches')


#for future wind plot best 50 correlations
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries, add a deltat for selecting the deltat after the data
 indp_bz50=startindex+top50_bz[j]+deltat
 bz50=bzgsm[indp_bz50:indp_bz50+deltat+1]
 #plot for future time
 plt.plot_date(timesfb,bz50, 'g', linewidth=weite, alpha=0.4)
 

#predicted wind
plt.plot_date(timesfb,bzp, 'b', linewidth=weite+1, label='Bz best match prediction')

#0 level
plt.plot_date([timesnp[0], timesfp[-1]], [0,0],'--k', alpha=0.5, linewidth=1) 
 

plt.ylabel('Bsouth [nT] GSM')
plt.xlim((timesnb[0], timesfb[-1]))
myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax2.xaxis.set_major_formatter(myformat)
plt.plot_date([timesnb[-1],timesnb[-1]],[min(bzgsmn)-15,max(bzgsmn)+15],'-r', linewidth=3)
plt.ylim(min(bzgsmn)-15,max(bzgsmn)+15)
#ax2.legend(loc=2, fontsize=fsize-2)

plt.yticks(fontsize=fsize) 
plt.xticks(fontsize=fsize) 




#------------------------- Panel 3 SPEED 

ax3 = fig.add_subplot(413)


#plot best 50 correlations
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries
 indp_v50=startindex+top50_v[j]
 speedp50=speed[indp_v50:indp_v50+deltat+1]
 #plot for previous time
 plt.plot_date(timesnp,speedp50, 'lightgrey', linewidth=weite, alpha=0.9)


plt.plot_date(timesnp,speedn, 'k', linewidth=weite, label='V observed by DSCOVR')

#plot best 50 correlations
for j in np.arange(49):
 #search for index in OMNI data for each of the top50 entries, add a deltat for selecting the deltat after the data
 indp_v50=startindex+top50_v[j]+deltat
 speedp50=speed[indp_v50:indp_v50+deltat+1]
 #plot for future time
 plt.plot_date(timesfp,speedp50, 'g', linewidth=weite, alpha=0.4)

plt.plot_date(0,0, 'lightgrey', linewidth=weite, alpha=0.8,label='50 best V matches')
plt.plot_date(0,0, 'g', linewidth=weite, alpha=0.8,label='V predictions from 50 matches')

#predicted wind
plt.plot_date(timesfp,speedp, 'b', linewidth=weite+1, label='V best match prediction')


plt.ylabel('Speed [km/s]')
plt.xlim((timesnp[0], timesfp[-1]))
myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax3.xaxis.set_major_formatter(myformat)
#time now
plt.plot_date([timesnp[-1],timesnp[-1]],[0,2500],'-r', linewidth=3)
plt.ylim(250,np.nanmax(speedp)+400)
#ax3.legend(loc=2, fontsize=fsize-2)

plt.yticks(fontsize=fsize) 
plt.xticks(fontsize=fsize) 


#add speed levels
plt.plot_date([timesnp[0], timesfp[-1]], [400,400],'--k', alpha=0.3, linewidth=1)
plt.annotate('slow',xy=(timesnp[0],400),xytext=(timesnp[0],400),color='k', fontsize=10)
plt.plot_date([timesnp[0], timesfp[-1]], [800,800],'--k', alpha=0.3, linewidth=1)
plt.annotate('fast',xy=(timesnp[0],800),xytext=(timesnp[0],800),color='k', fontsize=10	)




#--------------------------------- PANEL 4 Dst 

#make Dst index from solar wind observed+prediction in single array
#[dst_burton]=make_predstorm_dst(btoti, bygsmi, bzgsmi, speedi, deni, timesi)

#btotal timesnb btotn  timesfb btotp
#bzgsm timesnb bzgsmn timesfb bzp 
#speed: timesnp, speedn; dann  timesfp, speedp
#density timesnp denn timesfp denp
#times timesnp timesfp

#make one array of observed and predicted wind for Dst prediction:

timesdst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
btotdst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
bzdst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
speeddst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
dendst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)

#write times in one array, note the overlap at the now time
timesdst[:25]=timesnb
timesdst[25:49]=timesfb[1:]

btotdst[:25]=btotn
btotdst[25:49]=btotp[1:]

bzdst[:25]=bzgsmn
bzdst[25:49]=bzp[1:]

speeddst[:25]=speedn
speeddst[25:49]=speedp[1:]

dendst[:25]=denn
dendst[25:49]=denp[1:]


#[dst_burton]=make_predstorm_dst(btoti, bygsmi, bzgsmi, speedi, deni, timesi)
[pdst_burton, pdst_obrien]=make_predstorm_dst(btotdst,bzdst, speeddst, dendst, timesdst)


ax8 = fig.add_subplot(414)


#******************** added timeshift of 1 hour for L1 to Earth! This should be different for each timestep to be exact
#predicted dst
plt.plot_date(timesdst+1/24, pdst_burton,'b-', label='Dst Burton et al. 1975',markersize=5, linewidth=1)
plt.plot_date(timesdst+1/24, pdst_obrien,'r-', label='Dst OBrien & McPherron 2000',markersize=5, linewidth=1)
#real Dst
#for AER
#plt.plot_date(rtimes7, rdst7,'ko', label='Dst observed',markersize=4)
#for Kyoto
plt.plot_date(rdst_time, rdst,'ko', label='Dst observed',markersize=4)

plt.ylabel('Dst [nT]')
ax8.legend(loc=3)
plt.ylim([min(pdst_burton)-120,60])
#time limit similar to previous plots
plt.xlim((timesnp[0], timesfp[-1]))
myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax8.xaxis.set_major_formatter(myformat)
#time now
plt.plot_date([timesnp[-1],timesnp[-1]],[-1500, +500],'-r', linewidth=3)
ax8.legend(loc=3, fontsize=fsize-2,ncol=3)

plt.yticks(fontsize=fsize) 
plt.xticks(fontsize=fsize) 


#add geomagnetic storm levels
plt.plot_date([timesnp[0], timesfp[-1]], [-50,-50],'--k', alpha=0.3, linewidth=1)
plt.annotate('moderate storm',xy=(timesnp[0],-50+2),xytext=(timesnp[0],-50+2),color='k', fontsize=12)
plt.plot_date([timesnp[0], timesfp[-1]], [-100,-100],'--k', alpha=0.3, linewidth=1)
plt.annotate('intense storm',xy=(timesnp[0],-100+2),xytext=(timesnp[0],-100+2),color='k', fontsize=12)
plt.plot_date([timesnp[0], timesfp[-1]], [-250,-250],'--k', alpha=0.3, linewidth=1)
plt.annotate('super-storm',xy=(timesnp[0],-250+2),xytext=(timesnp[0],-250+2),color='k', fontsize=12)
#plt.plot_date([timesnp[0], timesfp[-1]], [-1000,-1000],'--k', alpha=0.8, linewidth=1)
#plt.annotate('Carrington event',xy=(timesnp[0],-1000+2),xytext=(timesnp[0],-1000+2),color='k', fontsize=12)


plt.annotate('Horizontal lines are sunset to sunrise intervals ',xy=(timesnp[0],45),xytext=(timesnp[0],45),color='k', fontsize=10)

#https://chrisramsay.co.uk/posts/2017/03/fun-with-the-sun-and-pyephem/
#get sunrise/sunset times for Reykjavik Iceland and Edmonton Kanada, and Dunedin New Zealand with ephem package

#use function defined above
[icenextrise,icenextset,iceprevrise,iceprevset]=sunriseset('iceland')
[ednextrise,ednextset,edprevrise,edprevset]=sunriseset('edmonton')
[dunnextrise,dunnextset,dunprevrise,dunprevset]=sunriseset('dunedin')

nightlevels_iceland=5
nightlevels_edmonton=20
nightlevels_dunedin=35


#ICELAND
#show night duration on plots - if day at current time, show 2 nights
if iceprevset < iceprevrise:
 #previous night
 plt.plot_date([mdates.date2num(iceprevset), mdates.date2num(iceprevrise)], [nightlevels_iceland,nightlevels_iceland],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Iceland',xy=(mdates.date2num(iceprevset),nightlevels_iceland+2),xytext=(mdates.date2num(iceprevset),nightlevels_iceland+2),color='k', fontsize=12)
 #next night
 plt.plot_date([mdates.date2num(icenextset), mdates.date2num(icenextrise)], [nightlevels_iceland,nightlevels_iceland],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Iceland',xy=(mdates.date2num(icenextset),nightlevels_iceland+2),xytext=(mdates.date2num(icenextset),nightlevels_iceland+2),color='k', fontsize=12)
 
 #indicate boxes for aurora visibility
 #matplotlib.patches.Rectangle(xy, width, height)
 #ax8.add_patch( matplotlib.patches.Rectangle([mdates.date2num(icenextset),-500], mdates.date2num(icenextrise)-mdates.date2num(icenextset), 475, linestyle='--', facecolor='g',edgecolor='k', alpha=0.3)) 
 
 
#if night now make a line from prevset to nextrise ****(not sure if this is correct to make the night touch the edge of the plot!
if iceprevset > iceprevrise:
 #night now
 plt.plot_date([mdates.date2num(iceprevset), mdates.date2num(icenextrise)], [nightlevels_iceland,nightlevels_iceland],'-k', alpha=0.8, linewidth=1)
 #previous night from left limit to prevrise
 plt.plot_date([timesnp[0], mdates.date2num(iceprevrise)], [nightlevels_iceland,nightlevels_iceland],'-k', alpha=0.8, linewidth=1)
 #next night from nextset to plot limit
 plt.plot_date([mdates.date2num(icenextset), timesfp[-1]], [nightlevels_iceland,nightlevels_iceland],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Iceland',xy=(mdates.date2num(iceprevset),nightlevels_iceland+2),xytext=(mdates.date2num(iceprevset),nightlevels_iceland+2),color='k', fontsize=12)
 



#NEW ZEALAND
if dunprevset < dunprevrise:
 plt.plot_date([mdates.date2num(dunprevset), mdates.date2num(dunprevrise)], [nightlevels_dunedin,nightlevels_dunedin],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Dunedin, New Zealand',xy=(mdates.date2num(dunprevset),nightlevels_dunedin+2),xytext=(mdates.date2num(dunprevset),nightlevels_dunedin+2),color='k', fontsize=12)
 plt.plot_date([mdates.date2num(dunnextset), mdates.date2num(dunnextrise)], [nightlevels_dunedin,nightlevels_dunedin],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Dunedin, New Zealand',xy=(mdates.date2num(dunnextset),nightlevels_dunedin+2),xytext=(mdates.date2num(dunnextset),nightlevels_dunedin+2),color='k', fontsize=12)
if dunprevset > dunprevrise:
 #night now
 plt.plot_date([mdates.date2num(dunprevset), mdates.date2num(dunnextrise)], [nightlevels_dunedin,nightlevels_dunedin],'-k', alpha=0.8, linewidth=1)
 #ax8.add_patch( matplotlib.patches.Rectangle([mdates.date2num(dunprevset),-500], mdates.date2num(dunnextrise)-mdates.date2num(dunprevset), 475, linestyle='--', facecolor='g',edgecolor='k', alpha=0.3)) 
 #previous night from left limit to prevrise
 plt.plot_date([timesnp[0], mdates.date2num(dunprevrise)], [nightlevels_dunedin,nightlevels_dunedin],'-k', alpha=0.8, linewidth=1)
 #next night from nextset to plot limit
 plt.plot_date([mdates.date2num(dunnextset), timesfp[-1]], [nightlevels_dunedin,nightlevels_dunedin],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Dunedin, New Zealand',xy=(mdates.date2num(dunprevset),nightlevels_dunedin+2),xytext=(mdates.date2num(dunprevset),nightlevels_dunedin+2),color='k', fontsize=12)
 

#CANADA 
if edprevset < edprevrise:
 plt.plot_date([mdates.date2num(edprevset), mdates.date2num(edprevrise)], [nightlevels_edmonton,nightlevels_edmonton],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Edmonton, Canada',xy=(mdates.date2num(edprevset),nightlevels_edmonton+2),xytext=(mdates.date2num(edprevset),nightlevels_edmonton+2),color='k', fontsize=12)
 plt.plot_date([mdates.date2num(ednextset), mdates.date2num(ednextrise)], [nightlevels_edmonton,nightlevels_edmonton],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Edmonton, Canada',xy=(mdates.date2num(ednextset),nightlevels_edmonton+2),xytext=(mdates.date2num(ednextset),nightlevels_edmonton+2),color='k', fontsize=12)

if edprevset > edprevrise:
 #night now
 plt.plot_date([mdates.date2num(edprevset), mdates.date2num(ednextrise)], [nightlevels_edmonton,nightlevels_edmonton],'-k', alpha=0.8, linewidth=1)
 plt.plot_date([timesnp[0], mdates.date2num(edprevrise)], [nightlevels_edmonton,nightlevels_edmonton],'-k', alpha=0.8, linewidth=1)
 plt.plot_date([mdates.date2num(ednextset), timesfp[-1]], [nightlevels_edmonton,nightlevels_edmonton],'-k', alpha=0.8, linewidth=1)
 plt.annotate('Edmonton, Canada',xy=(mdates.date2num(edprevset),nightlevels_edmonton+2),xytext=(mdates.date2num(edprevset),nightlevels_edmonton+2),color='k', fontsize=12)
 

#********** add level for aurora as rectangle plots



#outputs


print()
print()
print('-------------------------------------------------')
print()

print()
print('Predicted maximum of B total in next 24 hours:')
print(np.nanmax(btotp),' nT')
print('Predicted minimum of Bz GSM in next 24 hours:')
print(np.nanmin(bzp),' nT')
print('Predicted maximum V in next 24 hours:')
print(int(round(np.nanmax(speedp,0))),' km/s')
print('Predicted minimum of Dst in next 24 hours Burton/OBrien:')
print(int(round(np.nanmin(pdst_burton))), ' / ', int(round(np.nanmin(pdst_obrien))),'  nT')



plt.tight_layout()


plt.figtext(0.45,0.005, 'C. Moestl, IWF Graz. For method see Riley et al. 2017 AGU Space Weather, Owens et al. 2018 Solar Physics.', fontsize=9)


filename='real/predstorm_realtime_forecast_1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.jpg'
plt.savefig(filename)
#filename='real/predstorm_realtime_forecast_1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.eps'
#plt.savefig(filename)


plt.pause(0.001)
#for interactive mode
plt.show(block=True)

