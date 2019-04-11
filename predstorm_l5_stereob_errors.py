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
#sun_syn=26.24 #days
#use other values for equatorial coronal holes?
sun_syn=25.0#days
#sun_syn=27.5#days

#sun_syn=20#days



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




def IDL_time_to_num(time_in):  
 #convert IDL time to matplotlib datetime
 time_num=np.zeros(np.size(time_in))
 for ii in np.arange(0,np.size(time_in)):
   time_num[ii]=mdates.date2num(sunpy.time.parse_time(time_in[ii]))
   
 return time_num 




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

  
  
#def make_predstorm_dst(btot_in,bz_in,v_in,density_in,time_in):
def make_predstorm_dst(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):
 #with TL2000 : def make_predstorm_dst(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):

 #this makes from synthetic or observed solar wind the Dst index	
 #btot_in IMF total field, in nT
 #by_in - the IMF By field in nT
 #bz_in - the IMF Bz field in nT
 #v_in - the speed in km/s
 #vx_in - the solar wind speed x component (GSE or GSM?) in km/s
 #time_in - the time in matplotlib date format

 #define variables
 Ey=np.zeros(len(bz_in))
 
 #dynamic pressure
 pdyn1=np.zeros(len(bz_in)) #in nano Pascals
 protonmass=1.6726219*1e-27  #kg
 #assume pdyn is only due to protons
 pdyn1=density_in*1e6*protonmass*(v_in*1e3)**2*1e9  #in nanoPascal
 dststar1=np.zeros(len(bz_in))
 dstcalc1=np.zeros(len(bz_in))
 dststar2=np.zeros(len(bz_in))
 dstcalc2=np.zeros(len(bz_in))
 
 #set all fields above 0 to 0 
 bz_in_negind=np.where(bz_in > 0)  
 
 #important: make a deepcopy because you manipulate the input variable
 bzneg=copy.deepcopy(bz_in)
 bzneg[bz_in_negind]=0

 #define interplanetary electric field 
 Ey=v_in*abs(bzneg)*1e-3; #now Ey is in mV/m
 
 #### model 1: Burton et al. 1975 
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

 ######## model 2: OBrien and McPherron 2000 
 #constants
 Ec=0.49
 b=7.26  
 c=11  #nT
 for i in range(len(bz_in)-1):
  if Ey[i] > Ec:            #Ey in mV m
   Q=-4.4*(Ey[i]-Ec) 
  else: Q=0
  tau=2.4*np.exp(9.74/(4.69+Ey[i])) #tau in hours
  #OBrien abstract: Dst=Dst*+7.26P^1/2 -11
  #this is the ring current Dst
  #dststar2[i+1]=(Q-dststar2[i]/tau)+dststar2[i] #t is pro stunde, time intervall ist auch 1h
  deltat_hours=(time_in[i+1]-time_in[i])*24 #time_in is in days - convert to hours
  #deltat=dststar
  dststar2[i+1]=((Q-dststar2[i]/tau))*deltat_hours+dststar2[i] #t is pro stunde, time intervall ist auch 1h
  #this is the Dst of ring current and magnetopause currents 
  dstcalc2[i+1]=dststar2[i+1]+b*np.sqrt(pdyn1[i+1])-c; 
  
 
 
 ######## model 3: Xinlin Li LASP Colorado

#  
  #start with Temerin and Li 2002 (implemented here) and proceed to Temerin and Li (2006) later
#  
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
#  
#  
#  #define inital values (needed for convergence, see Temerin and Li 2002 note)
  dst1[0:10]=-15
  dst2[0:10]=-13
  dst3[0:10]=-2
#  
#  
# 
#  #define all constants
  p1=0.9
  p2=2.18e-4
  p3=14.7
#  
  s1=-2.788
  s2=1.44
  s3=-0.92
  s4=-1.054e-2
  s5=8.6e-6
#  
  a1=6.51e-2
  a2=1.37
  a3=8.4e-3  
  a4=6.053e-3
  a5=1.12e-3
  a6=1.55e-3
#  
  tau1=0.14 #days
  tau2=0.18 #days
  tau3=9e-2 #days
#  
  b1=0.792
  b2=1.326
  b3=1.29e-2
  
  c1=-24.3
  c2=5.2e2
 
  #Note: vx has to be used with a positive sign throughout the calculation
  
  #problem with t - days since 1995 dont work after 1999
  
  
  
  #----------------------------------------- loop over each timestep
  for i in range(len(bz_in)-1):
 
   #for 3DCORE use constant density in cm-3
   #if density is nan then set to 5
   
   if np.isnan(density_in[i]) > 0: density_in[i]=5
 
   #define terms
   
   #1 pressure term ****************** scheint OK etwa +30 nT
   #t time in days since beginning of 1995   #1 Jan 1995 in Julian days
   t1=sunpy.time.julian_day(mdates.num2date(time_in[i]))-sunpy.time.julian_day('1995-1-1 00:00')

    #Temerin and Li 2002
   yearli=365.24 
   tt=2*np.pi*t1/yearli
   ttt=2*np.pi*t1
   alpha=0.078
   beta=1.22
   cosphi=np.sin(tt+alpha)*np.sin(ttt-tt-beta)*(9.58589e-2)+np.cos(tt+alpha)*(0.39+0.104528*np.cos(ttt-tt-beta))
# 
#   
   #not that in Temerin and Li's code, these times are implemented very differently! - starting point for generalization
   #to arbitrary dates
#   
   #equation 1 use phi from equation 2
   sinphi=(1-cosphi**2)**0.5
#   
#   
   pressureterm[i]=(p1*(btot_in[i]**2)+density_in[i]*((p2*((v_in[i])**2)/(sinphi**2.52))+p3))**0.5
#   #2 directbzterm ************scheint OK etwa 0.5 (siehe paper, das passt)
   directterm[i]=0.478*bz_in[i]*(sinphi**11.0)
# 
#   #3 offset  ** stimmt sicher nicht fuer 2012, t1 macht keinen Sinn, wert 300 viel zu hoch, weglassen derweil (in TL 2006 auch nur bis 2002)
   offset[i]=s1+s2*np.sin(2*np.pi*t1/yearli+s3)+s4*t1+s5*(t1**2)
# 
   bt[i]=(by_in[i]**2+bz_in[i]**2)**0.5  
# 
#   #mistake in paper - bt is similarly defined as bp (with by bz); but in Temerin and Li's code (dst.pro) bp depends on by and bx
   bp[i]=(by_in[i]**2+bx_in[i]**2)**0.5  
# 
#   
#   #contains t1, but in cos and sin so it does not matter
   dh=bp[i]*np.cos(np.arctan2(bx_in[i],by_in[i])+6.10)  *   ((3.59e-2)*np.cos(2*np.pi*t1/yearli+0.04)-2.18e-2*np.sin(2*np.pi*t1-1.60))
#   
#   
   theta_li=-(np.arccos(-bz_in[i]/bt[i])-np.pi)/2
#   
   exx=1e-3*abs(vx_in[i])*bt[i]*np.sin(theta_li)**6.1
#  
#   #t1 and dt are in days
   dt=sunpy.time.julian_day(mdates.num2date(time_in[i+1]))-sunpy.time.julian_day(mdates.num2date(time_in[i]))
# 
#  
#   #4 dst1  *********** scheint zu passen
# 
#   #find value of dst1(t-tau1) 
#   #timesi ist im matplotlib format ist in Tagen: im times den index suchen wo timesi-tau1 am nächsten ist
#   #und dann bei dst1 den wert mit dem index nehmen der am nächsten ist, das ist dann dst(t-tau1)
#   #wenn index nicht existiert (am anfang) einfach index 0 nehmen
#   #check for index where timesi is greater than t minus tau
   indtau1=np.where(time_in > (time_in[i]-tau1))
   dst1tau1=dst1[indtau1[0][0]]
#   #similar search for others  
   dst2tau1=dst2[indtau1[0][0]]
#  
   th1=0.725*(sinphi**-1.46)
   th2=1.83*(sinphi**-1.46)
   fe1=(-4.96e-3)*  (1+0.28*dh)*  (2*exx+abs(exx-th1)+abs(exx-th2)-th1-th2)*  (abs(vx_in[i])**1.11)*((density_in[i])**0.49)*(sinphi**6.0)
# 
   dst1[i+1]=dst1[i]+  (a1*(-dst1[i])**a2   +fe1*   (1+(a3*dst1tau1+a4*dst2tau1)/(1-a5*dst1tau1-a6*dst2tau1)))  *dt
#   
#   
#   #5 dst2    
   indtau2=np.where(time_in > (time_in[i]-tau2))
   dst1tau2=dst1[indtau2[0][0]]
   df2=(-3.85e-8)*(abs(vx_in[i])**1.97)*(btot_in[i]**1.16)*(np.sin(theta_li)**5.7)*((density_in[i])**0.41)*(1+dh)
   fe2=(2.02*1e3)*(sinphi**3.13)*df2/(1-df2)
#    
   dst2[i+1]=dst2[i]+(b1*(-dst2[i])**b2+fe2*(1+(b3*dst1tau2)/(1-b3*dst1tau2)))*dt
#   
#   
#   #6 dst3  
#   
   indtau3=np.where(time_in > (time_in[i]-tau3))
   dst3tau3=dst3[indtau3[0][0]]
   df3=-4.75e-6*(abs(vx_in[i])**1.22)*(bt[i]**1.11)*np.sin(theta_li)**5.5*((density_in[i])**0.24)*(1+dh)
   fe3=3.45e3*sinphi**0.9*df3/(1-df3)
#   
   dst3[i+1]=dst3[i]+  (c1*dst3[i]   + fe3*(1+(c2*dst3tau3)/(1-c2*dst3tau3)))*dt
#   
#    
   #print(dst1[i], dst2[i], dst3[i], pressureterm[i], directterm[i], offset[i])
#   
#   #debugging
#   #if i == 30: pdb.set_trace()
# 
#   #final value for comparison to papers 1995-2002  
#   #this is used for comparison to Temerin and Li 2002 paper
#   #dst_temerin_li_out[i]=dst1[i]+dst2[i]+dst3[i]+pressureterm[i]+directterm[i]	+offset[i]
#   
#   
#   
#   #final value for more recent data:
#   #quick fix for wrong offset: this is set constant for years later than 2002 for now)
   offset[i]=6
#   
#   #for 2013 don't use dst3 - makes spikes
#   #dst_temerin_li_out[i]=dst1[i]+dst2[i]+dst3[i]+pressureterm[i]+directterm[i]	+offset[i]
   dst_temerin_li_out[i]=dst1[i]+dst2[i]+pressureterm[i]+directterm[i]	+offset[i]
# 
#  
#   #---------------- loop over
#   
  

 return (dstcalc1,dstcalc2,dst_temerin_li)
   
   

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



def convert_HEEQ_to_GSE_stb_l1(cbr,cbt,cbn,ctime,pos_stereo_heeq):




 ########## ******* for STB only convert from HEEQ to GSM
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
 
  
 heeq_bx=cbr
 heeq_by=cbt
 heeq_bz=cbn

  

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

print('Christian Moestl, IWF Graz, last update June 2018.')
print()
print('PREDSTORM -method for geomagnetic storm and aurora forecasting. ')
print('Error estimation using STEREO-B data')



########### get STEREO-B data



print( 'read STEREO-B data.')
#get insitu data
stb= pickle.load( open( "../catpy/DATACAT/STB_2007to2014_HEEQ.p", "rb" ) )
#stb_time=IDL_time_to_num(stb.time)
#pickle.dump([stb_time], open( "../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "wb" ) )
[stb_time]=pickle.load( open( "../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "rb" ) )
print( 'read data done.')


print('load spacecraft and planetary positions')
pos=getpositions('cats/positions_2007_2023_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)

##################################
print('load OMNI data for Dst')
data_from_omni_file = 0 #

if data_from_omni_file == 1:
 getdata()
 converttime()
 pickle.dump([spot,btot,bx,by,bz,bygsm,bzgsm,speed,speedx, dst,kp, den,pdyn,year,day,hour,times1], open( "OMNI2/omni2save.p", "wb" ) ) 
else: [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "OMNI2/omni2save.p", "rb" ) )


############load Lan Jian's CIR list for STEREO
sir=np.genfromtxt('../catpy/ALLCATS/CIRCAT master/STEREO_Level3_SIR_clean_clean.txt',
                  dtype='S4,S1,i4,S5,S5,i4,S5,S5')
sir_start=np.zeros(len(sir))                  
sir_end=np.zeros(len(sir))                  
sir_sc=np.empty(len(sir), dtype='str')     

          

                 
#make datetime object for start and end times                  
for i in np.arange(len(sir)):                  
    
    #STEREO A or B
    sir_sc[i]=str(sir[i][1])[2]
    
    #extract hours with minutes as fraction of day 
    hours=int(sir[i][4][0:2])/24+int(sir[i][4][3:5])/(24*60)
    #extract doy
    doy=int(sir[i][2])            
    sir_start[i]=mdates.date2num(datetime.datetime(int(sir[i][0]),1,1,)+datetime.timedelta(int(sir[i][2]))+datetime.timedelta(hours) )

    hours=int(sir[i][7][0:2])/24+int(sir[i][7][3:5])/(24*60)
    #extract doy
    doy=int(sir[i][5])            
    sir_end[i]=mdates.date2num(datetime.datetime(int(sir[i][0]),1,1,)+datetime.timedelta(int(sir[i][5]))+datetime.timedelta(hours) )

#CIRs only for STB
stbsirs=np.where(sir_sc=='B')
sir_start_b=sir_start[stbsirs]           
sir_end_b=sir_end[stbsirs] 


############# loadpositions
             

#position of STEREO-B
stb_r=pos.stb[0]
#get longitude and latitude
stb_long_heeq=pos.stb[1]*180/np.pi
stb_lat_heeq=pos.stb[2]*180/np.pi

#position of STEREO-A
sta_r=pos.sta[0]
#get longitude and latitude
sta_long_heeq=pos.sta[1]*180/np.pi
sta_lat_heeq=pos.sta[2]*180/np.pi


#position of Earth L1
l1_r=pos.earth_l1[0]
#get longitude and latitude
l1_long_heeq=pos.earth_l1[1]*180/np.pi
l1_lat_heeq=pos.earth_l1[2]*180/np.pi


plt.figure(1)
#plt.plot_date(pos_time_num,l1_lat_heeq-stb_lat_heeq),'-b')
#plt.plot_date(pos_time_num,abs(l1_long_heeq-stb_long_heeq),'-b')

plt.plot_date(pos_time_num,l1_lat_heeq-sta_lat_heeq,'-r')
#plt.plot_date(pos_time_num,abs(l1_long_heeq-sta_long_heeq),'-r')


#plt.figure(2)
#plt.plot_date(times1,dst,'k')

#sys.exit()






#convert STB to 1 hour time resolution and add time shift

#extract intervals from STB data, from -10 to -120 degree longitude


min_time=(pos_time_num[np.where(stb_long_heeq < -10)[0][0]])
max_time=(pos_time_num[np.where(stb_long_heeq < -110)[0][0]])

pos_ind=np.where(np.logical_and(pos_time_num > min_time, pos_time_num< max_time))
dat_ind=np.where(np.logical_and(stb_time > min_time, stb_time< max_time))







     


#go through all SIRs and check if the dat_ind is inside a SIR

#for q in np.arange(np.size(dat_ind)):
#  #if current time is inside any SIR, set to 1
#  flag=0
#  for i in np.arange(np.size(sir_start)):
#     if np.logical_and(stb_time[dat_ind][q] > sir_start[i] ,stb_time[dat_ind][q] < sir_end[i]): flag=1
  #if data point is not inside SIR, delete index   
#  if flag ==0: np.delete(dat_ind,q)
  
  
#need to round the start time so it starts with full hours
start_time=round(stb_time[dat_ind][0],1)
end_time=round(stb_time[dat_ind][-1],1)

stb_time_ext=stb_time[dat_ind]
stb_btot_ext=stb.btot[dat_ind]
stb_br_ext=stb.bx[dat_ind]
stb_bt_ext=stb.by[dat_ind]
stb_bn_ext=stb.bz[dat_ind]
stb_den_ext=stb.density[dat_ind]
stb_vtot_ext=stb.vtot[dat_ind]



#################

#first add corrections time lag and field and density to all times depending on position

####################################### APPLY CORRECTIONS TO STEREO-A data 
#(1) make correction for heliocentric distance of L1 position - take position of Earth and STEREO from pos file 
#(2) add timeshift for each datapoint

### first interpolate to 1 hour ******** passt nicht durch sprünge in der Zeit ...

stb_time_1h=np.arange(start_time+1/24.000,end_time,1.000/24.000)
stb_btot_int=np.interp(stb_time_1h,stb_time_ext,stb_btot_ext)
stb_br_int=np.interp(stb_time_1h,stb_time_ext,stb_br_ext)
stb_bt_int=np.interp(stb_time_1h,stb_time_ext,stb_bt_ext)
stb_bn_int=np.interp(stb_time_1h,stb_time_ext,stb_bn_ext)
stb_vtot_int=np.interp(stb_time_1h,stb_time_ext,stb_vtot_ext)
stb_den_int=np.interp(stb_time_1h,stb_time_ext,stb_den_ext)
stb_long_int=np.interp(stb_time_1h,pos_time_num,stb_long_heeq)
stb_lat_int=np.interp(stb_time_1h,pos_time_num,stb_lat_heeq)
l1_r_int=np.interp(stb_time_1h,pos_time_num,l1_r)
stb_r_int=np.interp(stb_time_1h,pos_time_num,stb_r)


dst_omni_int=np.interp(stb_time_1h,times1,dst)

stb_time_to_l1=np.zeros(np.size(stb_time_1h))

#go through all STEREO-B 1 hour data points
for q in np.arange(np.size(stb_time_1h)):
 timelag_stb_l1=abs(stb_long_int[q])/(360/sun_syn) #days
 stb_time_to_l1[q]=stb_time_1h[q]+timelag_stb_l1
 
 
 #stb_btot_int[q]=stb_btot_int[q]*(l1_r_int[q]/stb_r_int[q])**-2
 #stb_br_int[q]=stb_br_int[q]*(l1_r_int[q]/stb_r_int[q])**-2
 #stb_bt_int[q]=stb_bt_int[q]*(l1_r_int[q]/stb_r_int[q])**-2
 #stb_bn_int[q]=stb_bn_int[q]*(l1_r_int[q]/stb_r_int[q])**-2
 #stb_den_int[q]=stb_den_int[q]*(l1_r_int[q]/stb_r_int[q])**-2


############parker spiral correction
diff_r=l1_r_int-stb_r_int
#difference in degree along 1 AU circle
diff_r_deg=diff_r/(2*np.pi*1)*360
#time lag due to the parker spiral near 1 AU	- this is negative?? because the spiral leads 
#to an earlier arrival at larger heliocentric distances
time_lag_diff_r=np.round(diff_r_deg/(360/sun_syn),2)
stb_time_to_l1=stb_time_to_l1+time_lag_diff_r


#convert STEREO-A RTN data to GSE as if STEREO-A was along the Sun-Earth line
#[dbr,dbt,dbn]=convert_HEEQ_to_GSE_stb_l1(stb_br_int,stb_bt_int,stb_bn_int,stb_time_to_l1, pos.stb)
#GSE to GSM
#[stb_br_int,stb_bt_int,stb_bn_int]=convert_GSE_to_GSM(dbr,dbt,dbn,stb_time_to_l1)
#stb_btot_int=np.sqrt(stb_br_int**2+stb_bt_int**2+stb_bn_int**2)        

#make Dst 
#[dst_burton, dst_obrien]=make_predstorm_dst(stb_btot_int, stb_bn_int, stb_vtot_int, stb_den_int, stb_time_1h)
#[dst_burton, dst_obrien]=make_predstorm_dst(stb_btot_int, stb_bn_int, stb_vtot_int, stb_den_int, stb_time_to_l1)
#(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):
[dst_burton, dst_obrien, dst_temerinli]=make_predstorm_dst(stb_btot_int, stb_br_int,stb_bt_int,stb_bn_int, stb_vtot_int,stb_vtot_int, stb_den_int, stb_time_to_l1)


dst_burton=np.interp(stb_time_1h,stb_time_to_l1,dst_burton)
dst_temerinli=np.interp(stb_time_1h,stb_time_to_l1,dst_temerinli)


#jetzt die CIR werte filtern
sir_flag=np.zeros(np.size(stb_time_1h))

#for i in np.arange(np.size(dat_ind)):
   #nearest start point of SIR
#   nears=np.argmin(abs(stb_time[dat_ind][i]-sir_start))
   #nearest end point of SIR
#   neare=np.argmin(abs(stb_time[dat_ind][i]-sir_end))


#got through each SIR
for i in np.arange(np.size(sir_start_b)):
      ind_sir=np.where(np.logical_and(stb_time_1h > sir_start_b[i],stb_time_1h < sir_end_b[i]))
      sir_flag[ind_sir]=1

#reduce dat ind to only SIR intervals at STEREO-B
sir_flag_ind=np.where(sir_flag==1) 


plt.figure(3)
plt.plot_date(stb_time_1h[sir_flag_ind],dst_temerinli[sir_flag_ind],'or',markersize=3)
plt.plot_date(stb_time_1h[sir_flag_ind],dst_burton[sir_flag_ind],'ob',markersize=3)
plt.plot_date(stb_time_1h[sir_flag_ind],dst_omni_int[sir_flag_ind],'ok',markersize=3)



#diff=dst_burton[sir_flag_ind]-dst_omni_int[sir_flag_ind]

diff=dst_temerinli[sir_flag_ind]-dst_omni_int[sir_flag_ind]


sns.jointplot(stb_lat_int[sir_flag_ind],diff,kind='hex')
sns.jointplot(stb_long_int[sir_flag_ind],diff,kind='hex')

#sns.jointplot(dst_burton[sir_flag_ind],dst_omni_int[sir_flag_ind],kind='hex')
sns.jointplot(dst_temerinli[sir_flag_ind],dst_omni_int[sir_flag_ind],kind='hex')



#diffm=np.nanmean(dst_burton[sir_flag_ind]-dst_omni_int[sir_flag_ind])
#diffs=np.nanstd(dst_burton[sir_flag_ind]-dst_omni_int[sir_flag_ind])

diffm=np.nanmean(dst_temerinli[sir_flag_ind]-dst_omni_int[sir_flag_ind])
diffs=np.nanstd(dst_temerinli[sir_flag_ind]-dst_omni_int[sir_flag_ind])


print('Diff')
print(diffm)
print(diffs)













