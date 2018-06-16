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

#DSCOVR data:
#Nans for missing data should be handled better and interpolated over, OBrien stops with Nans

#training data:
#use stereo one hour data as training data set, corrected for 1 AU
#use VEX and MESSENGER as tests for HelioRing like forecasts, use STEREO at L5 for training data of the last few days

#forecast plot:
#add approximate levels of Dst for each location to see aurora, taken from ovation prime/worldview and Dst 
#add Temerin and Li method and kick out Burton/OBrien; make error bars for Dst
#take mean of ensemble forecast for final blue line forecast or only best match?



################################## INPUT PARAMETERS ######################################

#whether to show interpolated data points on the DSCOVR input plot
showinterpolated=1

#read in data from omni file -> 1 , from save_file -> 0
data_from_omni_file = 0 #

#the time interval for both the observed and predicted wind (** could be longer for predicted wind)
#Delta T in hours, start with 24 hours here (covers 1 night of aurora)
deltat=24

#take 4 solar minimum years as training data for 2018
trainstart='2006-Jan-01 00:00'
trainend='2010-Jan-01 00:00'


#######################################################






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
  
  
  
  
  
def make_predstorm_dst(btot_in,bz_in,v_in,density_in,time_in):
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
 pdyn1=density_in*1e6*protonmass*1e-27*(v_in*1e3)**2*1e9  #in nanoPascal
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
	#  
	#  #start with Temerin and Li 2002 (implemented here) and proceed to Temerin and Li (2006) later
	#  
	#  #define all terms
	#  dst1=np.zeros(len(bz_in))
	#  dst2=np.zeros(len(bz_in))
	#  dst3=np.zeros(len(bz_in))
	#  pressureterm=np.zeros(len(bz_in))
	#  directterm=np.zeros(len(bz_in))
	#  offset=np.zeros(len(bz_in))
	#  dst_temerin_li_out=np.zeros(len(bz_in))
	#  bp=np.zeros(len(bz_in))
	#  bt=np.zeros(len(bz_in))
	#  
	#  
	#  #define inital values (needed for convergence, see Temerin and Li 2002 note)
	#  dst1[0:10]=-15
	#  dst2[0:10]=-13
	#  dst3[0:10]=-2
	#  
	#  
	# 
	#  #define all constants
	#  p1=0.9
	#  p2=2.18e-4
	#  p3=14.7
	#  
	#  s1=-2.788
	#  s2=1.44
	#  s3=-0.92
	#  s4=-1.054e-2
	#  s5=8.6e-6
	#  
	#  a1=6.51e-2
	#  a2=1.37
	#  a3=8.4e-3  
	#  a4=6.053e-3
	#  a5=1.12e-3
	#  a6=1.55e-3
	#  
	#  tau1=0.14 #days
	#  tau2=0.18 #days
	#  tau3=9e-2 #days
	#  
	#  b1=0.792
	#  b2=1.326
	#  b3=1.29e-2
	#  
	#  c1=-24.3
	#  c2=5.2e2
	# 
	#  #Note: vx has to be used with a positive sign throughout the calculation
	#  
	#  #problem with t - days since 1995 dont work after 1999
	#  
	#  
	#  
	#  #----------------------------------------- loop over each timestep
	#  for i in range(len(bz_in)-1):
	# 
	#   #for 3DCORE use constant density in cm-3
	#   #if density is nan then set to 5
	#   
	#   if np.isnan(density_in[i]) > 0: density_in[i]=5
	# 
	#   #define terms
	#   
	#   #1 pressure term ****************** scheint OK etwa +30 nT
	#   #t time in days since beginning of 1995   #1 Jan 1995 in Julian days
	#   t1=sunpy.time.julian_day(mdates.num2date(time_in[i]))-sunpy.time.julian_day('1995-1-1 00:00')
	#   
	#   
	#   
	#   #Temerin and Li 2002
	#   yearli=365.24 
	#   tt=2*np.pi*t1/yearli
	#   ttt=2*np.pi*t1
	#   alpha=0.078
	#   beta=1.22
	#   cosphi=np.sin(tt+alpha)*np.sin(ttt-tt-beta)*(9.58589e-2)+np.cos(tt+alpha)*(0.39+0.104528*np.cos(ttt-tt-beta))
	# 
	#   
	#   #not that in Temerin and Li's code, these times are implemented very differently! - starting point for generalization
	#   #to arbitrary dates
	#   
	#   #equation 1 use phi from equation 2
	#   sinphi=(1-cosphi**2)**0.5
	#   
	#   
	#   pressureterm[i]=(p1*(btot_in[i]**2)+density_in[i]*((p2*((v_in[i])**2)/(sinphi**2.52))+p3))**0.5
	#   #2 directbzterm ************scheint OK etwa 0.5 (siehe paper, das passt)
	#   directterm[i]=0.478*bz_in[i]*(sinphi**11.0)
	# 
	#   #3 offset  ** stimmt sicher nicht fuer 2012, t1 macht keinen Sinn, wert 300 viel zu hoch, weglassen derweil (in TL 2006 auch nur bis 2002)
	#   offset[i]=s1+s2*np.sin(2*np.pi*t1/yearli+s3)+s4*t1+s5*(t1**2)
	# 
	#   bt[i]=(by_in[i]**2+bz_in[i]**2)**0.5  
	# 
	#   #mistake in paper - bt is similarly defined as bp (with by bz); but in Temerin and Li's code (dst.pro) bp depends on by and bx
	#   bp[i]=(by_in[i]**2+bx_in[i]**2)**0.5  
	# 
	#   
	#   #contains t1, but in cos and sin so it does not matter
	#   dh=bp[i]*np.cos(np.arctan2(bx_in[i],by_in[i])+6.10)  *   ((3.59e-2)*np.cos(2*np.pi*t1/yearli+0.04)-2.18e-2*np.sin(2*np.pi*t1-1.60))
	#   
	#   
	#   theta_li=-(np.arccos(-bz_in[i]/bt[i])-np.pi)/2
	#   
	#   exx=1e-3*abs(vx_in[i])*bt[i]*np.sin(theta_li)**6.1
	#  
	#   #t1 and dt are in days
	#   dt=sunpy.time.julian_day(mdates.num2date(time_in[i+1]))-sunpy.time.julian_day(mdates.num2date(time_in[i]))
	# 
	#  
	#   #4 dst1  *********** scheint zu passen
	# 
	#   #find value of dst1(t-tau1) 
	#   #timesi ist im matplotlib format ist in Tagen: im times den index suchen wo timesi-tau1 am nächsten ist
	#   #und dann bei dst1 den wert mit dem index nehmen der am nächsten ist, das ist dann dst(t-tau1)
	#   #wenn index nicht existiert (am anfang) einfach index 0 nehmen
	#   #check for index where timesi is greater than t minus tau
	#   indtau1=np.where(timesi > (timesi[i]-tau1))
	#   dst1tau1=dst1[indtau1[0][0]]
	#   #similar search for others  
	#   dst2tau1=dst2[indtau1[0][0]]
	#  
	#   th1=0.725*(sinphi**-1.46)
	#   th2=1.83*(sinphi**-1.46)
	#   fe1=(-4.96e-3)*  (1+0.28*dh)*  (2*exx+abs(exx-th1)+abs(exx-th2)-th1-th2)*  (abs(vx_in[i])**1.11)*((density_in[i])**0.49)*(sinphi**6.0)
	# 
	#   dst1[i+1]=dst1[i]+  (a1*(-dst1[i])**a2   +fe1*   (1+(a3*dst1tau1+a4*dst2tau1)/(1-a5*dst1tau1-a6*dst2tau1)))  *dt
	#   
	#   
	#   #5 dst2    
	#   indtau2=np.where(timesi > (timesi[i]-tau2))
	#   dst1tau2=dst1[indtau2[0][0]]
	#   df2=(-3.85e-8)*(abs(vx_in[i])**1.97)*(btot_in[i]**1.16)*(np.sin(theta_li)**5.7)*((density_in[i])**0.41)*(1+dh)
	#   fe2=(2.02*1e3)*(sinphi**3.13)*df2/(1-df2)
	#    
	#   dst2[i+1]=dst2[i]+(b1*(-dst2[i])**b2+fe2*(1+(b3*dst1tau2)/(1-b3*dst1tau2)))*dt
	#   
	#   
	#   #6 dst3  
	#   
	#   indtau3=np.where(timesi > (timesi[i]-tau3))
	#   dst3tau3=dst3[indtau3[0][0]]
	#   df3=-4.75e-6*(abs(vx_in[i])**1.22)*(bt[i]**1.11)*np.sin(theta_li)**5.5*((density_in[i])**0.24)*(1+dh)
	#   fe3=3.45e3*sinphi**0.9*df3/(1-df3)
	#   
	#   dst3[i+1]=dst3[i]+  (c1*dst3[i]   + fe3*(1+(c2*dst3tau3)/(1-c2*dst3tau3)))*dt
	#   
	#    
	#   #print(dst1[i], dst2[i], dst3[i], pressureterm[i], directterm[i], offset[i])
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
	#   offset[i]=6
	#   
	#   #for 2013 don't use dst3 - makes spikes
	#   #dst_temerin_li_out[i]=dst1[i]+dst2[i]+dst3[i]+pressureterm[i]+directterm[i]	+offset[i]
	#   dst_temerin_li_out[i]=dst1[i]+dst2[i]+pressureterm[i]+directterm[i]	+offset[i]
	# 
	#  
	#   #---------------- loop over
	#   
  

 return (dstcalc1,dstcalc2)
   
   

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







######################################## MAIN PROGRAM ####################################

#get current directory
#os.system('pwd')
#closes all plots
plt.close('all')

print()
print()
print('PREDSTORM -method for geomagnetic storm and aurora forecasting. ')
print('Based on results by Riley et al. 2017 Space Weather, and')
print('Owens, Riley and Horbury 2017 Solar Physics. ')
print()
print('This is a pattern recognition technique that searches ')
print('for similar intervals in historic data as the current solar wind.')
print()
print('It is currently an unsupervised learning method.')
print()
print('This is the real time version by Christian Moestl, IWF Graz, Austria. Last update: April 2018. ')
print()
print('-------------------------------------------------')







######################### (1) get real time DSCOVR data ##################################

#see https://docs.python.org/3/howto/urllib2.html
#data from http://services.swpc.noaa.gov/products/solar-wind/

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


#convert variables to numpy arrays
#mag
for k in np.arange(0,len(mr),1):

 #handle missing data, they show up as None from the JSON data file
 if mr[k][6] is None: mr[k][6]=np.nan
 if mr[k][3] is None: mr[k][3]=np.nan
 if mr[k][2] is None: mr[k][2]=np.nan

 rbtot[k]=float(mr[k][6])
 rbzgsm[k]=float(mr[k][3])
 rbygsm[k]=float(mr[k][2])
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


###calculate Dst for DSCOVR last 7 day data with Burton and OBrien
#this is the last 24 hours in 1 hour timesteps, 25 data points
#start on next day 0 UT, so rbtimes7 contains values at every full hour like the real Dst
rtimes7=np.arange(np.ceil(rbtime_num)[0],rbtime_num[-1],1.0000/24)
rbtot7=np.interp(rtimes7,rbtime_num,rbtot)
rbzgsm7=np.interp(rtimes7,rbtime_num,rbzgsm)
rbygsm7=np.interp(rtimes7,rbtime_num,rbygsm)
rpv7=np.interp(rtimes7,rptime_num,rpv)
rpn7=np.interp(rtimes7,rptime_num,rpn)

#interpolate NaN values in the hourly interpolated data ******* to add 


#------ get Dst of last 7 days
#-------
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


#make Dst index from solar wind data
[rdst_burton, rdst_obrien]=make_predstorm_dst(rbtot7, rbzgsm7, rpv7, rpn7, rtimes7)



##################### plot DSCOVR data
sns.set_context("talk")     
sns.set_style("darkgrid")  
fig=plt.figure(1,figsize=(12,10)) #fig=plt.figure(1,figsize=(14,14))
weite=1
fsize=11
msize=5

#panel 1
ax4 = fig.add_subplot(411)
plt.plot_date(rbtime_num, rbtot,'-k', label='B total', linewidth=weite)
if showinterpolated > 0: plt.plot_date(rbtimes24, rbtot24,'ro', label='B total interpolated last 24 hours',linewidth=weite,markersize=msize)
plt.plot_date(rbtime_num, rbzgsm,'-g', label='Bz GSM',linewidth=weite)
if showinterpolated > 0: plt.plot_date(rbtimes24, rbzgsm24,'go', label='Bz GSM interpolated last 24 hours',linewidth=weite,markersize=msize)


#indicate 0 level for Bz
plt.plot_date([rtimes7[0], rtimes7[-1]], [0,0],'--k', alpha=0.5, linewidth=1)


#test interpolation
#plt.plot_date(rtimes7, rbzgsm7,'-ko', label='B7',linewidth=weite)

plt.ylabel('Magnetic field [nT]',  fontsize=fsize+2)
myformat = mdates.DateFormatter('%Y %b %d %Hh')
ax4.xaxis.set_major_formatter(myformat)
ax4.legend(loc='upper left', fontsize=fsize-2,ncol=4)
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]])
plt.ylim(np.nanmin(rbzgsm)-10, np.nanmax(rbtot)+10)
plt.title('L1 DSCOVR real time solar wind provided by NOAA SWPC for '+ str(mdates.num2date(timenowb))[0:16]+ ' UT', fontsize=16)
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)


#panel 2
ax5 = fig.add_subplot(412)
#add speed levels
plt.plot_date([rtimes7[0], rtimes7[-1]], [400,400],'--k', alpha=0.3, linewidth=1)
plt.annotate('slow',xy=(rtimes7[0],400),xytext=(rtimes7[0],400),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]], [800,800],'--k', alpha=0.3, linewidth=1)
plt.annotate('fast',xy=(rtimes7[0],800),xytext=(rtimes7[0],800),color='k', fontsize=10	)

plt.plot_date(rptime_num, rpv,'-k', label='V observed',linewidth=weite)
if showinterpolated > 0: plt.plot_date(rptimes24, rpv24,'ro', label='V interpolated last 24 hours',linewidth=weite,markersize=msize)
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]])
#plt.plot_date(rtimes7, rpv7,'-ko', label='B7',linewidth=weite)


plt.ylabel('Speed $\mathregular{[km \\ s^{-1}]}$', fontsize=fsize+2)
ax5.xaxis.set_major_formatter(myformat)
ax5.legend(loc=2,fontsize=fsize-2,ncol=2)
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]])
plt.ylim([np.nanmin(rpv)-50,np.nanmax(rpv)+100])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

#panel 3
ax6 = fig.add_subplot(413)
plt.plot_date(rptime_num, rpn,'-k', label='N observed',linewidth=weite)
if showinterpolated > 0:  plt.plot_date(rptimes24, rpn24,'ro', label='N interpolated last 24 hours',linewidth=weite,markersize=msize)
plt.ylabel('Density $\mathregular{[ccm^{-3}]}$',fontsize=fsize+2)
ax6.xaxis.set_major_formatter(myformat)
ax6.legend(loc=2,ncol=2,fontsize=fsize-2)
plt.ylim([0,np.nanmax(rpn)+10])
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

#panel 4
ax6 = fig.add_subplot(414)

#model Dst
#******* added timeshift of 1 hour for L1 to Earth! This should be different for each timestep to be exact
plt.plot_date(rtimes7+1/24, rdst_burton,'-b', label='Dst Burton et al. 1975',markersize=3, linewidth=1)
plt.plot_date(rtimes7+1/24, rdst_obrien,'-r', label='Dst OBrien & McPherron 2000',markersize=3, linewidth=1)

#real Dst
#for AER
#plt.plot_date(rtimes7, rdst7,'ko', label='Dst observed',markersize=4)
#for Kyoto
plt.plot_date(rdst_time, rdst,'ko', label='Dst observed',markersize=4)


plt.ylabel('Dst [nT]', fontsize=fsize+2)
ax6.xaxis.set_major_formatter(myformat)
ax6.legend(loc=2,ncol=3,fontsize=fsize-2)
plt.xlim([np.ceil(rbtime_num)[0],rbtime_num[-1]])
plt.ylim([np.nanmin(rdst_burton)-50,50])
plt.xticks(fontsize=fsize)
plt.yticks(fontsize=fsize)

#add geomagnetic storm levels
plt.plot_date([rtimes7[0], rtimes7[-1]], [-50,-50],'--k', alpha=0.3, linewidth=1)
plt.annotate('moderate',xy=(rtimes7[0],-50+2),xytext=(rtimes7[0],-50+2),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]], [-100,-100],'--k', alpha=0.3, linewidth=1)
plt.annotate('intense',xy=(rtimes7[0],-100+2),xytext=(rtimes7[0],-100+2),color='k', fontsize=10)
plt.plot_date([rtimes7[0], rtimes7[-1]], [-250,-250],'--k', alpha=0.3, linewidth=1)
plt.annotate('super-storm',xy=(rtimes7[0],-250+2),xytext=(rtimes7[0],-250+2),color='k', fontsize=10)



plt.tight_layout()

#save plot 
filename='real/predstorm_realtime_input_1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.jpg'
plt.savefig(filename)
#filename='real/predstorm_realtime_input_1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.eps'
#plt.savefig(filename)







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
 

plt.ylabel('Bz [nT] GSM')
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
plt.plot_date(timesdst+1/24, pdst_burton+15,'b-', label='Dst Burton et al. 1975',markersize=5, linewidth=1)
plt.plot_date(timesdst+1/24, pdst_obrien+15,'r-', label='Dst OBrien & McPherron 2000',markersize=5, linewidth=1)


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

#save variables

filename_save='real/savefiles/predstorm_realtime_pattern_save_v1_'+timenowstr[0:10]+'-'+timenowstr[11:13]+'_'+timenowstr[14:16]+'.p'
print('All variables for plot saved in ', filename_save, ' for later verification usage.')
pickle.dump([timenowb, rbtime_num, rbtot, rbygsm, rbzgsm, rtimes7, rbtot7, rbygsm7, rbzgsm7, rbtimes24, rbtot24,rbygsm24,rbzgsm24, rptime_num, rpv, rpn, rtimes7, rpv7, rpn7, rptimes24, rpn24, rpv24,rdst_time, rdst, timesdst, pdst_burton, pdst_obrien], open(filename_save, "wb" ) )

