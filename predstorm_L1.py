##predstorm real time solar wind forecasting

#predicting the L1 solar wind and Dst index with unsupervised pattern recognition
#algorithms Riley et al. 2017, Owens et al. 2017
#soon Möstl et al. 2018 3DCORE, Reiss et al. background wind
#Author: C. Moestl 
#started April 2018

#This work is published under the MIT LICENSE
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
#INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
#PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
#FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
#TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
#OTHER DEALINGS IN THE SOFTWARE.

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


##############################################################################
############################# CODE START
##############################################################################



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

import scipy
from scipy import stats
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import pickle
import os
import copy
import pdb
import urllib
import json
import ephem
import seaborn as sns
import sunpy.time

from predstorm_module import make_dst_from_wind
from predstorm_module import sunriseset
from predstorm_module import getdata


######################## initialize
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
rbx=np.zeros(len(mr))


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
 rbx[k]=float(mr[k][1])
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
rbx24=np.interp(rbtimes24,rbtime_num,rbx)

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
rbx7=np.interp(rtimes7,rbtime_num,rbx)
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
#make_dst_from_wind(btot_in,bx_in, by_in,bz_in,v_in,vx_in,density_in,time_in):#
[rdst_burton, rdst_obrien, rdst_temerin_li]=make_dst_from_wind(rbtot7,rbx7,rbygsm7,rbzgsm7,rpv7,rpv7,rpn7,rtimes7)




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
#plt.plot_date(rtimes7+1/24, rdst_burton,'-b', label='Dst Burton et al. 1975',markersize=3, linewidth=1)
#plt.plot_date(rtimes7+1/24, rdst_obrien,'-k', label='Dst OBrien & McPherron 2000',markersize=3, linewidth=1)
plt.plot_date(rtimes7+1/24, rdst_temerin_li,'-r', label='Dst Temerin Li 2002',markersize=3, linewidth=1)

#**** This error is only a placeholder
error=15#
plt.fill_between(rtimes7+1/24, rdst_temerin_li-error, rdst_temerin_li+error, alpha=0.2)




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
 pickle.dump([spot,btot,bx,by,bz,bygsm,bzgsm,speed,speedx, dst,kp, den,pdyn,year,day,hour,times1], open( "cats/omni2save_april2018.p", "wb" ) ) 
else: [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "cats/omni2save_april2018.p", "rb" ) )


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
bxn=rbx24

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
corr_count_bx=np.zeros(trainsize)
corr_count_v=np.zeros(trainsize)
corr_count_n=np.zeros(trainsize)

#these are the arrays for the squared distances between now wind and training data
dist_count_b=np.zeros(trainsize)
dist_count_bz=np.zeros(trainsize)
dist_count_by=np.zeros(trainsize)
dist_count_bx=np.zeros(trainsize)
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
  dist_count_by[i]=np.sqrt(np.sum((bygsmn-bygsms)**2))/np.size(bygsmn)

  #same for bx
  bxs=bx[inds:inds+deltat+1]
  dist_count_bx[i]=np.sqrt(np.sum((bxn-bxs)**2))/np.size(bxn)
  

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


#for Bx

#method with minimum rms distance
maxval_bx=np.nanmin(dist_count_bx)
maxpos_bx=np.argmin(dist_count_bx)
top50_bx=np.argsort(dist_count_bx)[0:49]

print('find minimum of BzGSM distance at index:')
print(round(maxval_bx,1), ' nT   index: ',maxpos_bx)
#go forward in time from training data set start to the position of the best match + deltat hours 
#(so you take the future part coming after wind where the best match is seen)
indp_bx=startindex+maxpos_bx+deltat
#select array from OMNI data for predicted wind - predictions all have a p at the end
bxp=bx[indp_bx:indp_bx+deltat+1]




#for ByGSM

#method with minimum rms distance
maxval_by=np.nanmin(dist_count_by)
maxpos_by=np.argmin(dist_count_by)
top50_by=np.argsort(dist_count_by)[0:49]

print('find minimum of BzGSM distance at index:')
print(round(maxval_by,1), ' nT   index: ',maxpos_by)
#go forward in time from training data set start to the position of the best match + deltat hours 
#(so you take the future part coming after wind where the best match is seen)
indp_by=startindex+maxpos_by+deltat
#select array from OMNI data for predicted wind - predictions all have a p at the end
byp=bygsm[indp_by:indp_by+deltat+1]






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

bxdst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
bydst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
bzdst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
speeddst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)
dendst=np.zeros(np.size(timesnb)+np.size(timesfb)-1)

#write times in one array, note the overlap at the now time
timesdst[:25]=timesnb
timesdst[25:49]=timesfb[1:]

btotdst[:25]=btotn
btotdst[25:49]=btotp[1:]

bxdst[:25]=bxn
bxdst[25:49]=bxp[1:]

bydst[:25]=bygsmn
bydst[25:49]=byp[1:]

bzdst[:25]=bzgsmn
bzdst[25:49]=bzp[1:]

speeddst[:25]=speedn
speeddst[25:49]=speedp[1:]

dendst[:25]=denn
dendst[25:49]=denp[1:]


#[dst_burton]=make_predstorm_dst(btoti, bygsmi, bzgsmi, speedi, deni, timesi)
#old [pdst_burton, pdst_obrien]=make_predstorm_dst(btotdst,bzdst, speeddst, dendst, timesdst)
[pdst_burton, pdst_obrien, pdst_temerin_li]=make_dst_from_wind(btotdst,bxdst,bydst,bzdst,speeddst,speeddst,dendst,timesdst)



ax8 = fig.add_subplot(414)


#******************** added timeshift of 1 hour for L1 to Earth! This should be different for each timestep to be exact
#predicted dst
#plt.plot_date(timesdst+1/24, pdst_burton+15,'b-', label='Dst Burton et al. 1975',markersize=5, linewidth=1)
#plt.plot_date(timesdst+1/24, pdst_obrien+15,'r-', label='Dst OBrien & McPherron 2000',markersize=5, linewidth=1)
plt.plot_date(timesdst+1/24, pdst_temerin_li,'r-', label='Dst Temerin & Li 2002',markersize=5, linewidth=1)


#**** This error is only a placeholder
error=15#
#plt.fill_between(cdst_time+1/24, dst_burton-error, dst_burton+error, alpha=0.2)
#plt.fill_between(cdst_time+1/24, dst_obrien-error, dst_obrien+error, alpha=0.2)
plt.fill_between(timesdst+1/24, pdst_temerin_li-error, pdst_temerin_li+error, alpha=0.2)


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

##########################################################################################
################################# CODE STOP ##############################################
##########################################################################################

