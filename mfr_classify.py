# MFR classification
#
# analyses HELCATS ICMECAT data pattern recognition and types of MFRs
# Author: C. Moestl, Space Research Institute IWF Graz, Austria
# last update: May 2018


from scipy import stats
import scipy.io
from matplotlib import cm
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sunpy.time
import time
import pickle
import seaborn as sns
import pandas as pd

import warnings

warnings.filterwarnings('ignore')


def getcat(filename):
  print('reading CAT')
  cat=scipy.io.readsav(filename, verbose='true')  
  print('done CAT')
  return cat  
  
def decode_array(bytearrin):
 #for decoding the strings from the IDL .sav file to a list of python strings, not bytes 
 #make list of python lists with arbitrary length
 bytearrout= ['' for x in range(len(bytearrin))]
 for i in range(0,len(bytearrin)-1):
  bytearrout[i]=bytearrin[i].decode()
 #has to be np array so to be used with numpy "where"
 bytearrout=np.array(bytearrout)
 return bytearrout  
  

def time_to_num_cat(time_in):  

  #for time conversion from catalogue .sav to numerical time
  #this for 1-minute data or lower time resolution

  #for all catalogues
  #time_in is the time in format: 2007-11-17T07:20:00 or 2007-11-17T07:20Z
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
   time_str[j]
   #convert time to sunpy friendly time and to matplotlibdatetime
   #only for valid times so 9999 in year is not converted
   #pdb.set_trace()
   if year < 2100:
    	  time_num[j]=mdates.date2num(sunpy.time.parse_time(time_str[j]))
   j=j+1  
   #the date format in matplotlib is e.g. 735202.67569444
   #this is time in days since 0001-01-01 UTC, plus 1.
   
   #return time_num which is already an array and convert the list of strings to an array
  return time_num, np.array(time_str)


def IDL_time_to_num(time_in):  
 #convert IDL time to matplotlib datetime
 time_num=np.zeros(np.size(time_in))
 for ii in np.arange(0,np.size(time_in)):
   time_num[ii]=mdates.date2num(sunpy.time.parse_time(time_in[ii]))   
 return time_num 
  

def gaussian(x, amp, mu, sig):
     return amp * exp(-(x-cen)**2 /wid)


def dynamic_pressure(density, speed):
   # make dynamic pressure from density and speed
   #assume pdyn is only due to protons
   #pdyn=np.zeros(len([density])) #in nano Pascals
   protonmass=1.6726219*1e-27  #kg
   pdyn=np.multiply(np.square(speed*1e3),density)*1e6*protonmass*1e9  #in nanoPascal
   return pdyn




#####################################################################################
#################### main program ###############################################

plt.close('all')
print('Start cme_stats_v1.py main program. ICME parameters at all 4 planets.')

#-------------------------------------------------------- get cats


#solar radius
Rs_in_AU=7e5/149.5e6


filename_icmecat='ALLCATS/HELCATS_ICMECAT_v20_SCEQ.sav'
i=getcat(filename_icmecat)

#now this is a scipy structured array  
#access each element of the array see http://docs.scipy.org/doc/numpy/user/basics.rec.html
#access variables
#i.icmecat['id']
#look at contained variables
#print(i.icmecat.dtype)


#get spacecraft and planet positions
pos=getcat('../catpy/DATACAT/positions_2007_2018_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)[0]

#----------------- get all parameters from ICMECAT for easier handling

#id for each event
iid=i.icmecat['id']
#need to decode all strings
iid=decode_array(iid)

#observing spacecraft
isc=i.icmecat['sc_insitu'] #string
isc=decode_array(isc)

#all times need to be converted from the IDL format to matplotlib format
icme_start_time=i.icmecat['ICME_START_TIME']
[icme_start_time_num,icme_start_time_str]=time_to_num_cat(icme_start_time)

mo_start_time=i.icmecat['MO_START_TIME']
[mo_start_time_num,mo_start_time_str]=time_to_num_cat(mo_start_time)

mo_end_time=i.icmecat['MO_END_TIME']
[mo_end_time_num,mo_end_time_str]=time_to_num_cat(mo_end_time)

#this time exists only for Wind
icme_end_time=i.icmecat['ICME_END_TIME']
[icme_end_time_num,icme_end_time_str]=time_to_num_cat(icme_end_time)

sc_heliodistance=i.icmecat['SC_HELIODISTANCE']
sc_long_heeq=i.icmecat['SC_LONG_HEEQ']
sc_lat_heeq=i.icmecat['SC_LAT_HEEQ']
mo_bmax=i.icmecat['MO_BMAX']
mo_bmean=i.icmecat['MO_BMEAN']
mo_bstd=i.icmecat['MO_BSTD']
mo_bzmean=i.icmecat['MO_BZMEAN']
mo_bzmin=i.icmecat['MO_BZMIN']
mo_duration=i.icmecat['MO_DURATION']
mo_mva_axis_long=i.icmecat['MO_MVA_AXIS_LONG']
mo_mva_axis_lat=i.icmecat['MO_MVA_AXIS_LAT']
mo_mva_ratio=i.icmecat['MO_MVA_RATIO']
sheath_speed=i.icmecat['SHEATH_SPEED']
sheath_speed_std=i.icmecat['SHEATH_SPEED_STD']
mo_speed=i.icmecat['MO_SPEED']
mo_speed_st=i.icmecat['MO_SPEED_STD']
sheath_density=i.icmecat['SHEATH_DENSITY']
sheath_density_std=i.icmecat['SHEATH_DENSITY_STD']
mo_density=i.icmecat['MO_DENSITY']
mo_density_std=i.icmecat['MO_DENSITY_STD']
sheath_temperature=i.icmecat['SHEATH_TEMPERATURE']
sheath_temperature_std=i.icmecat['SHEATH_TEMPERATURE_STD']
mo_temperature=i.icmecat['MO_TEMPERATURE']
mo_temperature_std=i.icmecat['MO_TEMPERATURE_STD']
sheath_pdyn=i.icmecat['SHEATH_PDYN']
sheath_pdyn_std=i.icmecat['SHEATH_PDYN_STD']
mo_pdyn=i.icmecat['MO_PDYN']
mo_pdyn_std=i.icmecat['MO_PDYN_STD']


#get indices of events by different spacecraft
ivexind=np.where(isc == 'VEX')
istaind=np.where(isc == 'STEREO-A')
istbind=np.where(isc == 'STEREO-B')
iwinind=np.where(isc == 'Wind')
imesind=np.where(isc == 'MESSENGER')
iulyind=np.where(isc == 'ULYSSES')
imavind=np.where(isc == 'MAVEN')


#take MESSENGER only at Mercury, only events after orbit insertion
imercind=np.where(np.logical_and(isc =='MESSENGER',icme_start_time_num > mdates.date2num(sunpy.time.parse_time('2011-03-18'))))

#limits of solar minimum, rising phase and solar maximum

minstart=mdates.date2num(sunpy.time.parse_time('2007-01-01'))
minend=mdates.date2num(sunpy.time.parse_time('2009-12-31'))

risestart=mdates.date2num(sunpy.time.parse_time('2010-01-01'))
riseend=mdates.date2num(sunpy.time.parse_time('2011-06-30'))

maxstart=mdates.date2num(sunpy.time.parse_time('2011-07-01'))
maxend=mdates.date2num(sunpy.time.parse_time('2014-12-31'))


#extract events by limits of solar min, rising, max, too few events for MAVEN and Ulysses

#extract events by limits of solar min, rising, max, too few events for MAVEN and Ulysses

iallind_min=np.where(np.logical_and(icme_start_time_num > minstart,icme_start_time_num < minend))[0]
iallind_rise=np.where(np.logical_and(icme_start_time_num > risestart,icme_start_time_num < riseend))[0]
iallind_max=np.where(np.logical_and(icme_start_time_num > maxstart,icme_start_time_num < maxend))[0]

iwinind_min=iallind_min[np.where(isc[iallind_min]=='Wind')]
iwinind_rise=iallind_rise[np.where(isc[iallind_rise]=='Wind')]
iwinind_max=iallind_max[np.where(isc[iallind_max]=='Wind')]

ivexind_min=iallind_min[np.where(isc[iallind_min]=='VEX')]
ivexind_rise=iallind_rise[np.where(isc[iallind_rise]=='VEX')]
ivexind_max=iallind_max[np.where(isc[iallind_max]=='VEX')]

imesind_min=iallind_min[np.where(isc[iallind_min]=='MESSENGER')]
imesind_rise=iallind_rise[np.where(isc[iallind_rise]=='MESSENGER')]
imesind_max=iallind_max[np.where(isc[iallind_max]=='MESSENGER')]

istaind_min=iallind_min[np.where(isc[iallind_min]=='STEREO-A')]
istaind_rise=iallind_rise[np.where(isc[iallind_rise]=='STEREO-A')]
istaind_max=iallind_max[np.where(isc[iallind_max]=='STEREO-A')]

istbind_min=iallind_min[np.where(isc[iallind_min]=='STEREO-B')]
istbind_rise=iallind_rise[np.where(isc[iallind_rise]=='STEREO-B')]
istbind_max=iallind_max[np.where(isc[iallind_max]=='STEREO-B')]


#select the events at Mercury extra after orbit insertion, no events for solar minimum!
imercind_min=iallind_min[np.where(np.logical_and(isc[iallind_min] =='MESSENGER',icme_start_time_num[iallind_min] > mdates.date2num(sunpy.time.parse_time('2011-03-18'))))]
imercind_rise=iallind_rise[np.where(np.logical_and(isc[iallind_rise] =='MESSENGER',icme_start_time_num[iallind_rise] > mdates.date2num(sunpy.time.parse_time('2011-03-18'))))]
imercind_max=iallind_max[np.where(np.logical_and(isc[iallind_max] =='MESSENGER',icme_start_time_num[iallind_max] > mdates.date2num(sunpy.time.parse_time('2011-03-18'))))]





############################## get Wind data ################################

print( 'read Wind.')
#get insitu data
win= pickle.load( open( "../catpy/DATACAT/WIND_2007to2018_HEEQ_plasma_median21.p", "rb" ) )
#win_time=IDL_time_to_num(win.temperatureime)
#pickle.dump([win_time], open( "DATACAT/insitu_times_mdates_win_2007_2018.p", "wb" ) )
[win_time]=pickle.load( open( "../catpy/DATACAT/insitu_times_mdates_win_2007_2018.p", "rb" ) )
print( 'read data done.')

#############################################################################





#wind Data: win_time win.bx win.by ... win.vtot win.vy etc.
#MFR times: icme_start_time_num[iwinind] mo_start_time[iwinind]  mo_end_time[iwinind]


############ List of features - go through each MFR and extract values characterising them
# 
# winmfrbzmean=np.zeros(np.size(iwinind))
# winmfrbzstd=np.zeros(np.size(iwinind))
# winmfrbymean=np.zeros(np.size(iwinind))
# winmfrbystd=np.zeros(np.size(iwinind))
# winmfrbxmean=np.zeros(np.size(iwinind))
# winmfrbxstd=np.zeros(np.size(iwinind))
# winmfrbtotmean=np.zeros(np.size(iwinind))
# winmfrbtotstd=np.zeros(np.size(iwinind))
# 
# 
# winmfrvtotmean=np.zeros(np.size(iwinind))
# winmfrvtotstd=np.zeros(np.size(iwinind))
# 
# for p in np.arange(0,np.size(iwinind)):
# 
#     #extract values from MFR data    
#     winmfrbz=win.bz[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],win_time < mo_end_time_num[iwinind[0][p]]) )] 
#     winmfrbzmean[p]=np.nanmean(winmfrbz)
#     winmfrbzstd[p]=np.nanstd(winmfrbz)
# 
#     winmfrby=win.by[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],win_time < mo_end_time_num[iwinind[0][p]]) )] 
#     winmfrbymean[p]=np.nanmean(winmfrby)
#     winmfrbystd[p]=np.nanstd(winmfrby)
#     
#     winmfrbx=win.bx[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],win_time < mo_end_time_num[iwinind[0][p]]) )] 
#     winmfrbxmean[p]=np.nanmean(winmfrbx)
#     winmfrbxstd[p]=np.nanstd(winmfrbx)
#     
#     winmfrbtot=win.btot[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],win_time < mo_end_time_num[iwinind[0][p]]) )] 
#     winmfrbtotmean[p]=np.nanmean(winmfrbtot)
#     winmfrbtotstd[p]=np.nanstd(winmfrbtot)
# 
#     winmfrvtot=win.vtot[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],win_time < mo_end_time_num[iwinind[0][p]]) )] 
#     winmfrvtotmean[p]=np.nanmean(winmfrvtot)
#     winmfrvtotstd[p]=np.nanstd(winmfrvtot)
#     
#     #more values from fits? if Bz is ascending, descending, what are other features to derive?
#  
#   
#collect all derived features in pandas dataframe  

#d = {'btot_mean': winmfrbtotmean, 'btot_std': winmfrbtotstd, 'bx_mean': winmfrbxmean, 'bx_std': winmfrbxstd,'by_mean': winmfrbymean, 'by_std': winmfrbystd,'bz_mean': winmfrbzmean, 'bz_std': winmfrbzstd,'vtot_mean': winmfrvtotmean, 'vtot_std': winmfrvtotstd}
#df = pd.DataFrame(data=d)
#print(df.describe())

#plt.figure(1)
#df.hist()

#plt.figure(2)
#sns.jointplot(df['btot_mean'],df['vtot_mean'])

#plt.figure(5)
#corrmat=df.corr()
#sns.heatmap(corrmat, annot=True)

######################################################################################



############# Version (1)  - prediction of scalar labels with a linear model

############### X define features of first 6 hours


#first 6 hours of each MFR

print('extract features and labels')

feature_hours=4

bzmeanfeature=np.zeros(np.size(iwinind))
bzmeanlabel=np.zeros(np.size(iwinind))

btotmeanfeature=np.zeros(np.size(iwinind))
btotmeanlabel=np.zeros(np.size(iwinind))


#derive features and labels
for p in np.arange(0,np.size(iwinind)):

    #extract values from MFR data   for first 6 hours
    feature_bz=win.bz[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],
                                            win_time < mo_start_time_num[iwinind[0][p]]+feature_hours/24.0) )] 
    bzmeanfeature[p]=np.nanmean(feature_bz)
    
    feature_btot=win.btot[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]],
                                            win_time < mo_start_time_num[iwinind[0][p]]+feature_hours/24.0) )] 
    btotmeanfeature[p]=np.nanmean(feature_btot)

    #time is extracted from 6 hours to end of MFR
    label_bz=win.bz[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]]+feature_hours/24.0,
                                            win_time < mo_end_time_num[iwinind[0][p]]))] 
    bzmeanlabel[p]=np.nanmean(label_bz)

    label_btot=win.btot[np.where(np.logical_and(win_time > mo_start_time_num[iwinind[0][p]]+feature_hours/24.0,
                                            win_time < mo_end_time_num[iwinind[0][p]]))] 

    btotmeanlabel[p]=np.nanmean(label_btot)


#correct nans with mean of array
nans=np.where(np.isnan(bzmeanlabel) == True)
bzmeanlabel[nans]=np.nanmean(bzmeanlabel)

nans=np.where(np.isnan(btotmeanlabel) == True)
btotmeanlabel[nans]=np.nanmean(btotmeanlabel)

nans=np.where(np.isnan(bzmeanfeature) == True)
bzmeanfeature[nans]=np.nanmean(bzmeanfeature)

nans=np.where(np.isnan(btotmeanfeature) == True)
btotmeanfeature[nans]=np.nanmean(btotmeanfeature)



#fl feature labels
fl = {'bzmeanfeature': bzmeanfeature, 'btotmeanfeature': btotmeanfeature, 'bzmeanlabel': bzmeanlabel, 'btotmeanlabel': btotmeanlabel}
#fl frame
flf = pd.DataFrame(data=fl)
print(flf.describe())

#features
#convert to correct format
X1=btotmeanfeature
X1=X1.reshape(-1,1)

X=flf['btotmeanfeature']
#labels
y=flf['btotmeanlabel']


#check data

plt.figure(1)
flf.hist()


plt.figure(5)
corrmat=flf.corr()
sns.heatmap(corrmat, annot=True)



import sklearn.linear_model as lm
lr=lm.LinearRegression()
lr.fit(X1,y)


print('R**2 score')
print(lr.score(X1,y))



y_pred = lr.predict(X1)

print('mean absolute error ')
from sklearn.metrics import  mean_absolute_error
print(mean_absolute_error(y_pred,y))



plt.figure(2)
sns.jointplot(X,y,kind='reg')


plt.figure(3)
plt.plot(X,y,'bo')
plt.plot(X1,y_pred,'k-')





sys.exit()

###################same for Bz

#features
#convert to correct format
X1=bzmeanfeature
X1=X1.reshape(-1,1)

X=flf['bzmeanfeature']
#labels
y=flf['bzmeanlabel']




import sklearn.linear_model as lm
lr=lm.LinearRegression()
lr.fit(X1,y)


print('R**2 score')
print(lr.score(X1,y))





#split in training and test data does not work ...

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=8, stratify=y)






sys.exit()





############# Version (2)  - prediction of vector timeseries label with a neural network




#############################
#Questions?


#1 classify MFRs for type - need to classify manually first?

#split into training and test data

#2 how to predict the rest of the MFR if first 10, 20, 30, 40, 50% are seen?
#everything should be automatically with a deep learning method

#wie image recognition - aus einem Teil-Bild (Zeitserie der MFR + features sind ein 2D array)
#das Rest Bild vorhersagen

#oder Zeitserie der ersten 20% rein - klassifizieren - dann schauen wie es am wahrscheinlichsten weitergeht
#durch pattern vergleich dieser Kategorie


#die feature sind die Zeitserien über die ersten 20% und die parameter - label ist die gesamte Zeitserie

#aufteilen in training und 




