# MFR classification
#
# analyses HELCATS ICMECAT for predicting labels of CME MFRs
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

sns.set_context("talk")     
sns.set_style("darkgrid")  

plt.close('all')

######## functions

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
######################## main program ###############################################

plt.close('all')
print('MFR classify.')

#-------------------------------------------------------- get cats

#solar radius
Rs_in_AU=7e5/149.5e6


filename_icmecat='../catpy/ALLCATS/HELCATS_ICMECAT_v20_SCEQ.sav'
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











#############################
#Questions?


#1 classify MFRs for type - need to classify manually first?

#split into training and test data


#wie image recognition - aus einem Teil-Bild (Zeitserie der MFR + features sind ein 2D array)
#das Rest Bild vorhersagen

#oder Zeitserie der ersten 20% rein - klassifizieren - dann schauen wie es am wahrscheinlichsten weitergeht
#durch pattern vergleich dieser Kategorie


#die feature sind die Zeitserien über die ersten 20% und die parameter - label ist die gesamte Zeitserie







##################### (1) classify background wind, sheath, MFR based on 4 features in hourly data


#go through all wind data and check for each hour whether this is inside a MFR or sheath or background wind

#3 labels: background wind, sheath, MFR
#data: about 1e5 training instances (without test data) with 4 features: average btot, std btot, vtot, vstd

#get the features and labels for ICME classification

get_fl_classify=False
#get_fl_classify=True

interval_hours=2

#takes 22 minutes for full data
if get_fl_classify:
  
  start = time.time()
  print('extract features for classification')

  #test CME extraction April 2010
  #win=win[1690000:1720000]
  #win_time=win_time[1690000:1720000]

  win=win[1500000:2000000]
  win_time=win_time[1500000:2000000]

  #win=win[0:2500000]
  #win_time=win_time[0:2500000]


  hour_int_size=round(np.size(win)/(60*interval_hours))-1

  btot_ave_hour=np.zeros(hour_int_size)
  btot_std_hour=np.zeros(hour_int_size)
  bmax_hour=np.zeros(hour_int_size)

  bz_ave_hour=np.zeros(hour_int_size)
  bz_std_hour=np.zeros(hour_int_size)

  by_ave_hour=np.zeros(hour_int_size)
  by_std_hour=np.zeros(hour_int_size)

  bx_ave_hour=np.zeros(hour_int_size)
  bx_std_hour=np.zeros(hour_int_size)
  
  vtot_ave_hour=np.zeros(hour_int_size)
  vtot_std_hour=np.zeros(hour_int_size)
  vmax_hour=np.zeros(hour_int_size)

  t_ave_hour=np.zeros(hour_int_size)
  t_std_hour=np.zeros(hour_int_size)
  n_ave_hour=np.zeros(hour_int_size)
  n_std_hour=np.zeros(hour_int_size)
  
  time_hour=np.zeros(hour_int_size)
  

  #get features for 1 hour steps
  for p in np.arange(hour_int_size):

     
    #extract index for current hour 
    indexrange=np.where(np.logical_and(win_time > win_time[0]+p*interval_hours/24.0,win_time < win_time[0]+p*interval_hours/24.0+interval_hours/24.0))

    btot_ave_hour[p]=np.mean(win.btot[indexrange] )
    btot_std_hour[p]=np.std(win.btot[indexrange] )	
    bmax_hour[p]=np.max(win.btot[indexrange] )

    bx_ave_hour[p]=np.mean(win.btot[indexrange] )
    bx_std_hour[p]=np.std(win.btot[indexrange] )	

    by_ave_hour[p]=np.mean(win.btot[indexrange] )
    by_std_hour[p]=np.std(win.btot[indexrange] )	

    bz_ave_hour[p]=np.mean(win.btot[indexrange] )
    bz_std_hour[p]=np.std(win.btot[indexrange] )	
																																										
    vtot_ave_hour[p]=np.mean(win.vtot[indexrange] )
    vtot_std_hour[p]=np.std(win.vtot[indexrange] )
    vmax_hour[p]=np.max(win.vtot[indexrange] )
																																														
																																														
				#add temperature, density		
				
    t_ave_hour[p]=np.mean(win.temperature[indexrange] )
    t_std_hour[p]=np.std(win.temperature[indexrange] )
    n_ave_hour[p]=np.mean(win.density[indexrange] )
    n_std_hour[p]=np.std(win.density[indexrange] )

    time_hour[p]=win_time[0]+p*interval_hours/24+0.5*interval_hours/24 
  
  print('extract features done.')																																														

  #plot features over original time series

  ############# 
  #plt.figure(1)
  #plt.plot_date(win_time, win.btot,'k-', alpha=0.4)
  #plt.plot_date(time_hour, btot_std_hour)
  #plt.plot_date(time_hour, btot_ave_hour)
  #plt.tight_layout()
  
  
  #plt.figure(2)
  #plt.plot_date(win_time, win.vtot,'k-', alpha=0.4)
  #plt.plot_date(time_hour, vtot_std_hour)
  #plt.plot_date(time_hour, vtot_ave_hour)
  #plt.tight_layout()


  print('extract label for each hour interval with ICMECAT')

  #old try
  #1 if fully inside a sheath, 0 otherwise, same for others
  #sheath=np.zeros(hour_int_size)
  #mfr=np.zeros(hour_int_size)
  #bwind=np.zeros(hour_int_size)+1
  
  
  #classify: bwind =0, sheath=1, mfr=2
  sw_label=np.zeros(hour_int_size)

  for p in np.arange(hour_int_size):
  
  
    #icme_start_time_num[iwinind], mo_start_time_num[iwinind], mo_end_time_num[iwinind]
  
    #first try: check all ICMECAT events for each hour timestep
  
    for i in np.arange(np.size(iwinind)):
         #when current time is between ICME_START_TIME and MO_START_TIME, its a sheath    
         if np.logical_and(time_hour[p] > icme_start_time_num[iwinind][i],time_hour[p] < mo_start_time_num[iwinind][i]):
                sw_label[p]=1
         #this is a MFR       
         elif np.logical_and(time_hour[p] > mo_start_time_num[iwinind][i],time_hour[p] < mo_end_time_num[iwinind][i]):
                sw_label[p]=2

     


  #make nans to averages ***maybe better interpolation?
  nans=np.where(np.isnan(btot_ave_hour) == True)
  btot_ave_hour[nans]=np.nanmean(btot_ave_hour)

  nans=np.where(np.isnan(btot_std_hour) == True)
  btot_std_hour[nans]=np.nanmean(btot_std_hour)

  nans=np.where(np.isnan(bmax_hour) == True)
  bmax_hour[nans]=np.nanmean(bmax_hour)

  nans=np.where(np.isnan(bx_ave_hour) == True)
  bx_ave_hour[nans]=np.nanmean(bx_ave_hour)

  nans=np.where(np.isnan(bx_std_hour) == True)
  bx_std_hour[nans]=np.nanmean(bx_std_hour)

  nans=np.where(np.isnan(by_ave_hour) == True)
  by_ave_hour[nans]=np.nanmean(by_ave_hour)

  nans=np.where(np.isnan(by_std_hour) == True)
  by_std_hour[nans]=np.nanmean(by_std_hour)

  np.where(np.isnan(bz_ave_hour) == True)
  bz_ave_hour[nans]=np.nanmean(bz_ave_hour)

  nans=np.where(np.isnan(bz_std_hour) == True)
  bz_std_hour[nans]=np.nanmean(bz_std_hour)

  nans=np.where(np.isnan(vtot_ave_hour) == True)
  vtot_ave_hour[nans]=np.nanmean(vtot_ave_hour)

  nans=np.where(np.isnan(vtot_std_hour) == True)
  vtot_std_hour[nans]=np.nanmean(vtot_std_hour)
  
  nans=np.where(np.isnan(vmax_hour) == True)
  vmax_hour[nans]=np.nanmean(vmax_hour)

 
  nans=np.where(np.isnan(t_ave_hour) == True)
  t_ave_hour[nans]=np.nanmean(t_ave_hour)
  nans=np.where(np.isnan(t_std_hour) == True)
  t_std_hour[nans]=np.nanmean(t_std_hour)
  
  nans=np.where(np.isnan(n_ave_hour) == True)
  n_ave_hour[nans]=np.nanmean(n_ave_hour)
  nans=np.where(np.isnan(n_std_hour) == True)
  n_std_hour[nans]=np.nanmean(n_std_hour)


  # #save features and labels
#   #fl feature labels pandas dataframe
#   fl = {'B_ave': btot_ave_hour, 'B_std': btot_std_hour,'Bmax': bmax_hour, 
#         'Bx_ave': bx_ave_hour, 'Bx_std': bx_std_hour,
#         'By_ave': by_ave_hour, 'By_std': by_std_hour,
#         'Bz_ave': bz_ave_hour, 'Bz_std': bz_std_hour,        
#         'V_ave': vtot_ave_hour, 'V_std': vtot_std_hour,'Vmax': vmax_hour,
#         'T_ave': t_ave_hour,  'T_std': t_std_hour,
#         'N_ave': n_ave_hour,  'N_std': n_std_hour, 'sw_label': sw_label}
#   flf = pd.DataFrame(data=fl)
#   
#   #features: right format for sklearn, deeplearning, ...
#   
#   
#   #all features
#   X=np.zeros([hour_int_size,16])  
#   X[:,0]=btot_ave_hour
#   X[:,1]=btot_std_hour
#   X[:,2]=bmax_hour
#   X[:,3]=bx_ave_hour
#   X[:,4]=bx_std_hour
#   X[:,5]=by_ave_hour
#   X[:,6]=by_std_hour
#   X[:,7]=bz_ave_hour
#   X[:,8]=bz_std_hour
#   X[:,9]=vtot_ave_hour
#   X[:,10]=vtot_std_hour
#   X[:,11]=vmax_hour
#   X[:,12]=t_ave_hour
#   X[:,13]=t_std_hour
#   X[:,14]=n_ave_hour
#   X[:,15]=n_std_hour
# 

  #camporeale 2017 Xu 2015 : design features that allow discrimination
  
   
    
  alfv=np.sqrt(btot_ave_hour**2)/n_ave_hour  
  texp=((vtot_ave_hour/258)**3.113)/t_ave_hour
    
  fl = {'V_tot': vtot_ave_hour, 'T_ave': t_ave_hour,  'T_std': t_std_hour,
        'Alfven speed': alfv, 'T_exp':texp,'sw_label': sw_label}

  flf = pd.DataFrame(data=fl)
  
  X=np.zeros([hour_int_size,5])  
  
  X[:,0]=vtot_ave_hour
  X[:,1]=t_ave_hour
  X[:,2]=t_std_hour
  X[:,3]=alfv #alfven speed
  X[:,4]=texp #texp/t

  

  #X[:,1]=bmax_hour
  #X[:,2]=vtot_ave_hour
  #X[:,3]=vmax_hour
  #X[:,5]=n_ave_hour


  
  #labels: make one hot encoding for label array
  from keras.utils.np_utils import to_categorical
  y = to_categorical(sw_label, 3) #3 means number of categories
 
  #split into training and test data  
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

  
  
  #pickle.dump([btot_ave_hour, btot_std_hour, vtot_ave_hour, vtot_std_hour, time_hour,sheath, mfr, bwind], open( "mfr_classify/mfr_classify_features_labels_save_50p.p", "wb" ) ) 
  #pickle.dump([flf,X,y, X_train, X_test, y_train, y_test, sw_label], open( "mfr_classify/mfr_classify_features_labels_all_new.p", "wb" ) ) 
  #pickle.dump([flf,X,y, X_train, X_test, y_train, y_test,sw_label], open( "mfr_classify/mfr_classify_features_labels_small.p", "wb" ) ) 
  pickle.dump([flf,X,y, X_train, X_test, y_train, y_test,sw_label], open( "mfr_classify/mfr_classify_features_labels_small_campo.p", "wb" ) ) 


  print('labels extracted and saved. time in minutes:')
  end = time.time()
  print((end - start)/60)
########################################################

if get_fl_classify == False:
    #[btot_ave_hour, btot_std_hour, vtot_ave_hour, vtot_std_hour, time_hour,sheath, mfr, bwind]= pickle.load( open( "mfr_classify/mfr_features_labels_save_50p.p", "rb" ) )
    #[flf,X,y, X_train, X_test, y_train, y_test,sw_label]=pickle.load( open( "mfr_classify/mfr_classify_features_labels_all_new.p", "rb" ) )
    [flf,X,y, X_train, X_test, y_train, y_test,sw_label]=pickle.load( open( "mfr_classify/mfr_classify_features_labels_small_campo.p", "rb" ) )


    #[flf,X,y, X_train, X_test, y_train, y_test]=pickle.load( open( "mfr_classify/mfr_classify_features_labels_small.p", "rb" ) )

    print('loaded features for classification')



#check data histograms
flf.hist(bins=20,figsize=(10,10))
plt.tight_layout()
filename='mfr_classify/classify_hist.png'
plt.savefig(filename)


pd.plotting.scatter_matrix(flf,figsize=(10,10))
plt.tight_layout()
filename='mfr_classify/classify_scatter_matrix.png'
plt.savefig(filename)









###################### use a SVM

#use here the original sw_label

print()

from sklearn.model_selection import train_test_split
X_trains, X_tests, y_trains, y_tests = train_test_split(X, sw_label, test_size=0.20, random_state=42, stratify=y)

from sklearn.svm import SVC

svc = SVC(kernel='rbf', C=1e6, gamma=1e-5,verbose=True)
clf=svc.fit(X_trains, y_trains)


#for c in [ 1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]:
#  print(c)
#for gam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]:
#  print(gam)
#  svc = SVC(kernel='rbf', C=1,gamma=gam)#,verbose=True)

#svc = SVC(kernel='linear', C=1, gamma=1,verbose=True)


y_preds = clf.predict(X_test)
print()
print('Test score')
print(svc.score(X_tests, y_tests))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_tests, y_preds))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_tests, y_preds))

#print('ypreds')
#print(y_preds.tolist())


#model2 = SVC(kernel='rbf', C=1E6, gamma=1.)
#model2.fit(X, sw_label)
#print(model2.score(X,sw_label))





######################## ANN

#ANN with 1 hidden layer

from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

#input layer
inputs = Input(shape=(5, ))
#fully connected hidden layer
fc = Dense(5)(inputs)
#output
predictions = Dense(3, activation='softmax')(fc)

model = Model(input=inputs, output=predictions)

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100)

#loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128) 
#print(loss_and_metrics)

y_pred = model.predict(X_test)#, batch_size=128) 


y_testlab=np.zeros(len(y_test))
y_predlab=np.zeros(len(y_test))

for q in np.arange(len(y_test)):
    if y_test[q][0]==1: y_testlab[q]=0
    if y_test[q][1]==1: y_testlab[q]=1
    if y_test[q][2]==1: y_testlab[q]=2

    if y_pred[q][0]==1: y_predlab[q]=0
    if y_pred[q][1]==1: y_predlab[q]=1
    if y_pred[q][2]==1: y_predlab[q]=2



from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_testlab, y_predlab))

#from sklearn.metrics import accuracy_score
#print(accuracy_score(y_test, y_pred))

################ train model









sys.exit()








