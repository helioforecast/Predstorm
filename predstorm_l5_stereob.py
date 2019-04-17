'''
Testing PREDSTORM solar wind prediction with STEREO-B
Author: C. Moestl 
started April 2018

python 3.7 with sunpy 


training data:
use stereo one hour data as training data set, corrected for 1 AU
use VEX and MESSENGER as tests for HelioRing like forecasts, use STEREO at L5 for training data of the last few days

forecast plot:
add approximate levels of Dst for each location to see aurora, taken from ovation prime/worldview and Dst 
add Temerin and Li method and kick out Burton/OBrien; make error bars for Dst
take mean of ensemble forecast for final blue line forecast or only best match?


combined L1/L5 forecast
most important: implement pattern recognition for STEREO-A streams, and link this to the most probably outcome days later at L1
train with STB data around the location where STA is at the moment
das ist sofort ein paper!

 - prediction als txt file und variables rauschreiben
 - coordinate conversions checken (GSE to GSM ok)
 - problem sind cmes bei stereo, mit listen ausschliessen aus training data
 - eigene tests mit semi-supervised, also sodass known intervals benutzt werden)
 - eigenes programm zum hindsight testen mit stereo-a und omni (da ist dst auch drin)
 (das ist alles ein proposal... zb h2020, erc)
 irgendwann forecast wie wetter app machen auf website und verschiedene locations und ovation (aber eben nur für background wond)
 CCMC Bz berücksichtigen
 fundamental: how much is Bz stable for HSS from L5 to L1? are there big changes?
 is Bz higher for specific locations with respect to the HCS and the solar equator? 
 probabilities for magnetic storm magnitude, probabilities for aurora for many locations
 link with weather forcecast, wie am Handy in der app 
 am wichtigsten: validation sodass die % stimmen!!

temporal and spatial coherence
(1) STEREO-A and Earth are within 0.1° heliospheric latitude, so they could see similar parts of the stream + (2) the coronal hole hasn't changed much in the last few days.

'''


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
import scipy.io
import pdb

from predstorm_module import getpositions
from predstorm_module import make_dst_from_wind
from predstorm_module import time_to_num_cat
from predstorm_module import converttime
from predstorm_module import get_omni_data
from predstorm_module import convert_GSE_to_GSM


################################## INPUT PARAMETERS ######################################

# GET INPUT PARAMETERS
from predstorm_l5_input import *
#######################################################



def round_to_hour(dt):
    '''
    round datetime objects to nearest hour
    '''

    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + datetime.timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour

    return dt


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

#closes all plots
plt.close('all')

print()
print('PREDSTORM L1 solar wind forecasting. ')
print('Error estimation using STEREO-B data')
print()
print('Christian Moestl, IWF-helio, Graz, last update April 2019.')








########################################## (1) GET DATA ##################################



########### get STEREO-B in situ data

print( 'read STEREO-B data. Change directory if necessary.')
stb= pickle.load( open( "../catpy/DATACAT/STB_2007to2014_HEEQ.p", "rb" ) )
#stb_time=IDL_time_to_num(stb.time)
#pickle.dump([stb_time], open( "../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "wb" ) )
[stb_time]=pickle.load( open( "../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "rb" ) )
print( 'read data done.')


############# get sc positions
print('load spacecraft and planetary positions')
pos=getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
pos_time_num=time_to_num_cat(pos.time)
           
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
plt.plot_date(pos_time_num,l1_lat_heeq-stb_lat_heeq,'-b',label='STEREO-B')
plt.plot_date(pos_time_num,l1_lat_heeq-sta_lat_heeq,'-r',label='STEREO-A')
plt.xlabel('time')
plt.ylabel('Earth latitude - STEREO latitude [° HEEQ]')
plt.legend()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(50,50,600, 500)

######### get OMNI data
print('load OMNI data for Dst')
o=pickle.load(open('data/omni2_all_years_pickle.p', 'rb') )
# old variable names [spot,btot,bx,by,bz,bygsm, bzgsm,speed,speedx, dst,kp,den,pdyn,year,day,hour,times1]= pickle.load( open( "data/omni2_all_years_pickle.p", "rb" ) )

fig=plt.figure(2)
plt.plot_date(o.time,o.dst,'k')
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(50,600,600, 500)


######## load Lan Jian's SIR list for STEREO
sir=np.genfromtxt('../catpy/ALLCATS/CIRCAT master/STEREO_Level3_SIR_clean_clean.txt',
                  dtype='S4,S1,i4,S5,S5,i4,S5,S5')
sir_start=np.zeros(len(sir))                  
sir_end=np.zeros(len(sir))                  
sir_sc=np.empty(len(sir), dtype='str')     
      
    
#############          *************************
#make matplotlib times for SIR start and end times                  


#b'2007', b'A',  56, b'02/25', b'17:40',  60, b'03/01', b'01:10'


for i in np.arange(len(sir)):                  
    
    #STEREO A or B
    sir_sc[i]=str(sir[i][1])[2]
    
    #extract hours with minutes as fraction of day 
    hours=int(sir[i][4][0:2])/24+int(sir[i][4][3:5])/(24*60)
    sir_start[i]=mdates.date2num(datetime.datetime(int(sir[i][0]),1,1,)+datetime.timedelta(int(sir[i][2])-1)+datetime.timedelta(hours) )

    hours=int(sir[i][7][0:2])/24+int(sir[i][7][3:5])/(24*60)
    sir_end[i]=mdates.date2num(datetime.datetime(int(sir[i][0]),1,1,)+datetime.timedelta(int(sir[i][5])-1)+datetime.timedelta(hours) )

    
#extract SIRs only for STB
stbsirs=np.where(sir_sc=='B')
sir_start_b=sir_start[stbsirs]           
sir_end_b=sir_end[stbsirs] 


#########################################################################################



############################### (2) Interpolate STEREO-B to 1 hour ##################################


#convert STB to 1 hour time resolution and add time shift

# extract intervals from STB data, from -10 to -120 degree HEEQ longitude
#first get the selected times from the position data 
min_time=(pos_time_num[np.where(stb_long_heeq < -10)[0][0]])
max_time=(pos_time_num[np.where(stb_long_heeq < -110)[0][0]])

#then extract the position data and in situ data indices for these times
pos_ind=np.where(np.logical_and(pos_time_num > min_time, pos_time_num< max_time))
dat_ind=np.where(np.logical_and(stb_time > min_time, stb_time< max_time))



     
stb_time_sel=stb_time[dat_ind]
stb_btot_sel=stb.btot[dat_ind]
stb_bx_sel=stb.bx[dat_ind]
stb_by_sel=stb.by[dat_ind]
stb_bz_sel=stb.bz[dat_ind]
stb_den_sel=stb.density[dat_ind]
stb_vtot_sel=stb.vtot[dat_ind]


#interpolate STB data
#need to round the start time so it starts with full hours
start_time=round(stb_time[dat_ind][0],1)
end_time=round(stb_time[dat_ind][-1],1)

#make time array, hourly steps starting with start time and finishing at end time
#do this with datetime objects
stb_time_1h = mdates.num2date(start_time) + np.arange(0,int(np.ceil((end_time-start_time)*24))    ,1) * datetime.timedelta(hours=1) 
#convert back to matplotlib time
stb_time_1h=mdates.date2num(stb_time_1h)

#identify not NaN data points
good = np.where(np.isfinite(stb_btot_sel))
#interpolate
stb_btot_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_btot_sel[good])
good = np.where(np.isfinite(stb_bx_sel))
stb_bx_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_bx_sel[good])
good = np.where(np.isfinite(stb_by_sel))
stb_by_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_by_sel[good])
good = np.where(np.isfinite(stb_bz_sel))
stb_bz_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_bz_sel[good])

#same for plasma
good = np.where(np.isfinite(stb_vtot_sel))
stb_vtot_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_vtot_sel[good])
good = np.where(np.isfinite(stb_den_sel))
stb_den_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_den_sel[good])

#interpolate positions to hourly (better with heliopy positions, but is STEREO-B included there?)
stb_r_1h=np.interp(stb_time_1h,pos_time_num,stb_r)
stb_long_1h=np.interp(stb_time_1h,pos_time_num,stb_long_heeq)
stb_lat_1h=np.interp(stb_time_1h,pos_time_num,stb_lat_heeq)
l1_r_1h=np.interp(stb_time_1h,pos_time_num,l1_r)

plt.figure(3)
plt.plot_date(stb_time_1h,stb_btot_1h,'-k',label='Btot')
plt.plot_date(stb_time_1h,stb_bx_1h,'-r',label='Bx')
plt.plot_date(stb_time_1h,stb_by_1h,'-g',label='By')
plt.plot_date(stb_time_1h,stb_bz_1h,'-b',label='Btot')
plt.xlabel('time')
plt.ylabel('B [nT]')
plt.legend()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,50,600, 500)



print()
print('get times STB inside SIR')

########################




    

plt.figure(4)
plt.plot_date(stb_time_1h,stb_den_1h*5,'-r',label='Den*5')
plt.plot_date(stb_time_1h,stb_vtot_1h,'-k',label='Vtot')
plt.xlabel('time')
plt.ylabel('V, N*5')
plt.legend()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,600,600, 500)







#go through all SIRs and check for times when STEREO-B is inside a SIR

#round SIR start times to nearest hour
sir_ind=np.arange(0,np.size(sir_start_b),1)
sir_start_1h=np.array([])
sir_end_1h=np.array([])

for i in sir_ind:
  sir_start_1h=np.append(sir_start_1h,mdates.date2num(round_to_hour(mdates.num2date(sir_start_b[i]))))
  sir_end_1h=np.append(sir_end_1h,mdates.date2num(round_to_hour(mdates.num2date(sir_end_b[i]))))

#get indices for STEREO-B data start and end (better inside) SIRs
stb_sir_ind_1h=np.int(0)

for p in sir_ind:
    inds=np.argmin(abs(sir_start_1h[p]-stb_time_1h))
    inde=np.argmin(abs(sir_end_1h[p]-stb_time_1h))
    plt.axvspan(stb_time_1h[inds],stb_time_1h[inde],color='b',alpha=0.2)

    #append all indices between start and end times
    if np.min(abs(sir_start_1h[p]-stb_time_1h)) < 1: 
        #print(inds,'  ',inde,np.arange(inds,inde,1))
        stb_sir_ind_1h=np.append(stb_sir_ind_1h,np.arange(inds,inde,1)) 
  
#delete 0 from array definition  
np.delete(stb_sir_ind_1h,0)  

print('Data points inside SIRs, percent: ',len(stb_time_1h[stb_sir_ind_1h])/len(stb_time_1h)*100 )

stb_500_1h=np.zeros(len(stb_time_1h))
for k in np.arange(0,len(stb_time_1h),1):
    if stb_vtot_1h[k] > 500: stb_500_1h[k]=500
    if stb_vtot_1h[k] < 500: stb_500_1h[k]=np.nan


stb_600_1h=np.zeros(len(stb_time_1h))
for k in np.arange(0,len(stb_time_1h),1):
    if stb_vtot_1h[k] > 600: stb_600_1h[k]=600
    if stb_vtot_1h[k] < 600: stb_600_1h[k]=np.nan
         
                 
plt.plot(stb_time_1h,stb_500_1h,'g',linestyle='-',linewidth=5)
plt.plot(stb_time_1h,stb_600_1h,'y',linestyle='-',linewidth=5)


sys.exit()














####################################### APPLY CORRECTIONS TO STEREO-B data 
#(1) make correction for heliocentric distance of L1 position - take position of Earth and STEREO from pos file 
#(2) add timeshift for each datapoint


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













