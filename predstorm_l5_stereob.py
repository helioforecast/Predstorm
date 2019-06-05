'''
Testing PREDSTORM solar wind prediction with STEREO-B
Author: C. Moestl 
last update April 2019

python 3.7 anaconda with sunpy, predstorm_module

to do: 
- remove CMEs in situ at STB from test time range
- add conversion STB HEEQ to GSE to GSM
- more error analyses (point (4))
- make function to create position file

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

'''


#IMPORT 
from scipy import stats
import scipy
import sys
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num
import numpy as np
import sunpy.time
import pickle
import scipy.io
import seaborn as sns
import os
import pdb

import predstorm as ps
# Old imports (remove later)
from predstorm.data import SatData, getpositions
from predstorm.data import time_to_num_cat, converttime, round_to_hour
from predstorm.predict import make_dst_from_wind
from predstorm.data import convert_GSE_to_GSM

################################## INPUT PARAMETERS ######################################

# GET INPUT PARAMETERS
from predstorm_l5_input import *
##########################################################################################


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
        sunpy_time=sunpy.time.break_time(num2date(ctime[i]))
        jd[i]=sunpy.time.julian_day(sunpy_time)
        mjd[i]=float(int(jd[i]-2400000.5)) #use modified julian date    
        #then lambda_sun
        T00=(mjd[i]-51544.5)/36525.0
        dobj=num2date(ctime[i])
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

    return (bxgse,bygse,bzgse)



######################################## MAIN PROGRAM ####################################

logger = ps.init_logging(verbose=True)

AU=149597870.700 #AU in km

#closes all plots
plt.close('all')

print()
print('PREDSTORM L1 solar wind forecasting. ')
print('Error estimation using STEREO-B data')
print()
print('Christian Moestl, IWF-helio, Graz, last update April 2019.')


##########################################################################################
########################################## (1) GET DATA ##################################
##########################################################################################

########### get STEREO-B in situ data

logger.info( 'read STEREO-B data. Change path to the data file if necessary.')
try:
    stb_p= pickle.load( open( "../catpy/DATACAT/STB_2007to2014_HEEQ.p", "rb" ) )
except:
    # NOTE: SCEQ/HEEQ makes small difference in results
    stb_p= pickle.load( open( "data/STB_2007to2014_SCEQ.p", "rb" ) )
#convert time - takes a while, so save pickle and reload when done
#stb_time=IDL_time_to_num(stb.time)
#pickle.dump([stb_time], open( "../catpy/DATACAT/insitu_times_mdates_stb_2007_2014.p", "wb" ) )
[stb_time]=pickle.load( open( "data/insitu_times_mdates_stb_2007_2014.p", "rb" ) )
print( 'read data done.')

# Pack STEREO-B data into SatData object
# TODO: move this into normal read eventually
stb = SatData({'time': stb_time,
               'btot': stb_p.btot, 'bx': stb_p.bx, 'by': stb_p.by, 'bz': stb_p.bz,
               'speed': stb_p.vtot, 'density': stb_p.density, 'temp': stb_p.temperature},
               source='STEREO-B')
stb.h['DataSource'] = "STEREO-B (pickled)"
stb.h['SamplingRate'] = stb_p.time[1] - stb_p.time[0]
stb.h['CoordinateSystem'] = 'SCEQ'


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
omni = ps.get_omni_data()

fig=plt.figure(2)
plt.plot_date(omni['time'],omni['dst'],'k')
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(50,600,600, 500)


######## load Lan Jian's SIR list for STEREO
sir=np.genfromtxt('data/STEREO_Level3_SIR_clean_clean.txt',
                  dtype='S4,S1,i4,S5,S5,i4,S5,S5')
sir_start=np.zeros(len(sir))                  
sir_end=np.zeros(len(sir))                  
sir_sc=np.empty(len(sir), dtype='str')     
      
#make matplotlib times for SIR start and end times                  
#format: b'2007', b'A',  56, b'02/25', b'17:40',  60, b'03/01', b'01:10'
for i in np.arange(len(sir)):                  
    #STEREO A or B
    sir_sc[i]=str(sir[i][1])[2]
    #extract hours with minutes as fraction of day 
    hours=int(sir[i][4][0:2])/24+int(sir[i][4][3:5])/(24*60)
    sir_start[i]=date2num(datetime(int(sir[i][0]),1,1,)+timedelta(int(sir[i][2])-1)+timedelta(hours) )
    hours=int(sir[i][7][0:2])/24+int(sir[i][7][3:5])/(24*60)
    sir_end[i]=date2num(datetime(int(sir[i][0]),1,1,)+timedelta(int(sir[i][5])-1)+timedelta(hours) )
    
#extract SIRs only for STB
stbsirs=np.where(sir_sc=='B')
sir_start_b=sir_start[stbsirs]           
sir_end_b=sir_end[stbsirs] 


##########################################################################################
############################### (2) Interpolate STEREO-B to 1 hour #######################
##########################################################################################


# extract intervals from STB data, from -10 to -120 degree HEEQ longitude
#first get the selected times from the position data 
min_time=(pos_time_num[np.where(stb_long_heeq < -10)[0][0]])
max_time=(pos_time_num[np.where(stb_long_heeq < -110)[0][0]])

stbh = stb.make_hourly_data()
stbh = stbh.cut(starttime=num2date(np.round(min_time,0)), endtime=num2date(np.round(max_time,0))).interp_nans()
omni = omni.cut(starttime=num2date(np.round(min_time,0)), endtime=num2date(np.round(max_time,0)))

#then extract the position data and in situ data indices for these times, cut 1 array dimension
pos_ind=np.where(np.logical_and(pos_time_num > min_time, pos_time_num< max_time))[0]
dat_ind=np.where(np.logical_and(stb_time > min_time, stb_time< max_time))[0]

stb_time_sel=stbh['time']#stb_time[dat_ind]
stb_btot_1h=stbh['btot']
stb_bx_1h=stbh['bx']
stb_by_1h=stbh['by']
stb_bz_1h=stbh['bz']
stb_den_1h=stbh['density']
stb_vtot_1h=stbh['speed']

#interpolate STB data
#round the start time so it starts and ends with 00:00 UT
start_time=np.round(stb_time_sel[0],0)  
end_time=np.round(stb_time_sel[-1],0)  

#make time array, hourly steps starting with start time and finishing at end time
#do this with datetime objects
stb_time_1h = num2date(start_time) + np.arange(0,int(np.ceil((end_time-start_time)*24))    ,1) * timedelta(hours=1) 
#convert back to matplotlib time
stb_time_1h=date2num(stb_time_1h)

#identify not NaN data points
# good = np.where(np.isfinite(stb_btot_sel))
# #interpolate
# stb_btot_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_btot_sel[good])
# good = np.where(np.isfinite(stb_bx_sel))
# stb_bx_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_bx_sel[good])
# good = np.where(np.isfinite(stb_by_sel))
# stb_by_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_by_sel[good])
# good = np.where(np.isfinite(stb_bz_sel))
# stb_bz_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_bz_sel[good])

# #same for plasma
# good = np.where(np.isfinite(stb_vtot_sel))
# stb_vtot_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_vtot_sel[good])
# good = np.where(np.isfinite(stb_den_sel))
# stb_den_1h=np.interp(stb_time_1h,stb_time_sel[good],stb_den_sel[good])

#interpolate positions to hourly (better with heliopy positions, but is STEREO-B included there?)
stb_r_1h=np.interp(stb_time_1h,pos_time_num,stb_r)
stb_long_1h=np.interp(stb_time_1h,pos_time_num,stb_long_heeq)
stb_lat_1h=np.interp(stb_time_1h,pos_time_num,stb_lat_heeq)
l1_r_1h=np.interp(stb_time_1h,pos_time_num,l1_r)
l1_lon_1h=np.interp(stb_time_1h,pos_time_num,l1_long_heeq)
l1_lat_1h=np.interp(stb_time_1h,pos_time_num,l1_lat_heeq)

#plot data
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

#plot data
plt.figure(4)
plt.plot_date(stb_time_1h,stb_den_1h*5,'-r',label='Den*5')
plt.plot_date(stb_time_1h,stb_vtot_1h,'-k',label='Vtot')
plt.xlabel('time')
plt.ylabel('V, N*5')
plt.legend()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,600,600, 500)


print()
print('get times STB inside SIR')
    
#go through all SIRs and check for times when STEREO-B is inside a SIR

#round SIR start times to nearest hour
sir_ind=np.arange(0,np.size(sir_start_b),1)
sir_start_1h=np.array([])
sir_end_1h=np.array([])

for i in sir_ind:
    sir_start_1h=np.append(sir_start_1h,date2num(round_to_hour(num2date(sir_start_b[i]))))
    sir_end_1h=np.append(sir_end_1h,date2num(round_to_hour(num2date(sir_end_b[i]))))

######### get indices for STEREO-B data inside SIRs
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
print('Data points inside SIRs, percent: ',np.round(len(stb_time_1h[stb_sir_ind_1h])/len(stb_time_1h)*100,1) )
###########


####### get intervals for different HSS cutoffs (note: includes CMEs, but very few anyway in this timeframe)

print()
print('STEREO-B ...')

stb_450_ind_1h=np.where(stb_vtot_1h>450)[0]  
print('Data points inside > 450 km/s, percent: ',np.round(len(stb_time_1h[stb_450_ind_1h])/len(stb_time_1h)*100,1) )

stb_500_ind_1h=np.where(stb_vtot_1h>500)[0]  
print('Data points inside > 500 km/s, percent: ',np.round(len(stb_time_1h[stb_500_ind_1h])/len(stb_time_1h)*100,1) )

stb_550_ind_1h=np.where(stb_vtot_1h>550)[0]  
print('Data points inside > 550 km/s, percent: ',np.round(len(stb_time_1h[stb_550_ind_1h])/len(stb_time_1h)*100,1) )

stb_600_ind_1h=np.where(stb_vtot_1h>600)[0]  
print('Data points inside > 600 km/s, percent: ',np.round(len(stb_time_1h[stb_600_ind_1h])/len(stb_time_1h)*100,1) )

stb_650_ind_1h=np.where(stb_vtot_1h>650)[0]  
print('Data points inside > 650 km/s, percent: ',np.round(len(stb_time_1h[stb_650_ind_1h])/len(stb_time_1h)*100,1) )


#plot with markers to indicate intervals
plt.plot(stb_time_1h[stb_500_ind_1h],np.zeros(len(stb_500_ind_1h))+500,'g',marker='o',markersize=3,linestyle='None')

#get dst only for interval
odst=omni['dst']  


##########################################################################################
########################### (3) APPLY STEREO-B MAPPING TO L1 #############################
##########################################################################################


# (1) make correction for heliocentric distance of STEREO-B to L1 position
#go through all data points of the selected interval:
for i in np.arange(0,len(l1_r_1h),1):
    stb_btot_1h[i]=stb_btot_1h[i]   *(l1_r_1h[i]/stb_r_1h[i])**-2
    stb_bx_1h[i]  =stb_bx_1h[i]     *(l1_r_1h[i]/stb_r_1h[i])**-2
    stb_by_1h[i]  =stb_by_1h[i]     *(l1_r_1h[i]/stb_r_1h[i])**-2
    stb_bz_1h[i]  =stb_bz_1h[i]     *(l1_r_1h[i]/stb_r_1h[i])**-2
    stb_den_1h[i] =stb_den_1h[i]    *(l1_r_1h[i]/stb_r_1h[i])**-2
    
print()
print('Mapping of STEREO-B data:')
print('1: mean increase of B and N by factor ',np.round(np.mean((l1_r_1h/stb_r_1h)**-2),3)   )


# (2) correction for timing for the Parker spiral see
# Simunac et al. 2009 Ann. Geophys. equation 1, see also Thomas et al. 2018 Space Weather
# difference in heliocentric distance STEREO-B to Earth,
# actually different for every point so take average of solar wind speed
# Omega is 360 deg/sun_syn in days, convert to seconds; sta_r in AU to m to km;
# convert to degrees
# minus sign: from STEREO-A to Earth the diff_r_deg needs to be positive
# because the spiral leads to a later arrival of the solar wind at larger
# heliocentric distances (this is reverse for STEREO-B!)

#initialize array with correct size
timelag_diff_r=np.zeros(len(l1_r_1h))

# define time lag from STEREO-A to Earth
timelag_stb_l1=abs(stb_long_1h)/(360/sun_syn) #days

#got through all data points
for i in np.arange(0,len(l1_r_1h),1):
    diff_r_deg=(-360/(sun_syn*86400))*((stb_r_1h[i]-l1_r_1h[i])*AU)/stb_vtot_1h[i]
    timelag_diff_r[i]=np.round(diff_r_deg/(360/sun_syn),3)

## ADD BOTH time shifts to the stb_time_1h
stb_time_1h_to_l1=stb_time_1h+timelag_stb_l1+timelag_diff_r

print('2: time lag min to max , days:', np.round(np.min(timelag_stb_l1+timelag_diff_r),1),  '    ',  np.round(np.max(timelag_stb_l1+timelag_diff_r),1))

print('no 3: coordinate conversion of magnetic field components')

########## TO DO: NEED TO CHECK
#convert STEREO-B RTN data to GSM as if STEREO-B was along the Sun-Earth line
#[dbx,dby,dbz]=convert_HEEQ_to_GSE_stb_l1(stb_bx_1h,stb_by_1h,stb_bz_1h,stb_time_1h_to_l1, pos.stb)
#GSE to GSM
#[stb_bx_1h,stb_by_1h,stb_bz_1h]=convert_GSE_to_GSM(dbx,dby,dbz,stb_time_1h_to_l1)

## plot comparison to show timeshift mapping
plt.figure(5)
plt.plot_date(stb_time_1h_to_l1,stb_vtot_1h,'-b',label='V shift STB')
plt.plot_date(stb_time_1h,stb_vtot_1h,'--g',label='V STB')
plt.plot_date(omni['time'],omni['speed'],'-k',label='V L1')
plt.xlabel('time')
plt.ylabel('Vtot')
plt.legend()
plt.xlim([date2num(sunpy.time.parse_time('2010-Feb-1')), date2num(sunpy.time.parse_time('2010-Mar-1'))])
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(1350,600,600, 500)
#mngr.window.setGeometry(50,50,1500, 800)
plt.grid()


##########################################################################################
########################## (4) ANALYSE results ###########################################
##########################################################################################

print()
print('Dst calculation takes a few seconds')

dst_temerinli = stbh.make_dst_prediction()['dst']

############ Dst comparison plot

plt.figure(6)
plt.plot_date(stb_time_1h_to_l1,dst_temerinli,'-b',label='Dst shift STB')
plt.plot_date(omni['time'],odst,'-k',label='observed Dst')
plt.xlabel('time')
plt.ylabel('Dst')
plt.legend()
plt.xlim([date2num(sunpy.time.parse_time('2010-Feb-1')), date2num(sunpy.time.parse_time('2010-Mar-1'))])
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(1350,50,600, 500)
#mngr.window.setGeometry(50,50,1500, 800)
plt.grid()


############ comparison observed/calculated for subintervals:
print()

print('Time range covered for analysis:')
print('start:',num2date(min_time))
print('end:',num2date(max_time))

print('Parts of STEREO-B data for comparison: all data points')# > 500 km/s at STB')

stb_all_ind_1h=np.where(stb_vtot_1h>0)[0]  

#all above 500 km/s at STB (for other interval definitions see above)
#indices_for_comparison=stb_500_ind_1h
#all data points 
indices_for_comparison=stb_all_ind_1h


#difference predicted to observed Dst for > 500 km/s at STB
dst_diff=dst_temerinli[indices_for_comparison]-odst[indices_for_comparison]
dst_diffm=np.nanmean(dst_temerinli[indices_for_comparison]-odst[indices_for_comparison])
dst_diffs=np.nanstd(dst_temerinli[indices_for_comparison]-odst[indices_for_comparison])
print('Dst diff mean +/- std: {:.1f} +/- {:.1f}'.format(dst_diffm, dst_diffs))



##### dependence of Dst error on difference in latitude
diff_lat_heeq=stb_lat_1h[indices_for_comparison]-l1_lat_1h[indices_for_comparison]

plt.figure(7)
#number of events per degree bin
(histlat, bin_edges) = np.histogram(diff_lat_heeq, np.arange(-11,12,1))

diff_lat_bin_mean_dst=np.zeros(len(bin_edges))
diff_lat_bin_std_dst=np.zeros(len(bin_edges))

for i in np.arange(0,len(bin_edges)-1,1):
   diff_lat_bin_mean_dst[i]=np.mean(dst_diff[np.where(np.logical_and(diff_lat_heeq > bin_edges[i],diff_lat_heeq < bin_edges[i+1])    )  ])
   diff_lat_bin_std_dst[i]=np.std(dst_diff[np.where(np.logical_and(diff_lat_heeq > bin_edges[i],diff_lat_heeq < bin_edges[i+1])    )  ])

#plt.plot(bin_edges,diff_lat_bin_mean_dst,marker='o',markersize=10,color='r', linestyle='None')
plt.errorbar(bin_edges,diff_lat_bin_mean_dst,diff_lat_bin_std_dst,marker='o',markersize=10,color='r', linestyle='None')
plt.xlabel('Diff HEEQ latitude STB to L1', fontsize=10)
plt.ylabel('Dst C-O', fontsize=10)


#other way of plotting
#h=sns.jointplot(diff_lat_heeq,dst_diff,kind='hex')
#h.set_axis_labels('Diff HEEQ latitude STB to L1', 'Dst C-O', fontsize=10)
plt.tight_layout()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(50,300,600, 500)



########### dependence of Dst error on longitude
plt.figure(8)
h=sns.jointplot(stb_long_1h[indices_for_comparison],dst_diff,kind='hex')
h.set_axis_labels('HEEQ longitude STB', 'Dst C-O', fontsize=10)
#plt.plot(stb_long_1h[indices_for_comparison],dst_diff)
#plt.plot(diff_lat_heeq,odst[indices_for_comparison],kind='hex')
plt.tight_layout()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(700,300,600, 500)



########### predicted vs observed Dst

plt.figure(9)
h=sns.jointplot(dst_temerinli[indices_for_comparison],odst[indices_for_comparison],kind='hex')
h.set_axis_labels('Predicted Dst', 'Observed Dst', fontsize=10)
plt.tight_layout()
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(1300,300,600, 500)

plt.show()






