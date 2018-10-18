#This is the module for the predstorm package containing functions and procedures

import numpy as np
import scipy
import copy
import sunpy
import matplotlib.dates as mdates
import ephem

# use import importlib
# importlib.reload(predstorm_module)
# to update module while working in command line 


# LIST OF FUNCTIONS AND PROCEDURES

# **************************************

#

#


#


def getpositions(filename):  
    pos=scipy.io.readsav(filename)  
    print
    print('positions file:', filename) 
    return pos


def sphere2cart(r, phi, theta):
    #convert spherical to cartesian coordinates
    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
    return (x, y, z) 


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
 
 
 
 
 
  
def convert_RTN_to_GSE_sta_l1(cbr,cbt,cbn,ctime,pos_stereo_heeq,pos_time_num):

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
  
 return (dstcalc1,dstcalc2, dst_temerin_li_out)     