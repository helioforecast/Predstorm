#!/usr/bin/env python
"""
Functions for index predictions.

Author: C. Moestl, R. Bailey, IWF Graz, Austria
twitter @chrisoutofspace, https://github.com/cmoestl
started April 2018, last update May 2019

Python 3.7
Packages not included in anaconda installation: sunpy, cdflib (https://github.com/MAVENSDC/cdflib)

Issues:
- ...

To-dos:
- ...

Future steps:
- ...

"""

import copy
import numpy as np
from matplotlib.dates import num2date
import sunpy.time


def calc_coupling_newell(by, bz, v):
    ''' 
    Empirical Formula for dFlux/dt - the Newell coupling
    e.g. paragraph 25 in Newell et al. 2010 doi:10.1029/2009JA014805
    IDL ovation: sol_coup.pro - contains 33 coupling functions in total
    input: needs arrays for by, bz, v 
    ''' 
    
    bt = np.sqrt(by**2 + bz**2)
    bztemp = bz
    bztemp[bz == 0] = 0.001
    tc = np.arctan2(by,bztemp) #calculate clock angle (theta_c = t_c)
    neg_tc = bt*np.cos(tc)*bz < 0 #similar to IDL code sol_coup.pro
    tc[neg_tc] = tc[neg_tc] + np.pi
    sintc = np.abs(np.sin(tc/2.))
    ec = (v**(4/3))*(sintc**(8/3))*(bt**(2/3))
    
    return ec

 
def make_kp_from_wind(btot_in, by_in, bz_in, v_in, density_in):
    """
    speed v_in [km/s]
    density [cm-3]
    B in [nT]
    
    Newell et al. 2008
    https://onlinelibrary.wiley.com/resolve/doi?DOI=10.1029/2010SW000604
    see also https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/swe.20053

    The IMF clock angle is defined by thetac = arctan(By/Bz)
    this angle is 0° pointing toward north (+Z), 180° toward south (-Z); 
    its negative for the -Y hemisphere, positive for +Y hemisphere
    thus southward pointing fields have angles abs(thetac)> 90 
    the absolute value for thetac needs to be taken otherwise the fractional power (8/3)
    will lead to imaginary values
    """
 
    thetac= abs(np.arctan2(by_in,bz_in)) #in radians
    merging_rate=v_in**(4/3)*btot_in**(2/3)*(np.sin(thetac/2)**(8/3)) #flux per time
    kp=0.05+2.244*1e-4*(merging_rate)+2.844*1e-6*density_in**0.5*v_in**2 
    
    return kp


def make_aurora_power_from_wind(btot_in, by_in, bz_in, v_in, density_in):
    """
    speed v_in [km/s]
    density [cm-3]
    B in [nT]
    """

    #newell et al. 2008 JGR, doi:10.1029/2007JA012825, page 7 
    thetac= abs(np.arctan2(by_in,bz_in)) #in radians
    merging_rate=v_in**(4/3)*btot_in**(2/3)*(np.sin(thetac/2)**(8/3)) #flux per time
    #unit is in GW
    aurora_power=-4.55+2.229*1e-3*(merging_rate)+1.73*1e-5*density_in**0.5*v_in**2

    return aurora_power


def make_dst_from_wind(btot_in, bx_in, by_in, bz_in, v_in, vx_in, density_in, time_in):
    """
    this makes from synthetic or observed solar wind the Dst index	
    all nans in the input data must be removed prior to function call
    3 models are calculated: Burton et al., OBrien/McPherron, and Temerin/Li
    btot_in IMF total field, in nT, GSE or GSM (they are the same)
    bx_in - the IMF Bx field in nT, GSE or GSM (they are the same)
    by_in - the IMF By field in nT, GSM
    bz_in - the IMF Bz field in nT, GSM
    v_in - the speed in km/s
    vx_in - the solar wind speed x component (GSE is similar to GSM) in km/s
    time_in - the time in matplotlib date format
    """

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
        t1=sunpy.time.julian_day(num2date(time_in[i]))-sunpy.time.julian_day('2015-1-1 00:00')
        
       
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
        dttl=sunpy.time.julian_day(num2date(time_in[i+1]))-sunpy.time.julian_day(num2date(time_in[i]))

       
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
     
    return (dstcalc1, dstcalc2, dst_temerin_li_out)   

