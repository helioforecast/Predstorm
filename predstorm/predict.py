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
from datetime import datetime
from dateutil import tz

import numpy as np
from matplotlib.dates import num2date, date2num
from numba import njit, jit
import astropy.time
import scipy

# Machine learning specific:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer


class DstFeatureExtraction(BaseEstimator, TransformerMixin):
    """
    Takes an array in columns defined by input_keys variable and returns 
    Based off of example from:
    https://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator

    Init Parameters
    ===============
    --> SatData(input_dict, source=None, header=None)
    input_dict : dict(key: dataarray)
        Dict containing the input data in the form of key: data (in array or list)
        Example: {'time': timearray, 'bx': bxarray}. The available keys for data input
        can be accessed in SatData.default_keys.
    header : dict(headerkey: value)
        Dict containing metadata on the data array provided. Useful data headers are
        provided in SatData.empty_header but this can be expanded as needed.
    source : str
        Provide quick-access name of satellite/data type for source.

    Attributes
    ==========
    .look_back
        Number of hours in past to provide in output features.
    .input_keys
        List of keys of array provided to transform() method.
    .feature_keys (only after calling transform())
        List of keys of output features.
    .reduced_features
        Default=True, use reduced features.
    .v_power/.den_power
        Exponents in pressure term.
    .bz_power
        Power of Bz in output feature.
    .m1/.m2/.e1/.e2
        Parameters for OBrien's ring-current term (see calc_ring_current_term())

    Methods
    =======
    .fit()
        Filler function. Returns self.
    .transform(X)
        Transforms input matrix X (SatData.data.T) into ML features.

    Examples
    """
    def __init__(self, input_keys=[], v_power=1, den_power=1, bz_power=1, m1=-4.4, m2=2.4, e1=9.74, e2=4.69, look_back=5, reduced_features=True):
        self.look_back = look_back
        self.input_keys = input_keys
        self.reduced_features = reduced_features
        self.v_power = v_power
        self.den_power = den_power
        self.bz_power = bz_power
        self.m1 = m1
        self.m2 = m2
        self.e1 = e1
        self.e2 = e2


    def fit(self, X, y = None):
        return self


    def transform(self, X):
        # bx and by give no improvement, neither does np.gradient(da['density'], nor V**2)

        # TIME AND SOLAR WIND VARIABLES
        # -----------------------------
        time = X[:,0]
        bx, by, bz = X[:,self.input_keys.index('bx')], X[:,self.input_keys.index('by')], X[:,self.input_keys.index('bz')]
        btot = X[:,self.input_keys.index('btot')]
        speed, density, temp = X[:,self.input_keys.index('speed')], X[:,self.input_keys.index('density')], X[:,self.input_keys.index('temp')]
        sin_DOY, cos_DOY, sin_LT, cos_LT = extract_local_time_variables(time)

        # Combine all:
        if self.reduced_features:
            X = np.vstack((sin_DOY, cos_DOY, time, bx, by, bz)).T
        else:
            X = np.vstack((sin_DOY, cos_DOY, time, speed, density, btot, bx, by, bz)).T

        # OTHER VARIABLES
        # ---------------
        # Pressure term
        pressure = X[:,3]**self.den_power * X[:,2]**self.v_power
        # Gradient in Bz
        dbz = np.gradient(bz)
        # Newell coupling
        ec = calc_newell_coupling(by, bz, speed)
        # Bz to a certain power
        bz_exp = bz**self.bz_power
        # OBrien's ring current term
        deltat = np.asarray([(time[i+1] - time[i])*24. for i in range(len(time)-1)] + [0.])
        deltat[-1] = deltat[-2]
        rc = calc_ring_current_term(deltat, bz, speed, m1=self.m1, m2=self.m2, e1=self.e1, e2=self.e2)
        # Sinphi from Temerin-Li model:
        # Including this term improves overall accuracy but reduces accuracy of large negative values:
        tt, ttt = 2.*np.pi*(time-2449718.5)/365.24, 2.*np.pi*(time-2449718.5)
        cosphi = np.sin(tt+0.078) * np.sin(ttt-tt-1.22) * (9.58589e-2) + np.cos(tt+0.078) * (0.39+0.104528*np.cos(ttt-tt-1.22))
        sinphi = (1. - cosphi*cosphi)**0.5

        # PAST TERMS
        # ----------
        def create_past_dataset(data, look_back=1):
            shifted = []
            # Fill up empty values with mean:
            for i in range(look_back):
                shifted.append(np.full((look_back), np.nanmean(data)))
            # Fill rest of array with past values:
            for i in range(len(data)-look_back):
                a = data[i:(i+look_back)]
                shifted.append(a)
            return np.array(shifted)

        #theta = -(np.arccos(-X[:,5]/X[:,4]) - np.pi) / 2. # infinite?
        #exx = X[:,2] * X[:,4] * np.sin(theta)**7
        # Calculate past terms:
        past_pressure = create_past_dataset(pressure, look_back=self.look_back)
        past_rc = create_past_dataset(rc, look_back=self.look_back)
        past_speed = create_past_dataset(speed, look_back=self.look_back)
        past_bz = create_past_dataset(bz, look_back=self.look_back)
        past_dbz = create_past_dataset(dbz, look_back=self.look_back)
        past_sinphi = create_past_dataset(sinphi, look_back=self.look_back)

        if self.reduced_features:
            past_bz_24h = create_past_dataset(bz, look_back=24)
            past_rc_24h = create_past_dataset(rc, look_back=24)
            features = np.concatenate((X, rc.reshape(-1,1), ec.reshape(-1,1), pressure.reshape(-1,1), 
                                       sinphi.reshape(-1,1), dbz.reshape(-1,1), 
                                       past_bz_24h, past_rc_24h), axis=1)
            self.feature_keys = ['sin_DOY', 'cos_DOY', 'time', 'bx', 'by', 'bz'] + \
                                ['rc', 'ec', 'pressure', 'sinphi', 'dbz'] + \
                                ['bz(t-{})'.format(i) for i in range(1,24+1)] + \
                                ['rc(t-{})'.format(i) for i in range(1,24+1)]
        else:
            features = np.concatenate((X, rc.reshape(-1,1), ec.reshape(-1,1), pressure.reshape(-1,1), 
                                       bz_exp.reshape(-1,1), sinphi.reshape(-1,1), dbz.reshape(-1,1),
                                       past_pressure, past_rc, past_speed, past_bz, past_dbz, past_sinphi), axis=1)
            self.feature_keys = ['sin_DOY', 'cos_DOY', 'time', 'speed', 'density', 'btot', 'bx', 'by', 'bz'] + \
                                ['rc', 'ec', 'pressure', 'bzpower', 'sinphi', 'dbz'] + \
                                ['pressure(t-{})'.format(i) for i in range(1,self.look_back+1)] + \
                                ['rc(t-{})'.format(i) for i in range(1,self.look_back+1)] + \
                                ['speed(t-{})'.format(i) for i in range(1,self.look_back+1)] + \
                                ['bz(t-{})'.format(i) for i in range(1,self.look_back+1)] + \
                                ['dbz(t-{})'.format(i) for i in range(1,self.look_back+1)] + \
                                ['sinphi(t-{})'.format(i) for i in range(1,self.look_back+1)]


        return features


def dst_loss_function(y_true, y_pred):
    rsme_all = math.sqrt(mean_squared_error(y_true, y_pred))
    inds = np.where(y_true < 0.3)   # Lowest 30% (0 to 1 MinMaxScaler)
    rmse_cut = math.sqrt(mean_squared_error(y_true[inds], y_pred[inds]))
    return (rsme_all + rmse_cut)


def calc_dst_burton(time, bz, speed, density):
    """Calculates Dst from solar wind input according to Burton et al. 1975 method.

    Parameters
    ==========
    time : np.array
        Array containing time variables.
    bz : np.array
        Array containing Bz in coordinate system ?.
    speed : np.array
        Array containing solar wind speed.
    density : np.array
        Array containing Bz in coordinate system ?.

    Returns
    =======
    dst_burton : np.array
        Array with calculated values over timesteps time.
    """

    protonmass=1.6726219*1e-27  #kg
    bzneg = copy.deepcopy(bz)
    bzneg[bz > 0] = 0
    pdyn = density*1e6*protonmass*(speed*1e3)**2*1e9  #in nanoPascal
    Ey = speed*abs(bzneg)*1e-3 #now Ey is in mV/m

    dst_burton = np.zeros(len(bz))
    Ec=0.5  
    a=3.6*1e-5
    b=0.2*100 #*100 due to different dynamic pressure einheit in Burton
    c=20  
    d=-1.5/1000 
    lrc=0
    for i in range(len(bz)-1):
        if Ey[i] > Ec:
            F = d*(Ey[i]-Ec) 
        else: F=0
        #Burton 1975 p4208: Dst=Dst0+bP^1/2-c
        # Ring current Dst
        deltat_sec = (time[i+1]-time[i])*86400  #deltat must be in seconds
        rc = lrc + (F-a*lrc)*deltat_sec
        # Dst of ring current and magnetopause currents 
        dst_burton[i+1] = rc + b*np.sqrt(pdyn[i+1]) - c
        lrc = rc

    return dst_burton


def calc_dst_obrien(time, bz, speed, density):
    """Calculates Dst from solar wind input according to OBrien and McPherron 2000 method.

    Parameters
    ==========
    time : np.array
        Array containing time variables.
    bz : np.array
        Array containing Bz in coordinate system ?.
    speed : np.array
        Array containing solar wind speed.
    density : np.array
        Array containing Bz in coordinate system ?.

    Returns
    =======
    dst_burton : np.array
        Array with calculated values over timesteps time.
    """

    protonmass=1.6726219*1e-27  #kg
    bzneg = copy.deepcopy(bz)
    bzneg[bz > 0] = 0
    pdyn = density*1e6*protonmass*(speed*1e3)**2*1e9  #in nanoPascal
    Ey = speed*abs(bzneg)*1e-3; #now Ey is in mV/m

    Ec=0.49
    b=7.26  
    c=11  #nT
    lrc=0
    dst_obrien = np.zeros(len(bz))
    for i in range(len(bz)-1):
        if Ey[i] > Ec:            #Ey in mV m
            Q = -4.4 * (Ey[i]-Ec) 
        else: Q=0
        tau = 2.4 * np.exp(9.74/(4.69 + Ey[i])) #tau in hours
        # Ring current Dst
        deltat_hours=(time[i+1]-time[i])*24 # time should be in hours
        rc = ((Q - lrc/tau))*deltat_hours + lrc
        # Dst of ring current and magnetopause currents 
        dst_obrien[i+1] = rc + b*np.sqrt(pdyn[i+1])-c; 
        lrc = rc

    return dst_obrien


def calc_dst_temerin_li(time, btot, bx, by, bz, speed, speedx, density, version='2002n', linear_t_correction=False):
    """Calculates Dst from solar wind input according to Temerin and Li 2002 method.
    Credits to Xinlin Li LASP Colorado and Mike Temerin.
    Calls _jit_calc_dst_temerin_li. All constants are defined in there.
    Note: vx has to be used with a positive sign throughout the calculation.

    Parameters
    ==========
    time : np.array
        Array containing time variables.
    btot : np.array
        Array containing Btot.
    bx : np.array
        Array containing Bx in coordinate system ?.
    by : np.array
        Array containing By in coordinate system ?.
    bz : np.array
        Array containing Bz in coordinate system ?.
    speed : np.array
        Array containing solar wind speed.
    speedx : np.array
        Array containing solar wind speed in x-direction.
    density : np.array
        Array containing solar wind density.
    version : str (default='2002')
        String determining which model version should be used.

    Returns
    =======
    dst_burton : np.array
        Array with calculated Dst values over timesteps time.
    """
    
    # Arrays
    dst1=np.zeros(len(bz))
    dst2=np.zeros(len(bz))
    dst3=np.zeros(len(bz))
    dst_tl=np.zeros(len(bz))
    
    # Define initial values (needed for convergence, see Temerin and Li 2002 note)
    dst1[0:10]=-15
    dst2[0:10]=-13
    dst3[0:10]=-2

    if version == '2002':
        newparams = False
    else:
        newparams = True

    if version in ['2002', '2002n']:
        # julian_days = [sunpy.time.julian_day(num2date(x)) for x in time]
        julian_days = [astropy.time.Time(num2date(x), format='datetime', scale='utc').jd for x in time]
        return _jit_calc_dst_temerin_li_2002(time, btot, bx, by, bz, speed, speedx, density, dst1, dst2, dst3, dst_tl, julian_days, newparams=newparams)
    elif version == '2006':
        dst1[0:10], dst2[0:10], dst3[0:10] = -10, -5, -10
        ds1995 = time - date2num(datetime(1995,1,1))
        ds2000 = time - date2num(datetime(2000,1,1))

        # YEARLY DRIFT CORRECTION TERM (NOT IN PAPER)
        if linear_t_correction:
            drift_corr = -0.014435865642103548 * ds2000 + 9.57670996872173
        else:
            drift_corr = 0.
        return _jit_calc_dst_temerin_li_2006(ds1995, ds2000, btot, bx, by, bz, speed, speedx, density, dst1, dst2, dst3) + drift_corr

@njit
def _jit_calc_dst_temerin_li_2002(time, btot, bx, by, bz, speed, speedx, density, dst1, dst2, dst3, dst_tl, julian_days, newparams=True):
    """Fast(er) calculation of Dst using jit on Temerin-Li method."""

    #define all constants
    p1, p2, p3 = 0.9, 2.18e-4, 14.7
    
    # these need to be found with a fit for 1-2 years before calculation
    # taken from the TL code:    offset_term_s1 = 6.70       ;formerly named dsto
    #   offset_term_s2 = 0.158       ;formerly hard-coded     2.27 for 1995-1999
    #   offset_term_s3 = -0.94       ;formerly named phasea  -1.11 for 1995-1999
    #   offset_term_s4 = -0.00954    ;formerly hard-coded
    #   offset_term_s5 = 8.159e-6    ;formerly hard-coded

    #found by own offset optimization for 2015
    #s4 and s5 as in the TL 2002 paper are not used due to problems with the time
    if not newparams:
        s1, s2, s3, s4, s5 = -2.788, 1.44, -0.92, -1.054, 8.6e-6
        initdate = 2449718.5
    else:
        s1, s2, s3, s4, s5 = 4.29, 5.94, -3.97, 0., 0.
        initdate = 2457023.5

    a1, a2, a3 = 6.51e-2, 1.37, 8.4e-3 
    a4, a5, a6 = 6.053e-3, 1.21e-3, 1.55e-3 # a5 = 1.12e-3 before. Error?
    tau1, tau2, tau3 = 0.14, 0.18, 9e-2 #days
    b1, b2, b3 = 0.792, 1.326, 1.29e-2  
    c1, c2 = -24.3, 5.2e-2

    yearli=365.24 
    alpha=0.078
    beta=1.22

    for i in np.arange(1,len(bz)-1):

        #t time in days since beginning of 1995   #1 Jan 1995 in Julian days
        #t1=sunpy.time.julian_day(mdates.num2date(time_in[i]))-sunpy.time.julian_day('1995-1-1 00:00')
        # sunpy.time.julian_day('2015-1-1 00:00') = 2457023.5
        t1 = julian_days[i] - initdate
       
        tt = 2*np.pi*t1/yearli
        ttt = 2*np.pi*t1
        cosphi = np.sin(tt+alpha) * np.sin(ttt-tt-beta) * (9.58589e-2) + np.cos(tt+alpha) * (0.39+0.104528*np.cos(ttt-tt-beta))
       
        #equation 1 use phi from equation 2
        sinphi = (1. - cosphi**2.)**0.5
        
        pressureterm = (p1*(btot[i]**2) + density[i] * (p2*(speed[i])**2/(sinphi**2.52) + p3) )**0.5
        
        #2 direct IMF bz term
        directterm = 0.478 * bz[i]*(sinphi**11.0)

        #3 offset term - the last two terms were cut because don't make sense as t1 rises extremely for later years
        offset = s1 + s2 * np.sin(2*np.pi*t1/yearli + s3) + s4*t1 + s5*t1*t1
        #or just set it constant
        #offset[i]=-5
        bt = (by[i]**2 + bz[i]**2)**0.5  
        if bt == 0.: bt = 1e-12  # Escape dividing by zero error in theta_li
        #mistake in 2002 paper - bt is similarly defined as bp (with by bz); but in Temerin and Li's code (dst.pro) bp depends on by and bx
        bp = (by[i]**2 + bx[i]**2)**0.5  
        #contains t1, but in cos and sin 
        dh = bp*np.cos(np.arctan2(bx[i],by[i])+6.10) * (3.59e-2 * np.cos(2*np.pi*t1/yearli + 0.04) - 2.18e-2*np.sin(2*np.pi*t1-1.60))
        #print(i, bx[i], by[i], bz[i])
        theta_li = -(np.arccos(-bz[i]/bt)-np.pi)/2
        exx = 1e-3 * abs(speedx[i]) * bt * np.sin(theta_li)**6.1
        #t1 and dt are in days
        dttl = julian_days[i+1]-julian_days[i]
       
        #4 dst1 
        #find value of dst1(t-tau1) 
        #time is in matplotlib format in days: 
        #im time den index suchen wo time-tau1 am nächsten ist
        #und dann bei dst1 den wert mit dem index nehmen der am nächsten ist, das ist dann dst(t-tau1)
        #wenn index nicht existiert (am anfang) einfach index 0 nehmen
        #check for index where timesi is greater than t minus tau
        indtau1 = np.where(time > (time[i]-tau1))
        dst1tau1 = dst1[indtau1[0][0]]
        dst2tau1 = dst2[indtau1[0][0]]
        th1 = 0.725*(sinphi**-1.46)
        th2 = 1.83*(sinphi**-1.46)
        fe1 = (-4.96e-3) * (1+0.28*dh) * (2*exx+abs(exx-th1) + abs(exx-th2)-th1-th2) * (abs(speedx[i])**1.11) * ((density[i])**0.49) * (sinphi**6.0)
        dst1[i+1] = dst1[i] + (a1*(-dst1[i])**a2 + fe1*(1. + (a3*dst1tau1 + a4*dst2tau1)/(1. - a5*dst1tau1 - a6*dst2tau1))) * dttl
        
        #5 dst2    
        indtau2 = np.where(time > (time[i]-tau2))
        dst1tau2 = dst1[indtau2[0][0]]
        df2 = (-3.85e-8) * (abs(speedx[i])**1.97) * (bt**1.16) * np.sin(theta_li)**5.7 * (density[i])**0.41 * (1+dh)
        fe2 = (2.02e3) * (sinphi**3.13)*df2/(1-df2)
        dst2[i+1] = dst2[i] + (b1*(-dst2[i])**b2 + fe2*(1. + (b3*dst1tau2)/(1. - b3*dst1tau2))) * dttl
        
        #6 dst3  
        indtau3 = np.where(time > (time[i]-tau3))
        dst3tau3 = dst3[indtau3[0][0]]
        df3 = -4.75e-6 * (abs(speedx[i])**1.22) * (bt**1.11) * np.sin(theta_li)**5.5 * (density[i])**0.24 * (1+dh)
        fe3 = 3.45e3 * (sinphi**0.9) * df3/(1.-df3)
        dst3[i+1] = dst3[i] + (c1*dst3[i] + fe3*(1. + (c2*dst3tau3)/(1. - c2*dst3tau3))) * dttl
        
        #The dst1, dst2, dst3, (pressure term), (direct IMF bz term), and (offset terms) 
        # are added (after interpolations) with time delays of 7.1, 21.0, 43.4, 2.0, 23.1 and 7.1 min, 
        # respectively, for comparison with the ‘‘Kyoto Dst.’’ 
        dst_tl[i] = dst1[i] + dst2[i] + dst3[i] + pressureterm + directterm + offset

    return dst_tl

@njit(parallel=True)#"void(f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])"
def _jit_calc_dst_temerin_li_2006(t1, t2, btot, bx, by, bz, speed, speedx, density, dst1, dst2, dst3):
    """Fast(er) calculation of Dst using jit on Temerin-Li method."""
    
    fy = 2.*np.pi/365.24
    sun1 = np.sin(np.pi * 10.27 / 180.)
    sun2 = np.sin(np.pi * 10.27 / 180.) * np.cos(np.pi * 23.5 / 180.)
    alpha = 0.0449

    # SOLAR WIND VALUES
    # -----------------
    speedx = np.abs(speedx)
    bt = np.sqrt(by**2 + bz**2)
    bt[bt < 0.0001] = 1e-4      # Correction from dst.pro, escaping zero-division error
    bp = np.sqrt(by**2 + bx**2)
    btot = np.sqrt(bx**2. + by**2. + bz**2.)

    theta = -(np.arccos(-bz/bt) - np.pi) / 2.
    ang = np.arctan2(bx, by)

    exx = speedx * bt**0.993 * np.sin(theta)**7.29
    # exx2 = density**0.493 * speedx**2.955 * bt**1.105 * np.sin(theta)**5.24 # paper
    # exx3 = density**0.397 * speedx**0.576 * bt**1.413 * np.sin(theta)**8.56 # paper
    exx2 = speedx * bt**1.105 * np.sin(theta)**5.24 # code
    exx3 = speedx * bt**1.413 * np.sin(theta)**8.56 # code

    # TIME VALUES
    # -----------
    dh = 0.0435 * np.cos(fy*t1 + 0.1680) - 0.0208 * np.sin(2*np.pi*t1 - 1.589)
    itest = 40
    it1 = itest - np.where(t1 > (t1[itest] - 0.0486))[0][0]
    it2 = itest - np.where(t1 > (t1[itest] - 0.181))[0][0]
    it3 = itest - np.where(t1 > (t1[itest] - 0.271))[0][0]
    it4 = itest - np.where(t1 > (t1[itest] - 0.0625))[0][0]
    it5 = itest - np.where(t1 > (t1[itest] - 0.104))[0][0]
    it6 = itest - np.where(t1 > (t1[itest] - 0.0278))[0][0]
    it7 = itest - np.where(t1 > (t1[itest] - 0.139))[0][0]
    idst1t1 = itest - np.where(t1 > (t1[itest] - 0.132))[0][0]
    idst2t1 = itest - np.where(t1 > (t1[itest] - 0.0903))[0][0]
    idst1t2 = itest - np.where(t1 > (t1[itest] - 0.264))[0][0]

    # FUNCTION TERMS
    # --------------
    tt = t1*fy
    cosphi = sun2 * np.sin(tt + alpha) * np.sin(2.*np.pi*t1 - tt - 1.632) + \
                np.cos(tt + alpha) * (0.39 + sun1*np.cos(2*np.pi*t1 - tt - 1.632))
    cosphi5 = sun2 * np.sin(tt + alpha) * np.sin(2.*np.pi*t1 - tt + 0.27) + \
                np.cos(tt + alpha) * (0.39 + sun1*np.cos(2*np.pi*t1 - tt + 0.27))
    cosphi6 = sun2 * np.sin(tt + alpha) * np.sin(2.*np.pi*t1 - tt - 0.21) + \
                np.cos(tt + alpha) * (0.39 + sun1*np.cos(2*np.pi*t1 - tt - 0.21))
    cosphi7 = sun2 * np.sin(tt + alpha) * np.sin(2.*np.pi*t1 - tt - 0.79) + \
                np.cos(tt + alpha) * (0.39 + sun1*np.cos(2*np.pi*t1 - tt - 0.79))
    cosphi8 = sun2 * np.sin(tt + alpha) * np.sin(2.*np.pi*t1 - tt - 2.81) + \
                np.cos(tt + alpha) * (0.39 + sun1*np.cos(2*np.pi*t1 - tt - 2.81))

    sin_phi_factor = 0.95097
    tst3 = ( np.sqrt(1. - cosphi**2.) / sin_phi_factor )**-0.13
    tst4 = ( np.sqrt(1. - cosphi**2.) / sin_phi_factor )**6.54
    tst5 = ( np.sqrt(1. - cosphi5**2.) / sin_phi_factor )**5.13
    tst6 = ( np.sqrt(1. - cosphi6**2.) / sin_phi_factor )**-2.44
    tst7 = ( np.sqrt(1. - cosphi7**2.) / sin_phi_factor )**2.84
    tst8 = ( np.sqrt(1. - cosphi8**2.) / sin_phi_factor )**2.49

    fe1 = -1.703e-6 * (1. + erf(-0.09*bp * np.cos(ang - 0.015) * dh)) * \
                tst3 * ((exx - 1231.2/tst4 + np.abs(exx - 1231.2/tst4)) + \
                (exx - 3942./tst4 + np.abs(exx - 3942./tst4))) * speedx**1.307 * density**0.548
    # fe2 = 5.172e-8 * exx2 * (1. + erf(0.418*bp * np.cos(ang - 0.015) * dh) )    # paper
    # fe3 =  -0.0412 * exx3 * (1. + erf(1.721*bp * np.cos(ang - 0.015) * dh) )    # paper
    fe2 = -5.172e-8 * exx2 * (1. + erf(0.418*bp * np.cos(ang - 0.015) * dh) ) * speedx**1.955 * density**0.493   # code
    fe3 =  -0.0412 * exx3 * (1. + erf(1.721*bp * np.cos(ang - 0.015) * dh) ) * speedx**-0.424 * density**0.397   # code

    df2 = 1440. * tst7 * fe2/(-fe2 + 922.1)
    df3 = 272.9 * tst8 * fe3/(-fe3 + 60.5)

    # PRESSURE TERM
    # -------------
    pressureterm = ( 0.330*btot**2 * (1. + 0.100*density) + \
                (1.621e-4 * tst6 * speed**2 + 18.70)*density )**0.5

    # DIRECT BZ TERM
    # --------------
    directbzterm = 0.574 * tst5 * bz

    # OFFSET TERM
    # -----------
    offsetterm = 19.35 + 0.158*np.sin(fy*t2 - 0.94) + 0.01265*t2 - 2.224e-11*t2**2.

    # INITIAL DST LOOP
    # ----------------
    for i in range(0,40):

        dt = (t1[i+1] - t1[i])#/6.   # TODO remove this changed from hr to 10mins

        # Code
        dst1[i+1] = dst1[i] + (0.005041 * (-dst1[i])**2.017 + fe1[i]) * dt
        dst2[i+1] = dst2[i] + (0.00955 * (-dst2[i])**2.269 + df2[i] * (1. + 0.01482*dst1[i] / (1. - 0.01482*dst1[i]) )) * dt
        dst3[i+1] = dst3[i] + (-5.10*dst3[i] + df3[i]) * dt

    # MAIN DST LOOP
    # -------------
    for i in range(40,len(bz)-1):

        # DST TERMS
        # ---------
        bzt1 = bz[i-it1]
        bzt2 = bz[i-it2]
        bzt3 = bz[i-it3]
        dst1t1 = dst1[i-idst1t1]
        dst2t1 = dst2[i-idst2t1]
        dst1[i+1] = dst1[i] + (5.041e-3 * (-dst1[i])**2.017 * \
                    (1. + erf(-0.010*bz[i])) + fe1[i] * \
                    (1. + erf(-0.0094*bzt1 - 0.0118*bzt2 + 0.0138*bzt3)) * \
                    np.exp(0.00313*dst1t1 + 0.01872*dst2t1)) * dt

        bzt4 = bz[i-it4]
        bzt5 = bz[i-it5]
        dst1t2 = dst1[i-idst1t2]
        dst2[i+1] = dst2[i] + (0.00955 * (-dst2[i])**2.017 * \
                    (1. + erf(-0.014*bz[i])) + df2[i] * \
                    (1 + erf(-0.0656*bzt4 + 0.0627*bzt5)) * \
                    np.exp(0.01482*dst1t2)) * dt

        bzt6 = bz[i-it6]
        bzt7 = bz[i-it7]
        dst3[i+1] = dst3[i] + (5.10 * (-dst3[i])**0.952 * \
                    (1. + erf(-0.027*bz[i])) + df3[i] * \
                    (1. + erf(-0.0471*bzt6 + 0.0184*bzt7))) * dt

    # ANNUAL VARIATIONS
    # -----------------
    dst1_ = dst1 * (1. + 0.0807*np.sin(t1*fy + 1.886))
    dst2_ = dst2 * (1. + 0.0251*np.sin(t1*fy + 3.18))
    dst3_ = dst3 * (1. + 0.0901*np.sin(t1*fy + 5.31)) * (1.-0.00007*dst1_)
    # directbzterm_ = directbzterm * (1. + 0.293 * np.sin(t1[i]*fy + 3.19)) * (1. + 0.0034*dst1_) # paper
    directbzterm_ = directbzterm * (1. + 0.293 * np.sin(t1[i]*fy + 3.19))   # code
    pressureterm_ = pressureterm * (1. + 0.0986*np.sin(t1[i]*fy - 1.383)) * (1. + 0.00184*dst1_)

    # FINAL DST
    # ---------
    dst_tl = dst1_ + dst2_ + dst3_ + pressureterm_ + directbzterm_ + offsetterm

    return dst_tl

@njit
def erf(x):
    # adjusted from https://stackoverflow.com/questions/457408/is-there-an-easily-available-implementation-of-erf-for-python
    # save the sign of x
    sign = np.sign(x)
    x = np.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1) * t * np.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

@njit
def calc_newell_coupling(by, bz, v):
    """
    Empirical Formula for dFlux/dt - the Newell coupling
    e.g. paragraph 25 in Newell et al. 2010 doi:10.1029/2009JA014805
    IDL ovation: sol_coup.pro - contains 33 coupling functions in total
    input: needs arrays for by, bz, v 
    """
    
    bt = np.sqrt(by**2 + bz**2)
    bztemp = bz
    bztemp[bz == 0] = 0.001
    tc = np.arctan2(by,bztemp) #calculate clock angle (theta_c = t_c)
    neg_tc = bt*np.cos(tc)*bz < 0 #similar to IDL code sol_coup.pro
    tc[neg_tc] = tc[neg_tc] + np.pi
    sintc = np.abs(np.sin(tc/2.))
    ec = (v**(4/3))*(sintc**(8/3))*(bt**(2/3))
    
    return ec


def calc_ring_current_term(deltat, bz, speed, m1=-4.4, m2=2.4, e1=9.74, e2=4.69):
    """Calculates a term describing the ring current from the Burton Dst
    prediction method.

    Parameters
    ==========
    deltat : np.array
        Array of timestep deltas, so that deltat[i] = time[i+1]-time[i].
    bz : np.array
        Array of magnetic field in z-direction.
    speed : np.array
        Array of solar wind speed.
    m1, m2, e1, e2 : float
        Constants in equation. Default values taken from Burton paper.

    Returns
    =======
    rc : np.array
        Array containing ring current term.
    """

    bzneg = copy.deepcopy(bz)
    bzneg[bzneg > 0] = 0
    Ey = speed * abs(bzneg)*1e-3 #now Ey is in mV/m
    rc = np.zeros(len(bzneg))
    Ec = 0.5  
    lrc = 0
    for i in range(len(bzneg)-1):
        if Ey[i] > Ec:            #Ey in mV m
            Q = m1 * (Ey[i] - Ec) 
        else: 
            Q=0
        tau = m2 * np.exp(e1/(e2 + Ey[i])) #tau in hours
        # Ring current Dst
        #deltat_hours = (time[i+1] - time[i])*24 # time should be in hours
        rc[i+1] = (Q - lrc/tau) * deltat[i] + lrc
        lrc = rc[i+1]

    return rc


def extract_local_time_variables(time):
    """Takes the UTC time in numpy date format and 
    returns local time and day of year variables, cos/sin.

    Parameters:
    -----------
    time : np.array
        Contains timestamps in numpy format.

    Returns:
    --------
    sin_DOY, cos_DOY, sin_LT, cos_LT : np.arrays
        Sine and cosine of day-of-yeat and local-time.
    """

    dtime = num2date(time)
    utczone = tz.gettz('UTC')
    cetzone = tz.gettz('CET')
    # Original data is in UTC:
    dtimeUTC = [dt.replace(tzinfo=utczone) for dt in dtime]
    # Correct to local time zone (CET) for local time:
    dtimeCET = [dt.astimezone(cetzone) for dt in dtime]
    dtlocaltime = np.array([(dt.hour + dt.minute/60. + dt.second/3600.) for dt in dtimeCET])
    dtdayofyear = np.array([dt.timetuple().tm_yday for dt in dtimeCET])
    dtdayofyear = np.array([dt.timetuple().tm_yday for dt in dtimeCET]) + dtlocaltime
    
    sin_DOY, cos_DOY = np.sin(2.*np.pi*dtdayofyear/365.), np.sin(2.*np.pi*dtdayofyear/365.)
    sin_LT, cos_LT = np.sin(2.*np.pi*dtlocaltime/24.), np.sin(2.*np.pi*dtlocaltime/24.)

    return sin_DOY, cos_DOY, sin_LT, cos_LT


def get_scores(dst_real, dst_pred, tarray, source='L1', printtext=True):
    """Returns some scoring values for the real values and forecast, that's all.

    Parameters
    ==========
    dst_pred : np.array
        Predicted Dst values.
    dst_real : np.array
        Real (or preliminary) Dst values.
    tarray : np.array
        Array of time values (matplotlib date2num) for dst.
    source : str (default='L1')
        Location of data source. Only used with printtext.
    printtext : bool (default=True)

    Returns
    =======
    stat_dict : dict
        Dictionary containing values under the keys ['diff_mean', 'diff_std',
        'xcorr_offset', 'xcorr_max', 'ppmc', 'mae']
        """

    dst_diff = dst_real - dst_pred
    dst_diff_mean = np.nanmean(dst_diff)
    dst_diff_std = np.nanstd(dst_diff)
    t = np.linspace(0.0, tarray[-1] - tarray[0], len(dst_real), endpoint=False)
    dt = np.linspace(-t[-1], t[-1], 2*len(dst_real)-1)
    xcorr = scipy.signal.correlate(dst_real, dst_pred)
    ppmc = np.corrcoef(dst_real, dst_pred)[0][1]
    mae = np.sum(np.abs(dst_diff)) / len(dst_diff)
    if printtext:
        print("DATA FROM {}".format(source))
        print("----------"+'-'*len(source))
        print('Dst diff mean +/- std: {:.1f} +/- {:.1f}'.format(dst_diff_mean, dst_diff_std))
        print("")
        print('Dst obs  mean +/- std: {:.1f} +/- {:.1f}'.format(np.nanmean(dst_real), np.nanstd(dst_real)))
        print('Dst pred mean +/- std: {:.1f} +/- {:.1f}'.format(np.nanmean(dst_pred), np.nanstd(dst_pred)))
        print('Dst obs  min / max: {:.1f} / {:.1f}'.format(np.nanmin(dst_real), np.nanmax(dst_real)))
        print('Dst pred min / max: {:.1f} / {:.1f}'.format(np.nanmin(dst_pred), np.nanmax(dst_pred)))
        print()
        print("Pearson correlation: {:.2f} ".format(ppmc))
        print("Cross-correlation:   {:.1f} hours".format(24.*dt[xcorr.argmax()]))
        print("Mean absolute error: {:.2f} nT".format(mae))
        print()

    stat_dict = {}
    stat_dict['diff_mean'] = dst_diff_mean
    stat_dict['diff_std'] = dst_diff_std
    stat_dict['xcorr_offset'] = 24.*dt[xcorr.argmax()]
    stat_dict['xcorr_max'] = np.nanmax(xcorr)
    stat_dict['ppmc'] = ppmc
    stat_dict['mae'] = mae

    return stat_dict

 
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


