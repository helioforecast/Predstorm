#!/usr/bin/env python
"""
This is the data handling module for the predstorm package, containing the main
data class SatData and all relevant data handling functions and procedures.

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

import os
import sys  
import cdflib
import copy
import heliopy.spice as hspice
import logging
import numpy as np
import pdb
import pickle
import scipy
import scipy.io
import sunpy.time
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from glob import iglob
import json
import urllib
try:
    from netCDF4 import Dataset
except:
    pass

from . import spice
from .predict import make_kp_from_wind, make_dst_from_wind
from .predict import make_aurora_power_from_wind, calc_newell_coupling
from .predict import calc_dst_burton, calc_dst_obrien, calc_dst_temerin_li

logger = logging.getLogger(__name__)

# =======================================================================================
# -------------------------------- I. CLASSES ------------------------------------------
# =======================================================================================

class SatData():
    """Data object containing satellite data.

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
    .data : np.ndarray
        Array containing measurements/indices. Best accessed using SatData[key].
    .position : np.ndarray
        Array containing position data for satellite.
    .h : dict
        Dict of metadata as defined by input header.
    .vars : list
        List of variables stored in SatData.data.
    .source : str
        Data source name.

    Methods
    =======
    .convert_GSE_to_GSM()
        Coordinate conversion.
    .convert_RTN_to_GSE()
        Coordinate conversion.
    .cut(starttime=None, endtime=None)
        Cuts data to within timerange and returns.
    .get_position(timestamp)
        Returns position of spacecraft at time.
    .get_newell_coupling()
        Calculates Newell coupling indice for data.
    .interp_nans(keys=None)
        Linearly interpolates over nans in data.
    .interp_to_time()
        Linearly interpolates over nans.
    .load_position_data(position_data_file)
        Loads position data from file.
    .make_aurora_power_prediction()
        Calculates aurora power.
    .make_dst_prediction()
        Makes prediction of Dst from data.
    .make_kp_prediction()
        Prediction of kp.
    .make_hourly_data()
        Takes minute resolution data and interpolates to hourly data points.
    .shift_time_to_L1()
        Shifts time to L1 from satellite ahead in sw rotation.

    Examples
    ========
    """

    default_keys = ['time',
                    'speed', 'density', 'temp', 'pdyn',
                    'bx', 'by', 'bz', 'btot',
                    'br', 'bt', 'bn',
                    'dst', 'kp', 'aurora', 'ec']

    empty_header = {'DataSource': '',
                    'SourceURL' : '',
                    'SamplingRate': None,
                    'CoordinateSystem': '',
                    'FileVersion': {},
                    'Instruments': []
                    }

    def __init__(self, input_dict, source=None, header=None):
        """Create new instance of class."""

        # Check input data
        for k in input_dict.keys():
            if not k in SatData.default_keys: 
                raise NotImplementedError("Key {} not implemented in SatData class!".format(k))
        if 'time' not in input_dict.keys():
            raise Exception("Time variable is required for SatData object!")
        dt = [x for x in SatData.default_keys if x in input_dict.keys()]
        #data = [input_dict[x[0]] for x in dt]
        data = [input_dict[x] if x in dt else np.zeros(len(input_dict['time'])) for x in SatData.default_keys]
        # Cast this to be our class type
        self.data = np.asarray(data)
        # Add new attributes to the created instance
        self.source = source
        if header == None:               # Inititalise empty header
            self.h = copy.copy(SatData.empty_header)
        else:
            self.h = header
        self.pos = None
        self.vars = dt
        self.vars.remove('time')


    # -----------------------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------------------

    def __getitem__(self, var):
        if isinstance(var, str):
            if var in self.vars+['time']:
                return self.data[SatData.default_keys.index(var)]
            else:
                raise Exception("SatData object does not contain data under the key '{}'!".format(var))
        return self.data[:,var]


    def __setitem__(self, var, value):
        if isinstance(var, str):
            if var in self.vars:
                self.data[SatData.default_keys.index(var)] = value
            elif var in SatData.default_keys and var not in self.vars:
                self.data[SatData.default_keys.index(var)] = value
                self.vars.append(var)
            else:
                raise Exception("SatData object does not contain the key '{}'!".format(var))
        else:
            raise ValueError("Cannot interpret {} as index for __setitem__!".format(var))


    def __len__(self):
        return len(self.data[0])


    def __str__(self):
        """Print string describing object."""

        ostr = "Length of data:\t\t{}\n".format(len(self))
        ostr += "Keys in data:\t\t{}\n".format(self.vars)
        ostr += "First data point:\t{}\n".format(mdates.num2date(self['time'][0]))
        ostr += "Last data point:\t{}\n".format(mdates.num2date(self['time'][-1]))
        ostr += "\n"
        ostr += "Header information:\n"
        for j in self.h:
            if self.h[j] != None: ostr += "    {:>25}:\t{}\n".format(j, self.h[j])
        ostr += "\n"
        ostr += "Variable statistics:\n"
        ostr += "{:>12}{:>12}{:>12}\n".format('VAR', 'MEAN', 'STD')
        for k in self.vars:
            ostr += "{:>12}{:>12.2f}{:>12.2f}\n".format(k, np.nanmean(self[k]), np.nanstd(self[k]))

        return ostr


    # -----------------------------------------------------------------------------------
    # Position data handling and coordinate conversions
    # -----------------------------------------------------------------------------------

    def convert_GSE_to_GSM(self):
        """GSE to GSM conversion
        main issue: need to get angle psigsm after Hapgood 1992/1997, section 4.3
        for debugging pdb.set_trace()
        for testing OMNI DATA use
        [bxc,byc,bzc]=convert_GSE_to_GSM(bx[90000:90000+20],by[90000:90000+20],bz[90000:90000+20],times1[90000:90000+20])

        CAUTION: Overwrites original data.
        """
     
        mjd=np.zeros(len(self['time']))

        #output variables
        bxgsm=np.zeros(len(self['time']))
        bygsm=np.zeros(len(self['time']))
        bzgsm=np.zeros(len(self['time']))

        for i in np.arange(0,len(self['time'])):
            #get all dates right
            jd=sunpy.time.julian_day(sunpy.time.break_time(mdates.num2date(self['time'][i])))
            mjd[i]=float(int(jd-2400000.5)) #use modified julian date    
            T00=(mjd[i]-51544.5)/36525.0
            dobj=mdates.num2date(self['time'][i])
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
            GSE=np.matrix([[self['bx'][i]],[self['by'][i]],[self['bz'][i]]]) 
            GSM=np.dot(T3,GSE)   #equation 6 in Hapgood
            bxgsm[i]=GSM.item(0)
            bygsm[i]=GSM.item(1)
            bzgsm[i]=GSM.item(2)
        #-------------- loop over

        self['bx'] = bxgsm
        self['by'] = bygsm
        self['bz'] = bzgsm
        self.h['CoordinateSystem'].replace('GSE', 'GSM')

        return self


    def convert_RTN_to_GSE(self, pos_obj=[], pos_tnum=[]):
        """Converts RTN to GSE coordinates.

        function call [dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta)

        pdb.set_trace()  for debugging
        convert STEREO A magnetic field from RTN to GSE
        for prediction of structures seen at STEREO-A later at Earth L1
        so we do not include a rotation of the field to the Earth position
        """

        #output variables
        heeq_bx=np.zeros(len(self['time']))
        heeq_by=np.zeros(len(self['time']))
        heeq_bz=np.zeros(len(self['time']))
        
        bxgse=np.zeros(len(self['time']))
        bygse=np.zeros(len(self['time']))
        bzgse=np.zeros(len(self['time']))
        
        ########## first RTN to HEEQ 

        # NEW METHOD:
        if len(pos_obj) == 0 and len(pos_tnum) == 0:
            if self.pos == None:
                raise Exception("Load position data (SatData.load_position_data()) before calling convert_RTN_to_GSE()!")
            xyz_times = self.pos.positions.T
            # Which is equivalent to:
            # times = [mdates.num2date(t).replace(tzinfo=None) for t in self['time']]
            # xyz_times = spice.get_satellite_position('STEREO-A', times, refframe='HEEQ', rlonlat=True)
            AU = 149597870.700 #in km
            xyz_times = xyz_times/AU
        
        #go through all data points
        for i in np.arange(0,len(self['time'])):
            #make RTN vectors, HEEQ vectors, and project 
            #r, long, lat in HEEQ to x y z
            # OLD METHOD:
            if len(pos_obj) > 0 and len(pos_tnum) > 0:
                time_ind_pos=(np.where(pos_tnum < self['time'][i])[-1][-1])
                [xa,ya,za]=sphere2cart(pos_obj[0][time_ind_pos],pos_obj[1][time_ind_pos],pos_obj[2][time_ind_pos])
            # NEW METHOD:
            else:
                xa, ya, za = sphere2cart(xyz_times[i][0], xyz_times[i][1], xyz_times[i][2])

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
            heeq_bx[i]=np.dot(np.dot(self['br'][i],Xrtn)+np.dot(self['bt'][i],Yrtn)+np.dot(self['bn'][i],Zrtn),X_heeq)
            heeq_by[i]=np.dot(np.dot(self['br'][i],Xrtn)+np.dot(self['bt'][i],Yrtn)+np.dot(self['bn'][i],Zrtn),Y_heeq)
            heeq_bz[i]=np.dot(np.dot(self['br'][i],Xrtn)+np.dot(self['bt'][i],Yrtn)+np.dot(self['bn'][i],Zrtn),Z_heeq)

        #get modified Julian Date for conversion as in Hapgood 1992
        jd=np.zeros(len(self['time']))
        mjd=np.zeros(len(self['time']))
        
        #then HEEQ to GSM
        #-------------- loop go through each date
        for i in np.arange(0,len(self['time'])):
            sunpy_time=sunpy.time.break_time(mdates.num2date(self['time'][i]))
            jd[i]=sunpy.time.julian_day(sunpy_time)
            mjd[i]=float(int(jd[i]-2400000.5)) #use modified julian date    
            #then lambda_sun
            T00=(mjd[i]-51544.5)/36525.0
            dobj=mdates.num2date(self['time'][i])
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
        self['bx'] = bxgse
        self['by'] = bygse
        self['bz'] = bzgse
        self.h['CoordinateSystem'] += '-GSE'

        return self


    def get_position(self, timestamp):
        """Returns position of satellite at given timestamp. Coordinates
        are provided in (r,lon,lat) format. Change rlonlat to False to get
        (x,y,z) coordinates. Uses function predstorm.spice.get_satellite_position.

        Parameters
        ==========
        timestamp : datetime.datetime object / list of dt objs
            Times of positions to return.

        Returns
        =======
        position : array(x,y,z), list of arrays for multiple timestamps
            Position of satellite in x,y,z or r,lon,lat.
        """

        if self.pos == None:
            raise Exception("Load position data (SatData.load_position_data()) before calling get_position()!")

        tind = np.where(mdates.date2num(timestamp) < self.data[0])[0][0]
        return self.pos[tind]


    def load_positions(self, posfile, rlonlat=True, heliopy=True):
        """Loads data on satellite position into data object. Data is loaded from a
        pickled heliopy.spice.Trajectory object.oaded into local heliopy file.

        Parameters
        ==========
        posfile : str
            Path to where file will be stored.
        rlonlat : bool (default=True)
            If True, returns coordinates in (r, lon, lat) format, not (x,y,z).
        heliopy : bool (default=True)
            If True, heliopy object is loaded.

        Returns
        =======
        self with new data in self.pos
        """

        logger.info("load_positions: Loading position data into {} data".format(self.source))
        refframe = os.path.split(posfile)[-1].split('_')[-2]
        Positions = get_position_data(posfile, self['time'], rlonlat=rlonlat)
        self.pos = Positions

        return self

    # -----------------------------------------------------------------------------------
    # Object data handling
    # -----------------------------------------------------------------------------------

    def cut(self, starttime=None, endtime=None):
        """Cuts array down to range defined by starttime and endtime. One limit
        can be provided or both.

        Parameters
        ==========
        starttime : datetime.datetime object
            Start time (>=) of new array.
        endtime : datetime.datetime object
            End time (<) of new array.

        Returns
        =======
        self : obj within new time range
        """

        if starttime != None and endtime == None:
            self.data = self.data[:,np.where(self.data[0] >= mdates.date2num(starttime))[0]]
            if self.pos != None:
                self.pos = self.pos[:,np.where(self.data[0] >= mdates.date2num(starttime))[0]]
        elif starttime == None and endtime != None:
            self.data = self.data[:,np.where(self.data[0] < mdates.date2num(endtime))[0]]
            if self.pos != None:
                self.pos = self.pos[:,np.where(self.data[0] < mdates.date2num(endtime))[0]]
        elif starttime != None and endtime != None:
            self.data = self.data[:,np.where((self.data[0] >= mdates.date2num(starttime)) & (self.data[0] < mdates.date2num(endtime)))[0]]
            if self.pos != None:
                self.pos = self.pos[:,np.where((self.data[0] >= mdates.date2num(starttime)) & (self.data[0] < mdates.date2num(endtime)))[0]]
        return self


    def make_hourly_data(self):
        """Takes data with minute resolution and interpolates to hour.

        Parameters
        ==========
        None

        Returns
        =======
        Data_h : new SatData obj
            New array with hourly interpolated data. Header is copied from original.
        """

        # Round to nearest hour
        stime = self['time'][0] - self['time'][0]%(1./24.)
        # Create new time array
        time_h = np.array(stime + np.arange(0, len(self['time'])/60.) * (1./24.))
        data_dict = {'time': time_h}
        for k in self.vars:
            na = np.interp(time_h, self['time'], self[k])
            data_dict[k] = na

        # Create new data opject:
        Data_h = SatData(data_dict, header=copy.copy(self.h), source=copy.copy(self.source))
        Data_h.h['SamplingRate'] = 1./24.

        return Data_h


    def interp_nans(self, keys=None):
        """Linearly interpolates over nans in array.

        Parameters
        ==========
        keys : list (default=None)
            Provide list of keys (str) to be interpolated over, otherwise all.
        """

        logger.info("interp_nans: Interpolating nans in {} data".format(self.source))
        if keys == None:
            keys = self.vars
        for k in keys:
            inds = np.isnan(self[k])
            if len(inds) == 0:
                return self
            self[k][inds] = np.interp(inds.nonzero()[0], (~inds).nonzero()[0], self[k][~inds])
        return self


    def interp_to_time(self, tarray, keys=None):
        """Linearly interpolates over nans in array.

        Parameters
        ==========
        tarray : np.ndarray
            Array containing new timesteps in number format.
        keys : list (default=None)
            Provide list of keys (str) to be interpolated over, otherwise all.
        """

        resolution = tarray[1] - tarray[0]
        # Round to nearest timestep
        stime = self['time'][0] - self['time'][0] % resolution
        # Create new time array
        data_dict = {'time': tarray}
        for k in self.vars:
            na = np.interp(tarray, self['time'], self[k])
            data_dict[k] = na

        # Create new data opject:
        newData = SatData(data_dict, header=copy.copy(self.h), source=copy.copy(self.source))
        newData.h['SamplingRate'] = resolution

        return newData


    def shift_time_to_L1(self, sun_syn=26.24):
        """Shifts the time variable to roughly correspond to solar wind at L1."""

        lag_l1, lag_r = get_time_lag_wrt_earth(satname=self.source,
            timestamp=mdates.num2date(self['time'][-1]),
            v_mean=np.nanmean(self['speed']), sun_syn=sun_syn)
        logger.info("shift_time_to_L1: Shifting time by {:.2f} hours".format((lag_l1 + lag_r)*24.))
        self.data[0] = self.data[0] + lag_l1 + lag_r
        return self


    # -----------------------------------------------------------------------------------
    # Index calculations and predictions
    # -----------------------------------------------------------------------------------

    def get_newell_coupling(self):
        """
        Empirical Formula for dFlux/dt - the Newell coupling
        e.g. paragraph 25 in Newell et al. 2010 doi:10.1029/2009JA014805
        IDL ovation: sol_coup.pro - contains 33 coupling functions in total
        input: needs arrays for by, bz, v
        """

        ec = calc_newell_coupling(self['by'], self['bz'], self['speed'])
    
        ecData = SatData({'time': self['time'], 'ec': ec})
        ecData.h['DataSource'] = "Newell coupling parameter from {} data".format(self.source)
        ecData.h['SamplingRate'] = self.h['SamplingRate']

        return ecData


    def make_aurora_power_prediction(self):
        """Makes prediction with data in array.

        Parameters
        ==========
        self

        Returns
        =======
        auroraData : new SatData obj
            New object containing predicted Dst data.
        """

        aurora_power = np.round(make_aurora_power_from_wind(self['btot'], self['by'], self['bz'], self['speed'], self['density']), 2)
        #make sure that no values are < 0
        aurora_power[np.where(aurora_power < 0)]=0.0

        auroraData = SatData({'time': self['time'], 'aurora': aurora_power})
        auroraData.h['DataSource'] = "Auroral power prediction from {} data".format(self.source)
        auroraData.h['SamplingRate'] = 1./24.

        return auroraData


    def make_dst_prediction(self, method='temerin_li'):
        """Makes prediction with data in array.

        Parameters
        ==========
        method : str
            Options = ['burton', 'obrien', 'temerin_li']

        Returns
        =======
        dstData : new SatData obj
            New object containing predicted Dst data.
        """

        if method.lower() == 'temerin_li':
            dst_pred = calc_dst_temerin_li(self['time'], self['btot'], self['bx'], self['by'], self['bz'], self['speed'], self['speed'], self['density'])
        elif method.lower() == 'obrien':
            dst_pred = calc_dst_obrien(self['time'], self['bz'], self['speed'], self['density'])
        elif method.lower() == 'burton':
            dst_pred = calc_dst_burton(self['time'], self['bz'], self['speed'], self['density'])

        dstData = SatData({'time': self['time'], 'dst': dst_pred})
        dstData.h['DataSource'] = "Dst prediction from {} data using {} method".format(self.source, method)
        dstData.h['SamplingRate'] = 1./24.

        return dstData


    def make_kp_prediction(self):
        """Makes prediction with data in array.

        Parameters
        ==========
        self

        Returns
        =======
        kpData : new SatData obj
            New object containing predicted Dst data.
        """

        kp_pred = np.round(make_kp_from_wind(self['btot'], self['by'], self['bz'], self['speed'], self['density']), 1)

        kpData = SatData({'time': self['time'], 'kp': kp_pred})
        kpData.h['DataSource'] = "Kp prediction from {} data".format(self.source)
        kpData.h['SamplingRate'] = 1./24.

        return kpData


class PositionData():
    """Data object containing satellite position data.

    Init Parameters
    ===============
    --> PositionData(input_dict, source=None, header=None)
    posdata : list(x,y,z) or list(r,lon,lat)
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
    .positions : np.ndarray
        Array containing position information. Best accessed using SatData[key].
    .h : dict
        Dict of metadata as defined by input header.

    Methods
    =======
    ...

    Examples
    ========
    """

    empty_header = {'ReferenceFrame': '',
                    'CoordinateSystem': '',
                    'Units': '',
                    'Object': '',
                    }


    def __init__(self, posdata, postype, header=None):
        """Create new instance of class."""

        if not postype.lower() in ['xyz', 'rlonlat']:
            raise Exception("PositionData __init__: postype must be either 'xyz' or 'rlonlat'!")
        self.positions = np.asarray(posdata)
        if header == None:               # Inititalise empty header
            self.h = copy.copy(PositionData.empty_header)
        else:
            self.h = header
        self.h['CoordinateSystem'] = postype.lower()
        self.coors = ['x','y','z'] if postype == 'xyz' else ['r','lon','lat']


    def __getitem__(self, var):
        if isinstance(var, str):
            if var in self.coors:
                return self.positions[self.coors.index(var)]
            else:
                raise Exception("PositionData object does not contain data under the key '{}'!".format(var))
        return self.positions[:,var]


    def __setitem__(self, var, value):
        if isinstance(var, str):
            if var in self.coors:
                self.positions[self.coors.index(var)] = value
            else:
                raise Exception("PositionData object does not contain the key '{}'!".format(var))
        else:
            raise ValueError("Cannot interpret {} as index for __setitem__!".format(var))


    def __len__(self):
        return len(self.positions[0])


    def __str__(self):
        return self.positions.__str__()


# =======================================================================================
# -------------------------------- II. FUNCTIONS ----------------------------------------
# =======================================================================================

# ***************************************************************************************
# A. Coordinate conversion functions:
# ***************************************************************************************

def convert_GSE_to_GSM(bxgse,bygse,bzgse,timegse):
    """GSE to GSM conversion
    main issue: need to get angle psigsm after Hapgood 1992/1997, section 4.3
    for debugging pdb.set_trace()
    for testing OMNI DATA use
    [bxc,byc,bzc]=convert_GSE_to_GSM(bx[90000:90000+20],by[90000:90000+20],bz[90000:90000+20],times1[90000:90000+20])
    """
 
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
    """function call [dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta)

    pdb.set_trace()  for debugging
    convert STEREO A magnetic field from RTN to GSE
    for prediction of structures seen at STEREO-A later at Earth L1
    so we do not include a rotation of the field to the Earth position
    """

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


def get_time_lag_wrt_earth(timestamp=None, satname='STEREO-A', v_mean=400., sun_syn=26.24):
    """Determines time lag with respect to Earth in rotation of solar wind for satellite
    away from Earth.

    NOTES ON PARKER SPIRAL LAG (lag_diff_r):
    See Simunac et al. 2009 Ann. Geophys. equation 1, see also Thomas et al. 2018 Space Weather
    difference in heliocentric distance STEREO-A to Earth,
    actually different for every point so take average of solar wind speed
    Omega is 360 deg/sun_syn in days, convert to seconds; sta_r in AU to m to km;
    convert to degrees
    minus sign: from STEREO-A to Earth the diff_r_deg needs to be positive
    because the spiral leads to a later arrival of the solar wind at larger
    heliocentric distances (this is reverse for STEREO-B!)
    """

    AU = 149597870.700 #AU in km

    if timestamp == None:
        timestamp = datetime.utcnow()
    timestamp = mdates.date2num(timestamp)

    pos = getpositions('data/positions_2007_2023_HEEQ_6hours.sav')
    pos_time_num = time_to_num_cat(pos.time)
    # take position of STEREO-A for time now from position file
    pos_time_now_ind = np.where(timestamp < pos_time_num)[0][0]

    if satname == 'STEREO-A':
        sat_pos = pos.sta
    elif satname == 'STEREO-B':
        sat_pos = pos.stb
    else:
        raise Exception("Not a valid satellite name to find position!")
    sat_r = sat_pos[0][pos_time_now_ind]
    earth_r = pos.earth_l1[0][pos_time_now_ind]

    # Get longitude and latitude
    lon_heeq = sat_pos[1][pos_time_now_ind]*180./np.pi
    lat_heeq = sat_pos[2][pos_time_now_ind]*180./np.pi

    # define time lag from satellite to Earth
    lag_l1 = abs(lon_heeq)/(360./sun_syn)

    # time lag from Parker spiral
    diff_r_deg = -(360./(sun_syn*86400.))*((sat_r-earth_r)*AU)/v_mean
    lag_diff_r = round(diff_r_deg/(360./sun_syn),2)

    return lag_l1, lag_diff_r


def cart2sphere(x, y, z):
    # convert cartesian to spherical coordinates
    r = np.sqrt(x**2. + y**2. + z**2.)
    theta = np.arctan2(z,np.sqrt(x**2. + y**2.))
    phi = np.arctan2(y,x)
    return (r, theta, phi)


def sphere2cart(r, phi, theta):
    # convert spherical to cartesian coordinates
    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
    return (x, y, z) 


# ***************************************************************************************
# B. Data reading:
# ***************************************************************************************

def get_dscovr_data_real_old():
    """
    Downloads and returns DSCOVR data 
    data from http://services.swpc.noaa.gov/products/solar-wind/
    if needed replace with ACE
    http://legacy-www.swpc.noaa.gov/ftpdir/lists/ace/
    get 3 or 7 day data
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json'
    
    Parameters
    ==========
    None

    Returns
    =======
    (data_minutes, data_hourly)
    data_minutes : np.rec.array
         Array of interpolated minute data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    data_hourly : np.rec.array
         Array of interpolated hourly data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    """
    
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    #download, see URLLIB https://docs.python.org/3/howto/urllib2.html
    with urllib.request.urlopen(url_plasma) as url:
        pr = json.loads (url.read().decode())
    with urllib.request.urlopen(url_mag) as url:
        mr = json.loads(url.read().decode())
    logger.info('get_dscovr_data_real: DSCOVR plasma data available')
    logger.info(str(pr[0]))
    logger.info('get_dscovr_data_real: DSCOVR MAG data available')
    logger.info(str(mr[0]))
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
    rbxgsm=np.zeros(len(mr))

    #convert variables to numpy arrays
    #mag
    for k in np.arange(0,len(mr),1):

        #handle missing data, they show up as None from the JSON data file
        if mr[k][6] is None: mr[k][6]=np.nan
        if mr[k][3] is None: mr[k][3]=np.nan
        if mr[k][2] is None: mr[k][2]=np.nan
        if mr[k][1] is None: mr[k][1]=np.nan

        rbtot[k]=float(mr[k][6])
        rbzgsm[k]=float(mr[k][3])
        rbygsm[k]=float(mr[k][2])
        rbxgsm[k]=float(mr[k][1])

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


    #interpolate to minutes 
    #rtimes_m=np.arange(rbtime_num[0],rbtime_num[-1],1.0000/(24*60))
    rtimes_m= round_to_hour(mdates.num2date(rbtime_num[0])) + np.arange(0,len(rbtime_num)) * timedelta(minutes=1) 
    #convert back to matplotlib time
    rtimes_m=mdates.date2num(rtimes_m)

    rbtot_m=np.interp(rtimes_m,rbtime_num,rbtot)
    rbzgsm_m=np.interp(rtimes_m,rbtime_num,rbzgsm)
    rbygsm_m=np.interp(rtimes_m,rbtime_num,rbygsm)
    rbxgsm_m=np.interp(rtimes_m,rbtime_num,rbxgsm)
    rpv_m=np.interp(rtimes_m,rptime_num,rpv)
    rpn_m=np.interp(rtimes_m,rptime_num,rpn)
    rpt_m=np.interp(rtimes_m,rptime_num,rpt)
    
    #interpolate to hours 
    #rtimes_h=np.arange(np.ceil(rbtime_num)[0],rbtime_num[-1],1.0000/24.0000)
    rtimes_h= round_to_hour(mdates.num2date(rbtime_num[0])) + np.arange(0,len(rbtime_num)/(60)) * timedelta(hours=1) 
    rtimes_h=mdates.date2num(rtimes_h)

    
    rbtot_h=np.interp(rtimes_h,rbtime_num,rbtot)
    rbzgsm_h=np.interp(rtimes_h,rbtime_num,rbzgsm)
    rbygsm_h=np.interp(rtimes_h,rbtime_num,rbygsm)
    rbxgsm_h=np.interp(rtimes_h,rbtime_num,rbxgsm)
    rpv_h=np.interp(rtimes_h,rptime_num,rpv)
    rpn_h=np.interp(rtimes_h,rptime_num,rpn)
    rpt_h=np.interp(rtimes_h,rptime_num,rpt)

    #make recarrays
    data_hourly=np.rec.array([rtimes_h,rbtot_h,rbxgsm_h,rbygsm_h,rbzgsm_h,rpv_h,rpn_h,rpt_h], \
    dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')])
    
    data_minutes=np.rec.array([rtimes_m,rbtot_m,rbxgsm_m,rbygsm_m,rbzgsm_m,rpv_m,rpn_m,rpt_m], \
    dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')])
    
    logger.info('get_dscovr_data_real: DSCOVR data read and interpolated to hour/minute resolution.')
    
    return data_minutes, data_hourly


def get_dscovr_data_real():
    """
    Downloads and returns DSCOVR data 
    data from http://services.swpc.noaa.gov/products/solar-wind/
    if needed replace with ACE
    http://legacy-www.swpc.noaa.gov/ftpdir/lists/ace/
    get 3 or 7 day data
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json'
    
    Parameters
    ==========
    None

    Returns
    =======
    (data_minutes, data_hourly)
    data_minutes : np.rec.array
         Array of interpolated minute data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    data_hourly : np.rec.array
         Array of interpolated hourly data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    """
    
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    #download, see URLLIB https://docs.python.org/3/howto/urllib2.html
    with urllib.request.urlopen(url_plasma) as url:
        pr = json.loads (url.read().decode())
    with urllib.request.urlopen(url_mag) as url:
        mr = json.loads(url.read().decode())
    logger.info('get_dscovr_data_real: DSCOVR plasma data available')
    logger.info(str(pr[0]))
    logger.info('get_dscovr_data_real: DSCOVR MAG data available')
    logger.info(str(mr[0]))
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
    rbxgsm=np.zeros(len(mr))

    #convert variables to numpy arrays
    #mag
    for k in np.arange(0,len(mr),1):

        #handle missing data, they show up as None from the JSON data file
        if mr[k][6] is None: mr[k][6]=np.nan
        if mr[k][3] is None: mr[k][3]=np.nan
        if mr[k][2] is None: mr[k][2]=np.nan
        if mr[k][1] is None: mr[k][1]=np.nan

        rbtot[k]=float(mr[k][6])
        rbzgsm[k]=float(mr[k][3])
        rbygsm[k]=float(mr[k][2])
        rbxgsm[k]=float(mr[k][1])

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

    #interpolate to minutes 
    #rtimes_m=np.arange(rbtime_num[0],rbtime_num[-1],1.0000/(24*60))
    rtimes_m= round_to_hour(mdates.num2date(rbtime_num[0])) + np.arange(0,len(rbtime_num)) * timedelta(minutes=1) 
    #convert back to matplotlib time
    rtimes_m=mdates.date2num(rtimes_m)

    rbtot_m=np.interp(rtimes_m,rbtime_num,rbtot)
    rbzgsm_m=np.interp(rtimes_m,rbtime_num,rbzgsm)
    rbygsm_m=np.interp(rtimes_m,rbtime_num,rbygsm)
    rbxgsm_m=np.interp(rtimes_m,rbtime_num,rbxgsm)
    rpv_m=np.interp(rtimes_m,rptime_num,rpv)
    rpn_m=np.interp(rtimes_m,rptime_num,rpn)
    rpt_m=np.interp(rtimes_m,rptime_num,rpt)

    # Pack into object
    dscovr_data = SatData({'time': rtimes_m,
                           'btot': rbtot_m, 'bx': rbxgsm_m, 'by': rbygsm_m, 'bz': rbzgsm_m,
                           'speed': rpv_m, 'density': rpn_m, 'temp': rpt_m},
                           source='DSCOVR')
    dscovr_data.h['DataSource'] = "DSCOVR (NOAA)"
    dscovr_data.h['SamplingRate'] = 1./24./60.
    dscovr_data.h['CoordinateSystem'] = 'GSM'
    
    logger.info('get_dscovr_data_real: DSCOVR data read completed.')
    
    return dscovr_data


def get_dscovr_data_all(P_filepath=None, M_filepath=None, starttime=None, endtime=None):
    """ Reads .nc format DSCOVR data from NOAA archive and returns np recarrays with
    variables under the JSON file format names.
    Data sourced from: 
    https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/1542848400000;1554163200000/f1m;m1m

    Note: if providing a directory as the filepath, use '*' to denote all files,
    which should be in the NOAA archive file format with similar string:
    --> oe_m1m_dscovr_s20190409000000_e20190409235959_p20190410031442_pub.nc
    (This should only be used with starttime/endtime options.)

    - DSCOVR data archive:
    https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/1543017600000;1543363199999
    https://www.ngdc.noaa.gov/dscovr/portal/index.html#/
    https://www.ngdc.noaa.gov/next-web/docs/guide/catalog.html#dscovrCatalog
    ftp://spdf.gsfc.nasa.gov/pub/data/dscovr/h0/mag/2018
    or get data via heliopy and ftp:
    https://docs.heliopy.org/en/stable/api/heliopy.data.dscovr.mag_h0.html#heliopy.data.dscovr.mag_h0

    BEST WAY TO GET UPDATED DSCOVR DATA:
    monthly folders, e.g.
    https://www.ngdc.noaa.gov/dscovr/data/2018/11/

    then 
    curl -O https://www.ngdc.noaa.gov/dscovr/data/2018/12/oe_f1m_dscovr_s20181207000000_e20181207235959_p20181208031650_pub.nc.gz

    zum entpacken:
    gunzip oe_f1m_dscovr_s20181207000000_e20181207235959_p20181208031650_pub.nc.gz
    netcdf files

    filenames are f1m for faraday and m1m for magnetometer, 1 minute averages

    Parameters
    ==========
    P_filepath : str
        Directory containing the DSCOVR plasma data.
    M_filepath : str
        Directory containing the DSCOVR magnetic field data.
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.

    Returns
    =======
    (data_minutes, data_hourly)
    data_minutes : np.rec.array
         Array of interpolated minute data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    data_hourly : np.rec.array
         Array of interpolated hourly data with format:
         dtype=[('time','f8'),('btot','f8'),('bxgsm','f8'),('bygsm','f8'),('bzgsm','f8'),\
            ('speed','f8'),('den','f8'),('temp','f8')]
    """

    if 'netCDF4' not in sys.modules:
        raise ImportError("read_archive_dscovr_data: netCDF4 not imported for DSCOVR data read!")

    # Pick out files to read:
    # -----------------------
    # A wildcard file search makes things much easier because these files have unpredictable names
    # due to the processing time being added into the file name.
    P_readfiles = []
    if P_filepath[-1] == '*':
        for filename in iglob(P_filepath):
            if 'f1m' in filename:
                testdate = [datetime.strptime(s.strip('s'), "%Y%m%d%H%M%S")
                            for s in filename.split('_') if s[0] == 's'][0]
                if testdate >= starttime and testdate < endtime:
                    P_readfiles.append(filename)
    P_readfiles.sort()

    M_readfiles = []
    if M_filepath[-1] == '*':
        for filename in iglob(M_filepath):
            if 'm1m' in filename:
                testdate = [datetime.strptime(s.strip('s'), "%Y%m%d%H%M%S")
                            for s in filename.split('_') if s[0] == 's'][0]
                if testdate >= starttime and testdate < endtime:
                    M_readfiles.append(filename)
    M_readfiles.sort()

    logger.info("get_dscovr_data_all: Reading {} DSCOVR archive files for time range {} till {}".format(len(P_readfiles),
                                                  datetime.strftime(starttime, "%Y-%m-%d"),
                                                  datetime.strftime(endtime, "%Y-%m-%d")))

    # Particle data:
    # --------------
    dp_v_m, dp_p_m, dp_t_m  = np.array([]), np.array([]), np.array([])
    dp_v_h, dp_p_h, dp_t_h  = np.array([]), np.array([]), np.array([])
    dp_time_m, dp_time_h = np.array([]), np.array([])
    for filepath in P_readfiles:
        ncdata = Dataset(filepath, 'r')
        time = np.array([mdates.date2num(datetime.utcfromtimestamp(x/1000.))
                         for x in ncdata.variables['time'][...]])
        dtype = [('time_tag', 'float'), ('density', 'float'), ('speed', 'float')]
        density = np.array((ncdata.variables['proton_density'])[...])
        speed = np.array((ncdata.variables['proton_speed'])[...])
        temp = np.array((ncdata.variables['proton_temperature'])[...])

        # Replace missing data:
        # Note: original arrays are masked - this only works because they're converted to np.array before
        density[np.where(density==float(ncdata.variables['proton_density'].missing_value))] = np.NaN
        speed[np.where(speed==float(ncdata.variables['proton_speed'].missing_value))] = np.NaN
        temp[np.where(temp==float(ncdata.variables['proton_temperature'].missing_value))] = np.NaN

        # Minute data:
        dp_v_m = np.hstack((dp_v_m, speed))
        dp_p_m = np.hstack((dp_p_m, density))
        dp_t_m = np.hstack((dp_t_m, temp))
        dp_time_m = np.hstack((dp_time_m, time))

    # Magnetic data:
    # --------------
    dm_bx_m, dm_by_m, dm_bz_m, dm_bt_m  = np.array([]), np.array([]), np.array([]), np.array([])
    dm_bx_h, dm_by_h, dm_bz_h, dm_bt_h  = np.array([]), np.array([]), np.array([]), np.array([])
    dm_time_m, dm_time_h = np.array([]), np.array([])
    for filepath in M_readfiles:
        ncdata = Dataset(filepath, 'r')
        time = np.array([mdates.date2num(datetime.utcfromtimestamp(x/1000.))
                         for x in ncdata.variables['time'][...]])
        bx = np.array((ncdata.variables['bx_gse'])[...])
        by = np.array((ncdata.variables['by_gse'])[...])
        bz = np.array((ncdata.variables['bz_gse'])[...])
        bt = np.array((ncdata.variables['bt'])[...])

        # Replace missing data:
        # Note: original arrays are masked - this only works because they're converted to np.array before
        bx[np.where(bx==float(ncdata.variables['bx_gse'].missing_value))] = np.NaN
        by[np.where(by==float(ncdata.variables['by_gse'].missing_value))] = np.NaN
        bz[np.where(bz==float(ncdata.variables['bz_gse'].missing_value))] = np.NaN
        bt[np.where(bt==float(ncdata.variables['bt'].missing_value))] = np.NaN

        # Minute data:
        dm_bx_m = np.hstack((dm_bx_m, bx))
        dm_by_m = np.hstack((dm_by_m, by))
        dm_bz_m = np.hstack((dm_bz_m, bz))
        dm_bt_m = np.hstack((dm_bt_m, bt))
        dm_time_m = np.hstack((dm_time_m, time))

    # Pack into arrays:
    # --------------------
    dscovr_data = SatData({'time': dm_time_m,
                           'btot': dm_bt_m, 'bx': dm_bx_m, 'by': dm_by_m, 'bz': dm_bz_m,
                           'speed': dp_v_m, 'density': dp_p_m, 'temp': dp_t_m},
                           source='DSCOVR')
    dscovr_data.h['DataSource'] = "DSCOVR (NOAA archives)"
    dscovr_data.h['SamplingRate'] = 1./24./60.
    dscovr_data.h['CoordinateSystem'] = 'GSM'
    dscovr_data = dscovr_data.cut(starttime=starttime, endtime=endtime)
    
    logger.info('get_dscovr_data_all: DSCOVR data successfully read.')
            
    return dscovr_data


def get_noaa_dst():
    """Loads real-time Dst data from NOAA webpage:
    http://services.swpc.noaa.gov/products/kyoto-dst.json

    Parameters
    ==========
    None

    Returns
    =======
    dst : SatData object
        Object containing arrays of time and dst values.
    """

    url_dst='http://services.swpc.noaa.gov/products/kyoto-dst.json'
    with urllib.request.urlopen(url_dst) as url:
        dr = json.loads    (url.read().decode())
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
    logger.info("NOAA real-time Dst data loaded.")

    dst_data = SatData({'time': rdst_time, 'dst': rdst},
                       source='KyotoDst')
    dst_data.h['DataSource'] = "Kyoto Dst (NOAA)"
    dst_data.h['SamplingRate'] = 1./24.

    return dst_data


def get_past_dst(filepath=None, starttime=None, endtime=None):
    """Will read Dst values from IAGA2002-format file. Data can be 
    downloaded from this webpage:
    http://wdc.kugi.kyoto-u.ac.jp/dstae/index.html

    Parameters
    ==========
    filepath : str
        Direct filepath to file containing Dst. If None, realtime data is used.
    starttime: datetime.datetime
        Starttime of Dst data.
    endtime : datetime.datetime
        Endtime of Dst data.

    Returns
    =======
    dst : SatData object
        Object containing arrays of time and dst values.
    """

    f = open(filepath, 'r')
    lines = f.readlines()

    # Remove header data and split strings:
    datastr = [c.strip().split(' ') for c in lines if (c[0] != ' ' and c[0] != 'D')]
    dst_time = np.array([mdates.date2num(datetime.strptime(d[0]+d[1], "%Y-%m-%d%H:%M:%S.%f")) for d in datastr])
    dst = np.array([float(d[-1]) for d in datastr])

    dst_data = SatData({'time': dst_time, 'dst': dst},
                       source='KyotoDst')
    dst_data.h['DataSource'] = "Kyoto Dst (Kyoto WDC)"
    dst_data.h['SamplingRate'] = 1./24.
    dst_data = dst_data.cut(starttime=starttime, endtime=endtime)
    
    return dst_data


def get_omni_data(filepath='', download=False, dldir='data'):
    """
    Will download and read OMNI2 data file (in .dat format).
    Variable definitions from OMNI2 dataset:
    see http://omniweb.gsfc.nasa.gov/html/ow_data.html

    FORMAT(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2, F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1)
    1963   1  0 1771 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9  7  23    -6  119 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 999.9 999.9 99999 99999 99.9

    Parameters
    ==========
    filepath : str (default='', reverts to 'data/omni2_all_years.dat')
        Path to file to read.
    download : bool (default=False)
        If True, file will be downloaded to directory (dldir)
    dldir : str (default='data')
        If using download, file will be downloaded to this dir and then be read.

    Returns
    =======
    omni_data : predstorm.SatData
    """

    if filepath != '' and not os.path.exists(filepath):
        raise Exception("get_omni_data: {} does not exist! Run get_omni_data(download=True) to download file.".format(filepath))

    if download:
        omni2_url='ftp://nssdcftp.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
        logger.info("get_omni_data: downloading OMNI2 data from {}".format(omni2_url))
        tofile = os.path.join(dldir, 'omni2_all_years.dat')
        try: 
            urllib.request.urlretrieve(omni2_url, tofile)
            logger.info("get_omni_data: OMNI2 data successfully downloaded.")
            filepath = tofile
        except urllib.error.URLError as e:
            logger.error("get_omni_data: OMNI2 data download failed (reason: {})".format(e.reason))
        pickle.dump(o, open('data/omni2_all_years_pickle.p', 'wb') )

    if filepath == '':
        filepath = 'data/omni2_all_years.dat'

    #check how many rows exist in this file
    f=open(filepath)
    dataset= len(f.readlines())
    #print(dataset)
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
    
    
    j=0
    logger.info('get_omni_data: Reading OMNI2 data ...')
    with open('data/omni2_all_years.dat') as f:
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

    #convert time to matplotlib format
    times1=np.zeros(len(year)) #datetime time
    for index in range(0,len(year)):
        #first to datetimeobject 
        timedum=datetime(int(year[index]), 1, 1) + timedelta(day[index] - 1) +timedelta(hours=hour[index])
        #then to matlibplot dateformat:
        times1[index] = mdates.date2num(timedum)

    omni_data = SatData({'time': times1,
                         'btot': btot, 'bx': bx, 'by': bygsm, 'bz': bzgsm,
                         'speed': speed, 'density': den, 'pdyn': pdyn,
                         'dst': dst, 'kp': kp},
                         source='OMNI')
    omni_data.h['DataSource'] = "OMNI (NASA OMNI2 data)"
    if download:
        omni_data.h['SourceURL'] = omni2_url
    omni_data.h['SamplingRate'] = times1[1] - times1[0]
    omni_data.h['CoordinateSystem'] = 'GSM'
    
    return omni_data


def get_omni_data_old():
    """FORMAT(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2, F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1)
    1963   1  0 1771 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9  7  23    -6  119 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 999.9 999.9 99999 99999 99.9

    define variables from OMNI2 dataset
    see http://omniweb.gsfc.nasa.gov/html/ow_data.html

    omni2_url='ftp://nssdcftp.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
    """

    #check how many rows exist in this file
    f=open('data/omni2_all_years.dat')
    dataset= len(f.readlines())
    #print(dataset)
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
    
    
    j=0
    print('Read OMNI2 data ...')
    with open('data/omni2_all_years.dat') as f:
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
      

    #convert time to matplotlib format
    #http://docs.sunpy.org/en/latest/guide/time.html
    #http://matplotlib.org/examples/pylab_examples/date_demo2.html

    times1=np.zeros(len(year)) #datetime time
    print('convert time start')
    for index in range(0,len(year)):
        #first to datetimeobject 
        timedum=datetime(int(year[index]), 1, 1) + timedelta(day[index] - 1) +timedelta(hours=hour[index])
        #then to matlibplot dateformat:
        times1[index] = mdates.date2num(timedum)
    print('convert time done')   #for time conversion

    print('all done.')
    print(j, ' datapoints')   #for reading data from OMNI file
    
    #make structured array of data
    omni_data=np.rec.array([times1,btot,bx,by,bz,bygsm,bzgsm,speed,speedx,den,pdyn,dst,kp], \
    dtype=[('time','f8'),('btot','f8'),('bx','f8'),('by','f8'),('bz','f8'),\
    ('bygsm','f8'),('bzgsm','f8'),('speed','f8'),('speedx','f8'),('den','f8'),('pdyn','f8'),('dst','f8'),('kp','f8')])
    
    return omni_data


def get_predstorm_data_realtime(resolution='hour'):
    """Reads data from PREDSTORM real-time output.

    Parameters
    ==========
    resolution : str ['hour'(=default) or 'minute']
        Data resolution, only two available.

    Returns
    =======
    pred_data : predstorm.SatData
        Object containing all data.
    """

    if resolution in ['h','hour']:
        filepath = "https://www.iwf.oeaw.ac.at/fileadmin/staff/SP/cmoestl/readtime/predstorm_real.txt"
    elif resolution in ['m','minute']:
        filepath = "https://www.iwf.oeaw.ac.at/fileadmin/staff/SP/cmoestl/readtime/predstorm_real_1m.txt"
    else:
        logger.error("get_predstorm_data_realtime: {} is not a valid option for resolution! Use 'hour' or 'minute.")

    logger.info("get_predstorm_data_realtime: Downloading data from {}".format(filepath))
    dtype = [('time', 'float'), ('btot', 'float'), ('bx', 'float'), ('by', 'float'), ('bz', 'float'),
           ('density', 'float'), ('speed', 'float'), ('dst', 'float'), ('kp', 'float')]
    data = np.loadtxt(filepath, usecols=[6,7,8,9,10,11,12,13,14], dtype=dtype)

    pred_data = SatData({var[0]: data[var[0]] for var in dtype},
                        source='PREDSTORM')
    pred_data.h['DataSource'] = "PREDSTORM (L5-to-L1 prediction)"
    pred_data.h['SamplingRate'] = 1./24.
    pred_data.h['CoordinateSystem'] = 'GSM'

    return pred_data


def download_stereoa_data_beacon(filedir="sta_beacon", starttime=None, endtime=None, ndays=14):
    """
    Downloads STEREO-A beacon data files to folder. If starttime/endtime are not
    defined, the data from the last two weeks is downloaded automatically.

    D STEREO SCIENCE CENTER these are the cdf files for the beacon data, daily
    browse data, ~ 200kb
    https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic/2018/05/STA_LB_PLA_BROWSE_20180502_V12.cdf       
    original data, ~1 MB
    https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic/2018/05/STA_LB_PLA_20180502_V12.cdf
    
    Last 2 hours are available here at NOAA:
    http://legacy-www.swpc.noaa.gov/ftpdir/lists/stereo/
    http://legacy-www.swpc.noaa.gov/stereo/STEREO_data.html
    
    Parameters
    ==========
    filedir : str
        Path to directory where files should be saved.
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.
    ndays : int
        Number of days of data to read (before starttime). Invalid if both
        starttime and endtime are provided.

    Returns
    =======
    None
    """
 
    logger.info('download_stereoa_data_beacon: Starting download of STEREO-A beacon data from STEREO SCIENCE CENTER...')

    # If folder is not here, create
    if os.path.isdir(filedir) == False: os.mkdir(filedir)

    plastic_location='https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/plastic'
    impact_location='https://stereo-ssc.nascom.nasa.gov/data/beacon/ahead/impact'

    if starttime == None and endtime == None:       # Read most recent data
        starttime = datetime.utcnow() - timedelta(days=ndays-1)
        endtime = datetime.utcnow()
    elif starttime != None and endtime == None:     # Read past data
        endtime = starttime + timedelta(days=ndays)
    elif starttime == None and endtime != None:
        starttime = endtime - timedelta(days=ndays)
    else:
        ndays = (endtime-starttime).days

    dates = [starttime+timedelta(days=n) for n in range(0, ndays)]
    logger.info("Following dates listed to be downloaded: {}".format(dates))

    for date in dates:
        stayear = datetime.strftime(date, "%Y")
        stamonth = datetime.strftime(date, "%m")
        staday = datetime.strftime(date, "%d")
        daynowstr = stayear+stamonth+staday

        # Plastic
        sta_pla_file_str = 'STA_LB_PLASTIC_'+daynowstr+'_V12.cdf'
        # Impact
        sta_mag_file_str = 'STA_LB_IMPACT_'+daynowstr+'_V02.cdf' 
        
        # Check if file is already there, otherwise download
        if not os.path.exists(os.path.join(filedir, sta_pla_file_str)):
            # Download files if they are not already downloaded
            http_sta_pla_file_str = os.path.join(plastic_location, stayear, stamonth, sta_pla_file_str)
            # Check if url exists
            try: 
                urllib.request.urlretrieve(http_sta_pla_file_str, os.path.join(filedir, sta_pla_file_str))
                logger.info(sta_pla_file_str+" downloaded")
            except urllib.error.URLError as e:
                logger.error("download_stereoa_data_beacon: Could not download {} for reason:".format(http_sta_pla_file_str, e.reason))
                http_sta_pla_file_str = os.path.join(plastic_location, stayear, stamonth, sta_pla_file_str.replace("V12", "V11"))
                logger.info("Trying file version V11...")
                try: 
                    urllib.request.urlretrieve(http_sta_pla_file_str, os.path.join(filedir, sta_pla_file_str))
                    logger.info(sta_pla_file_str+" downloaded")
                except:
                    logger.error("download_stereoa_data_beacon: Could not download {} for reason:".format(http_sta_pla_file_str, e.reason))

        
        if not os.path.exists(os.path.join(filedir, sta_mag_file_str)):
            http_sta_mag_file_str = os.path.join(impact_location, stayear, stamonth, sta_mag_file_str)
            try: 
                urllib.request.urlretrieve(http_sta_mag_file_str, os.path.join(filedir, sta_mag_file_str))
                logger.info(sta_mag_file_str+" downloaded")
            except urllib.error.URLError as e:
                logger.error("download_stereoa_data_beacon: Could not download {} for reason:".format(http_sta_mag_file_str, e.reason))

    logger.info('download_stereoa_data_beacon: STEREO-A beacon data download complete.')

    return
    

def read_stereoa_data_beacon(filepath="sta_beacon/", starttime=None, endtime=None, ndays=14):
    """
    Reads STEREO-A beacon data from CDF files. Files should be stored under filepath
    with the naming format STA_LB_PLA_20180502_V12.cdf. Use the function
    download_stereoa_data_beacon to get these files.
    
    Parameters
    ==========
    filedir : str
        Path to directory where files should be saved.
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.

    Returns
    =======
    (data_minutes, data_hourly)
    data_minutes : np.rec.array
         Array of interpolated minute data with format:
         dtype=[('time','f8'),('btot','f8'),('br','f8'),('bt','f8'),('bn','f8'),\
            ('speedr','f8'),('den','f8')]
    data_hourly : np.rec.array
         Array of interpolated hourly data with format:
         dtype=[('time','f8'),('btot','f8'),('br','f8'),('bt','f8'),('bn','f8'),\
            ('speedr','f8'),('den','f8')]
    """

    # Define stereo-a variables with open size
    sta_ptime=np.zeros(0)  
    sta_vr=np.zeros(0)  
    sta_den=np.zeros(0)  
    sta_temp=np.zeros(0)  

    sta_btime=np.zeros(0)  
    sta_br=np.zeros(0)  
    sta_bt=np.zeros(0)  
    sta_bn=np.zeros(0)

    if starttime == None and endtime == None:       # Read most recent data
        starttime = datetime.utcnow() - timedelta(days=ndays-1)
        endtime = datetime.utcnow()
    elif starttime != None and endtime == None:     # Read past data
        endtime = starttime + timedelta(days=ndays)
    elif starttime == None and endtime != None:
        starttime = endtime - timedelta(days=ndays)
    else:
        ndays = (endtime-starttime).days

    readdates = [datetime.strftime(starttime+timedelta(days=n), "%Y%m%d") for n in range(0, ndays)]

    logger.info("read_stereoa_data_beacon: Starting data read for {} days from {} till {}".format(ndays, readdates[0], readdates[-1]))

    for date in readdates:
    
        # PLASMA
        # ------
        pla_version = 'V12'
        sta_pla_file = os.path.join(filepath, 'STA_LB_PLASTIC_'+date+'_'+pla_version+'.cdf')
        if os.path.exists(sta_pla_file):
            sta_file =  cdflib.CDF(sta_pla_file)
        else:
            logger.error("read_stereoa_data_beacon: File {} for reading doesn't exist!".format(sta_pla_file))
            raise Exception("File {} for reading doesn't exist!".format(sta_pla_file))
            
        # Variables Epoch_MAG: Epoch1: CDF_EPOCH [1875]
        sta_time=epoch_to_num(sta_file.varget('Epoch1'))
        sta_dvr=sta_file.varget('Velocity_RTN')[:,0]
        sta_dden=sta_file.varget('Density')

        # Replace missing data with nans:
        mis=np.where(sta_time < -1e30)
        sta_time[mis]=np.nan
        mis=np.where(sta_dvr < -1e30)
        sta_dvr[mis]=np.nan
        mis=np.where(sta_dden < -1e30)
        sta_dden[mis]=np.nan
        
        sta_ptime=np.append(sta_ptime, sta_time)
        sta_vr=np.append(sta_vr,sta_dvr)
        sta_den=np.append(sta_den,sta_dden)
        
        # MAGNETIC FIELD
        # --------------
        mag_version = 'V02'
        sta_mag_file = os.path.join(filepath, 'STA_LB_IMPACT_'+date+'_'+mag_version+'.cdf' )
        if os.path.exists(sta_mag_file):
            sta_filem =  cdflib.CDF(sta_mag_file)
        else:
            logger.error("read_stereoa_data_beacon: File {} for reading doesn't exist!".format(sta_mag_file))

        #variables Epoch_MAG: CDF_EPOCH [8640]
        #MAGBField: CDF_REAL4 [8640, 3]
        #pdb.set_trace()
        sta_time=epoch_to_num(sta_filem.varget('Epoch_MAG'))
        #d stands for dummy
        sta_dbr=sta_filem.varget('MAGBField')[:,0]
        sta_dbt=sta_filem.varget('MAGBField')[:,1]
        sta_dbn=sta_filem.varget('MAGBField')[:,2]

        sta_btime=np.append(sta_btime, sta_time)
        sta_br=np.append(sta_br,sta_dbr)
        sta_bt=np.append(sta_bt,sta_dbt)
        sta_bn=np.append(sta_bn,sta_dbn)
  
    #make total field variable
    sta_btot=np.sqrt(sta_br**2+sta_bt**2+sta_bn**2)
    
    # Interpolate to minutes:
    sta_time_m=np.arange(np.ceil(sta_btime)[0],sta_btime[-1],1.0000/(24*60))
    sta_btot_m=np.interp(sta_time_m,sta_btime,sta_btot)
    sta_br_m=np.interp(sta_time_m,sta_btime,sta_br)
    sta_bt_m=np.interp(sta_time_m,sta_btime,sta_bt)
    sta_bn_m=np.interp(sta_time_m,sta_btime,sta_bn)
    sta_vr_m=np.interp(sta_time_m,sta_ptime,sta_vr)
    sta_den_m=np.interp(sta_time_m,sta_ptime,sta_den)
    
    # Pack into arrays:
    # --------------------
    empty = np.zeros(len(sta_time_m))
    stereo_data = SatData({'time': sta_time_m,
                           'btot': sta_btot_m, 'br': sta_br_m, 'bt': sta_bt_m, 'bn': sta_bn_m,
                           # Create empty keys for future coordinate conversion:
                           # (Not the most elegant solution but it'll do for now)
                           'bx': empty, 'by': empty, 'bz': empty,
                           'speed': sta_vr_m, 'density': sta_den_m},
                           source='STEREO-A')
    stereo_data.h['DataSource'] = "STEREO-A (beacon)"
    stereo_data.h['SamplingRate'] = 1./24./60.
    stereo_data.h['CoordinateSystem'] = 'RTN'
    stereo_data.h['Instruments'] = ['PLASTIC', 'IMPACT']
    stereo_data.h['FileVersion'] = {'PLASTIC': pla_version, 'IMPACT': mag_version}
    stereo_data = stereo_data.cut(starttime=starttime, endtime=endtime)

    logger.info('STEREO-A (RTN) beacon data interpolated to hour/minute resolution.')
    
    return stereo_data


def get_position_data(filepath, times, rlonlat=False):
    """Reads position data from pickled heliopy.spice.Trajectory into
    PositionData object.

    Parameters
    ==========
    filepath : str
        Path to pickled position data.
    times : np.array
        Array of time values (in date2num format) for position data.
    rlonlat : bool (default=False)
        If True, returns coordinates in (r, lon, lat) format, not (x,y,z).

    Returns
    =======
    Positions : predstorm.PositionData object
    """

    logger.info("get_position_data: Loading position data from {}".format(filepath))
    refframe = os.path.split(filepath)[-1].split('_')[-2]
    posdata = pickle.load(open(filepath, 'rb'))
    posx = np.interp(times, mdates.date2num(posdata.times), posdata.x)
    posy = np.interp(times, mdates.date2num(posdata.times), posdata.y)
    posz = np.interp(times, mdates.date2num(posdata.times), posdata.z)

    if rlonlat:
        r, theta, phi = cart2sphere(posx, posy, posz)
        Positions = PositionData([r, theta, phi], 'rlonlat')
    else:
        Positions = PositionData([posx, posy, posz], 'xyz')
    Positions.h['Units'] = 'km'
    Positions.h['ReferenceFrame'] = refframe
    
    return Positions


# ***************************************************************************************
# C. SatData handling functions:
# ***************************************************************************************

def merge_Data(satdata1, satdata2, keys=None):
    """Concatenates two SatData objects into one. Dataset #1 should be behind in time
    so that dataset #2 can just be added on to the end.

    Parameters
    ==========
    satdata1 : predstorm_module.SatData
        Path to directory where files should be saved.
    satdata2 : predstorm_module.SatData
        Datetime object with the required starttime of the input data.
    keys : list (default=None)
        List of input keys. If None, all are used.

    Returns
    =======
    new SatData object with both datasets
    """

    logger.info("merge_Data: Will merge data from {} and {}...".format(satdata1.source, satdata2.source))
    if keys == None:
        keys = list(set(satdata1.vars).intersection(satdata2.vars))
        logger.info("merge_Data: No keys defined, using common keys: {}".format(keys))

    for k in keys:
        if k not in satdata2.vars:
            logger.error("merge_Data: Dataset1 contains key ({}) not available in Dataset2!".format(k))
            raise Exception("Dataset1 contains key ({}) not available in Dataset2!".format(k))

    # Find num of points for addition:
    timestep = satdata1.h['SamplingRate']
    n_new_time = len(np.arange(satdata1['time'][-1] + timestep, satdata2['time'][-1], timestep))  
    # Make time array with matching steps
    new_time = satdata1['time'][-1] + timestep + np.arange(0, n_new_time) * timestep

    datadict = {}
    datadict['time'] = np.concatenate((satdata1['time'], new_time))
    for k in keys:
        # Interpolate dataset #2 to array matching dataset #1
        int_var = np.interp(new_time, satdata2['time'], satdata2[k])
        # Make combined array data
        datadict[k] = np.concatenate((satdata1[k], int_var))

    MergedData = SatData(datadict, source=satdata1.source+'+'+satdata2.source)
    tf = "%Y-%m-%d %H:%M:%S"
    MergedData.h['DataSource'] = '{} ({} - {}) & {} ({} - {})'.format(satdata1.h['DataSource'],
                                                datetime.strftime(mdates.num2date(satdata1['time'][0]), tf),
                                                datetime.strftime(mdates.num2date(satdata1['time'][-1]), tf),
                                                satdata2.h['DataSource'],
                                                datetime.strftime(mdates.num2date(new_time[0]), tf),
                                                datetime.strftime(mdates.num2date(new_time[-1]), tf))
    MergedData.h['SamplingRate'] = timestep
    MergedData.h['CoordinateSystem'] = satdata1.h['CoordinateSystem']
    MergedData.h['Instruments'] = satdata1.h['Instruments'] + satdata2.h['Instruments']
    MergedData.h['FileVersion'] = {**satdata1.h['FileVersion'], **satdata2.h['FileVersion']}

    logger.info("merge_Data: Finished merging data.")

    return MergedData


# ***************************************************************************************
# D. Basic data handling functions:
# ***************************************************************************************

def getpositions(filename):  
    pos=scipy.io.readsav(filename)  
    print
    return pos


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
    """http://docs.sunpy.org/en/latest/guide/time.html
    http://matplotlib.org/examples/pylab_examples/date_demo2.html
    """

    print('convert time start')
    for index in range(0,dataset):
        #first to datetimeobject 
        timedum=datetime(int(year[index]), 1, 1) + timedelta(day[index] - 1) +timedelta(hours=hour[index])
        #then to matlibplot dateformat:
        times1[index] = matplotlib.dates.date2num(timedum)
        #print time
        #print year[index], day[index], hour[index]
    print('convert time done')   #for time conversion


def round_to_hour(dt):
    '''
    round datetime objects to nearest hour
    '''
    dt_start_of_hour = dt.replace(minute=0, second=0, microsecond=0)
    dt_half_hour = dt.replace(minute=30, second=0, microsecond=0)

    if dt >= dt_half_hour:
        # round up
        dt = dt_start_of_hour + timedelta(hours=1)
    else:
        # round down
        dt = dt_start_of_hour
    return dt


def interp_nans(ar):
    """Linearly interpolates over nans in array."""

    inds = np.isnan(ar)
    ar[inds] = np.interp(inds.nonzero()[0], (~inds).nonzero()[0], ar[~inds])
    return ar

#def sunriseset(location_name):
"""
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
"""

def epoch_to_num(epoch):
    """
    Taken from spacepy https://pythonhosted.org/SpacePy/_modules/spacepy/pycdf.html#Library.epoch_to_num
    Convert CDF EPOCH to matplotlib number.

    Same output as :func:`~matplotlib.dates.date2num` and useful for
    plotting large data sets without converting the times through datetime.

    Parameters
    ==========
    epoch : double
        EPOCH to convert. Lists and numpy arrays are acceptable.

    Returns
    =======
    out : double
        Floating point number representing days since 0001-01-01.
    """
    #date2num day 1 is 1/1/1 00UT
    #epoch 1/1/1 00UT is 31622400000.0 (millisecond)
    return (epoch - 31622400000.0) / (24 * 60 * 60 * 1000.0) + 1.0
 

# ***************************************************************************************
# E. Other:
# ***************************************************************************************


def init_logging(verbose=False):

    # DEFINE LOGGING MODULE:
    logging.config.fileConfig(os.path.join(os.path.dirname(__file__), 'config/logging.ini'), disable_existing_loggers=False)
    logger = logging.getLogger(__name__)
    # Add handler for logging to shell:
    sh = logging.StreamHandler()
    if verbose:
        sh.setLevel(logging.INFO)
    else:
        sh.setLevel(logging.ERROR)
    shformatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    sh.setFormatter(shformatter)
    logger.addHandler(sh)

    return logger
