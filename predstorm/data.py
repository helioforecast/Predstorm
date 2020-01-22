#!/usr/bin/env python
"""
This is the data handling module for the predstorm package, containing the main
data class SatData and all relevant data handling functions and procedures.

Author: C. Moestl, R. Bailey, IWF Graz, Austria
twitter @chrisoutofspace, https://github.com/cmoestl
started April 2018, last update May 2019

Python 3.7
Packages not included in anaconda installation: cdflib (https://github.com/MAVENSDC/cdflib)

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

# Standard
import os
import sys
import copy
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from dateutil import tz
import gzip
import logging
import numpy as np
import pdb
import pickle
import re
import scipy
import scipy.io
import shutil
import subprocess
from matplotlib.dates import date2num, num2date
from glob import iglob
import json
import urllib

# External
import cdflib
import heliosat
from heliosat.spice import transform_frame
import astropy.time
import spiceypy
try:
    from netCDF4 import Dataset
except:
    pass

# Local
from .predict import make_kp_from_wind, calc_ring_current_term
from .predict import make_aurora_power_from_wind, calc_newell_coupling
from .predict import calc_dst_burton, calc_dst_obrien, calc_dst_temerin_li
from .config.constants import AU, dist_to_L1

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
    .state : np.array (dtype=object)
        Array of None, str if defining state of data (e.g. 'quiet', 'cme').
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
                    'speed', 'speedx', 'density', 'temp', 'pdyn',
                    'bx', 'by', 'bz', 'btot',
                    'br', 'bt', 'bn',
                    'dst', 'kp', 'aurora', 'ec', 'ae']

    empty_header = {'DataSource': '',
                    'SourceURL' : '',
                    'SamplingRate': None,
                    'ReferenceFrame': '',
                    'FileVersion': {},
                    'Instruments': [],
                    'RemovedTimes': [],
                    'PlasmaDataIntegrity': 10
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
        if len(input_dict['time']) == 0:
            logger.warning("SatData.__init__: Inititating empty array! Is the data missing?")
        # Create data array attribute
        data = [input_dict[x] if x in dt else np.zeros(len(input_dict['time'])) for x in SatData.default_keys]
        self.data = np.asarray(data)
        # Create array for state classifiers (currently empty)
        self.state = np.array([None]*len(self.data[0]), dtype='object')
        # Add new attributes to the created instance
        self.source = source
        if header == None:               # Inititalise empty header
            self.h = copy.deepcopy(SatData.empty_header)
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
        ostr += "First data point:\t{}\n".format(num2date(self['time'][0]))
        ostr += "Last data point:\t{}\n".format(num2date(self['time'][-1]))
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

    def convert_mag_to(self, refframe):
        """Converts MAG from one refframe to another."""
        
        barray = np.stack((self['bx'], self['by'], self['bz']), axis=1)
        tarray = [num2date(t).replace(tzinfo=None) for t in self['time']]
        #b_transformed = transform_frame(tarray, barray, self.h['ReferenceFrame'], refframe)
        for i in range(0, len(tarray)):
            barray[i] = spiceypy.mxv(spiceypy.pxform(self.h['ReferenceFrame'], refframe,
                                     spiceypy.datetime2et(tarray[i])),
                                     barray[i])
        self['bx'], self['by'], self['bz'] = barray[:,0], barray[:,1], barray[:,2]
        self.h['ReferenceFrame'] = refframe

        return self


    def convert_GSE_to_GSM(self):
        """GSE to GSM conversion
        main issue: need to get angle psigsm after Hapgood 1992/1997, section 4.3
        for debugging pdb.set_trace()
        for testing OMNI DATA use
        [bxc,byc,bzc]=convert_GSE_to_GSM(bx[90000:90000+20],by[90000:90000+20],bz[90000:90000+20],times1[90000:90000+20])

        CAUTION: Overwrites original data.
        """
     
        logger.info("Converting GSE magn. values to GSM")
        mjd=np.zeros(len(self['time']))

        #output variables
        bxgsm=np.zeros(len(self['time']))
        bygsm=np.zeros(len(self['time']))
        bzgsm=np.zeros(len(self['time']))

        for i in np.arange(0,len(self['time'])):
            #get all dates right
            jd=astropy.time.Time(num2date(self['time'][i]), format='datetime', scale='utc').jd
            mjd[i]=float(int(jd-2400000.5)) #use modified julian date    
            T00=(mjd[i]-51544.5)/36525.0
            dobj=num2date(self['time'][i])
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
        self.h['ReferenceFrame'] = 'GSM'

        return self


    def convert_RTN_to_GSE(self, pos_obj=[], pos_tnum=[]):
        """Converts RTN to GSE coordinates.

        function call [dbr,dbt,dbn]=convert_RTN_to_GSE_sta_l1(sta_br7,sta_bt7,sta_bn7,sta_time7, pos.sta)

        pdb.set_trace()  for debugging
        convert STEREO A magnetic field from RTN to GSE
        for prediction of structures seen at STEREO-A later at Earth L1
        so we do not include a rotation of the field to the Earth position
        """

        logger.info("Converting RTN magn. values to GSE")
        #output variables
        heeq_bx=np.zeros(len(self['time']))
        heeq_by=np.zeros(len(self['time']))
        heeq_bz=np.zeros(len(self['time']))
        
        bxgse=np.zeros(len(self['time']))
        bygse=np.zeros(len(self['time']))
        bzgse=np.zeros(len(self['time']))

        AU = 149597870.700 #in km
        
        ########## first RTN to HEEQ 

        # NEW METHOD:
        if len(pos_obj) == 0 and len(pos_tnum) == 0:
            if self.pos == None:
                raise Exception("Load position data (SatData.load_position_data()) before calling convert_RTN_to_GSE()!")
        
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
                if self.pos.h['CoordinateSystem'] == 'rlonlat':
                    xa, ya, za = sphere2cart(self.pos['r'][i], self.pos['lon'][i], self.pos['lat'][i])
                    xa, ya, za = xa/AU, ya/AU, za/AU
                else:
                    xa, ya, za = self.pos[i]/AU

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

        # just testing - remove this later!
        # heeq_bx=self['br']
        # heeq_by=self['bt']
        # heeq_bz=self['bn']

        #get modified Julian Date for conversion as in Hapgood 1992
        jd=np.zeros(len(self['time']))
        mjd=np.zeros(len(self['time']))
        
        #then HEEQ to GSM
        #-------------- loop go through each date
        for i in np.arange(0,len(self['time'])):
            jd[i]=astropy.time.Time(num2date(self['time'][i]), format='datetime', scale='utc').jd
            mjd[i]=float(int(jd[i]-2400000.5)) #use modified julian date    
            #then lambda_sun
            T00=(mjd[i]-51544.5)/36525.0
            dobj=num2date(self['time'][i])
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
        self.h['ReferenceFrame'] += '-GSE'

        return self


    def get_position(self, timestamp):
        """Returns position of satellite at given timestamp. Coordinates
        are provided in (r,lon,lat) format. Change rlonlat to False to get
        (x,y,z) coordinates.

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

        tind = np.where(date2num(timestamp) < self.data[0])[0][0]
        return self.pos[tind]


    def load_positions(self, refframe='HEEQ', units='AU', observer='SUN', rlonlat=True, l1_corr=False):
        """Loads data on satellite position into data object. Data is loaded using a
        heliosat.SpiceObject.

        Parameters
        ==========
        refframe : str (default=='HEEQ')
            observer reference frame
        observer : str (default='SUN')
            observer body name
        units : str (default='AU')
            output units - m / km / AU
        rlonlat : bool (default=True)
            If True, returns coordinates in (r, lon, lat) format, not (x,y,z).
        l1_corr : bool (default=False)
            Corrects Earth position to L1 position if True.

        Returns
        =======
        self with new data in self.pos
        """

        logger.info("load_positions: Loading position data into {} data".format(self.source))
        t_traj = [num2date(i).replace(tzinfo=None) for i in self['time']]
        traj = self.h['HeliosatObject'].trajectory(t_traj, frame=refframe, units=units,
                                                   observer=observer)
        posx, posy, posz = traj[:,0], traj[:,1], traj[:,2]

        if l1_corr:
            if units == 'AU':
                corr = dist_to_L1/AU
            elif units == 'm':
                corr = dist_to_L1/1e3
            elif units == 'km':
                corr = dist_to_L1
            posx = posx - corr
        if rlonlat:
            r, theta, phi = cart2sphere(posx, posy, posz)
            Positions = PositionData([r, phi, theta], 'rlonlat')
        else:
            Positions = PositionData([posx, posy, posz], 'xyz')

        Positions.h['Units'] = units
        Positions.h['ReferenceFrame'] = refframe
        Positions.h['Observer'] = observer
        self.pos = Positions

        return self


    def return_position_details(self, timestamp, sun_syn=26.24):
        """Returns a string describing STEREO-A's current whereabouts.

        Parameters
        ==========
        positions : ???
            Array containing spacecraft positions at given time.
        DSCOVR_lasttime : float
            Date of last DSCOVR measurements in number form.

        Returns
        =======
        stereostr : str
            Nicely formatted string with info on STEREO-A's location with
            with respect to Earth and L5/L1.
        """

        if self.pos == None:
            logger.warning("Loading position data (SatData.load_positions()) for return_position_details()!")
            self.load_positions()
        L1Pos = get_l1_position(timestamp, units=self.pos.h['Units'], refframe=self.pos.h['ReferenceFrame'])

        # Find index of current position:
        ts_ind = np.where(self['time'] < date2num(timestamp))[0][0]
        r = self.pos['r'][ts_ind]

        # Get longitude and latitude
        long_heeq = self.pos['lon'][ts_ind]*180./np.pi
        lat_heeq = self.pos['lat'][ts_ind]*180./np.pi

        # Define time lag from STEREO-A to Earth
        timelag_l1 = abs(long_heeq)/(360./sun_syn)
        arrival_time_l1 = date2num(timestamp) + timelag_l1
        arrival_time_l1_str = str(num2date(arrival_time_l1))

        satstr = ''
        satstr += '{} HEEQ longitude wrt Earth is {:.1f} degrees.\n'.format(self.source, long_heeq)
        satstr += 'This is {:.2f} times the location of L5.\n'.format(abs(long_heeq)/60.)
        satstr += '{} HEEQ latitude is {:.1f} degrees.\n'.format(self.source, lat_heeq)
        satstr += 'Earth L1 HEEQ latitude is {:.1f} degrees.\n'.format(L1Pos['lat']*180./np.pi,1)
        satstr += 'Difference HEEQ latitude is {:.1f} degrees.\n'.format(abs(lat_heeq-L1Pos['lat']*180./np.pi))
        satstr += '{} heliocentric distance is {:.3f} AU.\n'.format(self.source, r)
        satstr += 'The solar rotation period with respect to Earth is chosen as {:.2f} days.\n'.format(sun_syn)
        satstr += 'This is a time lag of {:.2f} days.\n'.format(timelag_l1)
        satstr += 'Arrival time of {} wind at L1: {}\n'.format(self.source, arrival_time_l1_str[0:16])

        return satstr

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
            new_inds = np.where(self.data[0] >= date2num(starttime))[0]
            self.data = self.data[:,new_inds]
            if self.pos != None:
                self.pos.positions = self.pos.positions[:,new_inds]
        elif starttime == None and endtime != None:
            new_inds = np.where(self.data[0] < date2num(endtime))[0]
            self.data = self.data[:,new_inds]
            if self.pos != None:
                self.pos.positions = self.pos.positions[:,new_inds]
        elif starttime != None and endtime != None:
            new_inds = np.where((self.data[0] >= date2num(starttime)) & (self.data[0] < date2num(endtime)))[0]
            self.data = self.data[:,new_inds]
            if self.pos != None:
                self.pos.positions = self.pos.positions[:,new_inds]
        return self


    def get_weighted_average(self, key, past_timesteps=4, past_weights=0.65):
        """
        Calculates a weighted average of speed and magnetic field bx, by, bz and the Newell coupling ec
        for a number of ave_hours (4 by default) back in time
        input data time resolution should be 1 hour
        aurora output time resolution as given by dt can be higher
        corresponds roughly to ap_inter_sol.pro in IDL ovation

        Parameters
        ==========
        self : ...
        key : str
            String of key to return average for.
        past_timesteps : int (default=4)
            Timesteps previous to integrate over, usually 4 (hours)
        past_weights : float (default=0.65)
            Reduce weights by factor with each hour back

        Returns
        =======
        avg : np.ndarray
            Array containing averaged values. Same length as original.
        """

        if key not in self.vars:
            raise Exception("Key {} not available in this ({}) SatData object!".format(key, self.source))

        avg = np.zeros((len(self)))

        for t_ind, timestep in enumerate(self.data[0]):
            weights = np.ones(past_timesteps)  #make array with weights
            for k in np.arange(1,past_timesteps,1):
                weights[k] = weights[k-1] * past_weights

            t_inds_for_weights = np.arange(t_ind, t_ind-past_timesteps,-1)
            t_inds_for_weights[t_inds_for_weights < 0] = 0

            #sum last hours with each weight and normalize
            avg[t_ind] = np.round(np.nansum(self[key][t_inds_for_weights]*weights) / np.nansum(weights),1)

        return avg


    def make_hourly_data(self):
        """Takes data with minute resolution and interpolates to hour.
        Uses .interp_to_time(). See that function for more usability options.

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
        # Roundabout way to get time_h ensures timings with full hours:
        nhours = (num2date(self['time'][-1])-num2date(stime)).total_seconds()/60./60.
        # Create new time array
        time_h = np.array(stime + np.arange(1, nhours)*(1./24.))
        Data_h = self.interp_to_time(time_h)

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

        if keys == None:
            keys = self.vars

        # Create new time array
        data_dict = {'time': tarray}
        for k in keys:
            na = np.interp(tarray, self['time'], self[k])
            data_dict[k] = na

        # Create new data opject:
        newData = SatData(data_dict, header=copy.deepcopy(self.h), source=copy.deepcopy(self.source))
        newData.h['SamplingRate'] = tarray[1] - tarray[0]
        # Interpolate position data:
        if self.pos != None:
            newPos = self.pos.interp_to_time(self['time'], tarray)
            newData.pos = newPos

        return newData


    def remove_icmes(self, spacecraft=None):
        """Replaces ICMES in data object with NaNs.
        ICMEs are automatically loaded using the HELCATS catalogue in the function
        get_icme_catalogue().

        NOTE: if you want to remove ICMEs from L1 data, set spacecraft='Wind'.

        Parameters
        ==========
        spacecraft : str (default=None)
            Specify spacecraft for ICMEs removal. If None, self.source is used.

        Returns
        =======
        self : obj with ICME periods removed
        """

        if spacecraft == None:
            spacecraft = self.source
        icmes = get_icme_catalogue(spacecraft=spacecraft, starttime=num2date(self['time'][0]), endtime=num2date(self['time'][-1]))

        if len(set(icmes['SC_INSITU'])) > 1:
            logger.warning("Using entire CME list! Variable 'spacecraft' was not defined correctly. Options={}".format(set(icmes['SC_INSITU'])))

        for i in icmes: 
            if spacecraft == 'Wind':
                icme_inds = np.where(np.logical_and(self['time'] > i['ICME_START_TIME'], self['time'] < i['ICME_END_TIME']))
            else:
                icme_inds = np.where(np.logical_and(self['time'] > i['ICME_START_TIME'], self['time'] < i['MO_END_TIME']))
            self.data[1:,icme_inds] = np.nan

        if self['time'][0] < date2num(datetime(2007,1,1)):
            logger.warning("ICMES have only been removed after 2007-01-01. There may be ICMEs before this date unaccounted for!")
        if self['time'][-1] > date2num(datetime(2016,1,1)):
            logger.warning("ICMES have only been removed until 2016-01-01. There may be ICMEs after this date unaccounted for!")

        return self


    def remove_nans(self, key=''):
        """Removes nans from data object.

        Parameters
        ==========
        key : str (optional, default=self.vars[0])
            Key for variable to be used for picking out NaNs.
            If multiple variables, call function for each variable.

        Returns
        =======
        self : obj with nans removed
        """

        if key == '':
            key = self.vars[0]
        key_ind = self.default_keys.index(key)
        self.data = self.data[:,~np.isnan(self.data[key_ind])]

        return self


    def remove_times(self, start_remove, end_remove):
        """Removes data within period given by starttime and endtime.

        Parameters
        ==========
        start_remove : datetime.datetime object
            Start time (>=) of new array.
        end_remove : datetime.datetime object
            End time (<) of new array.

        Returns
        =======
        newData : new obj with time range removed
        """

        before = self.data[:, self.data[0] < date2num(start_remove)]
        after = self.data[:, self.data[0] > date2num(end_remove)]

        new = np.hstack((before, after))

        newData = SatData({'time': [1,2,3], 'bz': [1,2,3]}, header=copy.deepcopy(self.h), source=copy.deepcopy(self.source))
        newData.data = new
        newData.pos = copy.deepcopy(self.pos)
        newData.state = copy.deepcopy(self.state)
        newData.vars = copy.deepcopy(self.vars)
        newData.h['RemovedTimes'].append("{}--{}".format(start_remove.strftime("%Y-%m-%dT%H:%M:%S"), 
                                                         end_remove.strftime("%Y-%m-%dT%H:%M:%S")))

        return newData


    def shift_time_to_L1(self, sun_syn=26.24, method='new'):
        """Shifts the time variable to roughly correspond to solar wind at L1 using a
        correction for timing for the Parker spiral.
        See Simunac et al. 2009 Ann. Geophys. equation 1 and Thomas et al. 2018 Space Weather,
        difference in heliocentric distance STEREO-A to Earth. The value is actually different 
        for every point but can take average of solar wind speed (method='old').
        Omega is 360 deg/sun_syn in days, convert to seconds; sta_r in AU to m to km;
        convert to degrees
        minus sign: from STEREO-A to Earth the diff_r_deg needs to be positive
        because the spiral leads to a later arrival of the solar wind at larger
        heliocentric distances (this is reverse for STEREO-B!)

        Parameters
        ==========
        sun_syn : float
            Sun synodic rotation in days.
        method : str (default='new')
            Method to be used. 'old' means average of time diff is added, 'new' means full
            array of time values is added to original time array.

        Returns
        =======
        self
        """

        if method == 'old':
            lag_l1, lag_r = get_time_lag_wrt_earth(satname=self.source,
                timestamp=num2date(self['time'][-1]),
                v_mean=np.nanmean(self['speed']), sun_syn=sun_syn)
            logger.info("shift_time_to_L1: Shifting time by {:.2f} hours".format((lag_l1 + lag_r)*24.))
            self.data[0] = self.data[0] + lag_l1 + lag_r

        elif method == 'new':
            if self.pos == None:
                logger.warning("Loading position data (SatData.load_positions()) for shift_time_to_L1()!")
                self.load_positions()
            dttime = [num2date(t).replace(tzinfo=None) for t in self['time']]
            L1Pos = get_l1_position(dttime, units=self.pos.h['Units'], refframe=self.pos.h['ReferenceFrame'])
            L1_r = L1Pos['r']
            timelag_diff_r = np.zeros(len(L1_r))

            # define time lag from satellite to Earth
            timelag_L1 = abs(self.pos['lon']*180/np.pi)/(360/sun_syn) #days

            # Go through all data points
            for i in np.arange(0,len(L1_r),1):
                if self.pos.h['Units'] == 'AU':
                    sat_r = self.pos['r'][i]*AU
                    l1_r = L1_r[i]*AU
                elif self.pos.h['Units'] == 'm':
                    sat_r = self.pos['r'][i]/1000.
                    l1_r = L1_r[i]/1000.
                else:
                    sat_r = self.pos['r'][i]
                    l1_r = L1_r[i]
                # Thomas et al. (2018): angular speed of rotation of sun * radial diff/speed
                # note: dimensions in seconds
                diff_r_deg = (360/(sun_syn*86400))*(l1_r - sat_r)/self['speed'][i]
                # From lon diff, calculate time by dividing  by rotation speed (in days)
                timelag_diff_r[i] = np.round(diff_r_deg/(360/sun_syn),3)

            ## ADD BOTH time shifts to the stbh_t
            self.data[0] = self.data[0] + timelag_L1 + timelag_diff_r
            logger.info("shift_time_to_L1: Shifting time by {:.1f}-{:.1f} hours".format(
                (timelag_L1+timelag_diff_r)[0]*24., (timelag_L1+timelag_diff_r)[-1]*24.))

        return self


    def shift_wind_to_L1(self):
        """Corrects for differences in B and density values due to solar wind
        expansion at different radii.
        
        Exponents taken from Kivelson and Russell, Introduction to Space Physics (Ch. 4.3.2).
        Density, btot, bt, bn, bx and bz are scaled according to 1/r**2.
        bt and by are scaled according to 1/r.

        Parameters
        ==========
        None

        Returns
        =======
        self
        """

        dttime = [num2date(t).replace(tzinfo=None) for t in self['time']]
        L1Pos = get_l1_position(dttime, units=self.pos.h['Units'], refframe=self.pos.h['ReferenceFrame'])

        shift_vars_2 = ['density', 'btot', 'bt', 'bn', 'bx', 'bz']      # behave according to inverse-square law
        shift_vars = [v for v in shift_vars_2 if v in self.vars]
        for var in shift_vars:
            self[var] = self[var] * (self.pos['r']/L1Pos['r'])**(-2.)
        
        shift_vars_1 = ['bt', 'by']
        shift_vars = [v for v in shift_vars_1 if v in self.vars]      # behave according to 1/r
        for var in shift_vars:
            self[var] = self[var] * (self.pos['r']/L1Pos['r'])**(-1.)
        logger.info("shift_wind_to_L1: Extrapolated B and density values to L1 distance")

        return self


    # -----------------------------------------------------------------------------------
    # Index calculations and predictions
    # -----------------------------------------------------------------------------------

    def extract_local_time_variables(self):
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

        dtime = num2date(self['time'])
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

        logger.info("Making auroral power prediction")
        aurora_power = np.round(make_aurora_power_from_wind(self['btot'], self['by'], self['bz'], self['speed'], self['density']), 2)
        #make sure that no values are < 0
        aurora_power[np.where(aurora_power < 0)]=0.0

        auroraData = SatData({'time': self['time'], 'aurora': aurora_power})
        auroraData.h['DataSource'] = "Auroral power prediction from {} data".format(self.source)
        auroraData.h['SamplingRate'] = 1./24.

        return auroraData


    def make_dst_prediction(self, method='temerin_li', t_correction=False):
        """Makes prediction with data in array.

        Parameters
        ==========
        method : str
            Options = ['burton', 'obrien', 'temerin_li', 'temerin_li_2006']
        t_correction : bool
            For TL-2006 method only. Add a time-dependent linear correction to
            Dst values (required for anything beyond 2002).

        Returns
        =======
        dstData : new SatData obj
            New object containing predicted Dst data.
        """

        if method.lower() == 'temerin_li':
            if 'speedx' in self.vars:
                vx = self['speedx']
            else:
                vx = self['speed']
            logger.info("Calculating Dst for {} using Temerin-Li model 2002 version (updated parameters)".format(self.source))
            dst_pred = calc_dst_temerin_li(self['time'], self['btot'], self['bx'], self['by'], self['bz'], self['speed'], vx, self['density'], version='2002n')
        elif method.lower() == 'temerin_li_2002':
            if 'speedx' in self.vars:
                vx = self['speedx']
            else:
                vx = self['speed']
            logger.info("Calculating Dst for {} using Temerin-Li model 2002 version".format(self.source))
            dst_pred = calc_dst_temerin_li(self['time'], self['btot'], self['bx'], self['by'], self['bz'], self['speed'], vx, self['density'], version='2002')
        elif method.lower() == 'temerin_li_2006':
            if 'speedx' in self.vars:
                vx = self['speedx']
            else:
                vx = self['speed']
            logger.info("Calculating Dst for {} using Temerin-Li model 2006 version".format(self.source))
            dst_pred = calc_dst_temerin_li(self['time'], self['btot'], self['bx'], self['by'], self['bz'], self['speed'], vx, self['density'], 
                                           version='2006', linear_t_correction=t_correction)
        elif method.lower() == 'obrien':
            logger.info("Calculating Dst for {} using OBrien model".format(self.source))
            dst_pred = calc_dst_obrien(self['time'], self['bz'], self['speed'], self['density'])
        elif method.lower() == 'burton':
            logger.info("Calculating Dst for {} using Burton model".format(self.source))
            dst_pred = calc_dst_burton(self['time'], self['bz'], self['speed'], self['density'])

        dstData = SatData({'time': copy.deepcopy(self['time']), 'dst': dst_pred})
        dstData.h['DataSource'] = "Dst prediction from {} data using {} method".format(self.source, method)
        dstData.h['SamplingRate'] = 1./24.

        return dstData


    def make_dst_prediction_from_model(self, model):
        """Makes prediction of Dst from previously trained machine learning model
        with data in array.

        Parameters
        ==========
        method : sklearn/keras model
            Trained model with predict() method.

        Returns
        =======
        dstData : new SatData obj
            New object containing predicted Dst data.
        """

        logger.info("Making Dst prediction for {} using machine learning model".format(self.source))
        dst_pred = model.predict(self.data.T)

        dstData = SatData({'time': self['time'], 'dst': dst_pred})
        dstData.h['DataSource'] = "Dst prediction from {} data using ML model".format(self.source)
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

        logger.info("Making kp prediction")
        kp_pred = np.round(make_kp_from_wind(self['btot'], self['by'], self['bz'], self['speed'], self['density']), 1)

        kpData = SatData({'time': self['time'], 'kp': kp_pred})
        kpData.h['DataSource'] = "Kp prediction from {} data".format(self.source)
        kpData.h['SamplingRate'] = 1./24.

        return kpData


    # -----------------------------------------------------------------------------------
    # Definition of state
    # -----------------------------------------------------------------------------------

    def get_state(self):
        """Finds state of wind and fills self.state attribute."""

        print("Coming soon.")

    # -----------------------------------------------------------------------------------
    # Data archiving
    # -----------------------------------------------------------------------------------

    def archive(self):
        """Make archive of long-term data."""

        print("Not yet implemented.")


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

    # -----------------------------------------------------------------------------------
    # Internal methods
    # -----------------------------------------------------------------------------------

    def __init__(self, posdata, postype, header=None):
        """Create new instance of class."""

        if not postype.lower() in ['xyz', 'rlonlat']:
            raise Exception("PositionData __init__: postype must be either 'xyz' or 'rlonlat'!")
        self.positions = np.asarray(posdata)
        if header == None:               # Inititalise empty header
            self.h = copy.deepcopy(PositionData.empty_header)
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

    # -----------------------------------------------------------------------------------
    # Object data handling
    # -----------------------------------------------------------------------------------

    def interp_to_time(self, t_orig, t_new):
        """Linearly interpolates over nans in array.

        Parameters
        ==========
        t_orig : np.ndarray
            Array containing original timesteps.
        t_new : np.ndarray
            Array containing new timesteps.
        """

        na = []
        for k in self.coors:
            na.append(np.interp(t_new, t_orig, self[k]))

        # Create new data opject:
        newData = PositionData(na, copy.deepcopy(self.h['CoordinateSystem']), header=copy.deepcopy(self.h))

        return newData


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
        jd=astropy.time.Time(num2date(timegse[i]), format='datetime', scale='utc').jd
        mjd[i]=float(int(jd-2400000.5)) #use modified julian date    
        T00=(mjd[i]-51544.5)/36525.0
        dobj=num2date(timegse[i])
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
        jd[i]=astropy.time.Time(num2date(ctime[i]), format='datetime', scale='utc').jd
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

    if timestamp == None:
        timestamp = datetime.utcnow()

    if satname == 'STEREO-A':
        SAT = heliosat.STA()
    elif satname == 'STEREO-B':
        SAT = heliosat.STB()
    else:
        raise Exception("Not a valid satellite name to find position!")

    traj = SAT.trajectory([timestamp], frame='HEEQ', units='AU',
                          observer='SUN')

    posx, posy, posz = traj[:,0][0], traj[:,1][0], traj[:,2][0]
    sat_r, lat_heeq, lon_heeq = cart2sphere(posx, posy, posz)
    earth_r = 1.

    # define time lag from satellite to Earth
    lag_l1 = abs(lon_heeq*180/np.pi)/(360./sun_syn)

    # time lag from Parker spiral
    diff_r_deg = -(360./(sun_syn*86400.))*((sat_r-earth_r)*AU)/v_mean
    lag_diff_r = round(diff_r_deg/(360./sun_syn),2)

    return lag_l1, lag_diff_r


def cart2sphere(x, y, z):
    """convert cartesian to spherical coordinates
    theta = polar angle/elevation angle = latitude
    phi = azimuthal angle = longitude
    Returns (r, theta, phi)
    """
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
# B. Data reading and writing:
# ***************************************************************************************


def get_dscovr_data(starttime, endtime, resolution='min', skip_files=True):

    if (datetime.utcnow() - starttime).days < 7.:
        dscovr_data = ps.get_dscovr_realtime_data()
        dscovr_data = dscovr_data.cut(starttime=starttime, endtime=endtime)
        return dscovr_data
    else:
        dscovr_data = get_dscovr_archive_data(starttime, endtime, 
                            resolution=resolution, skip_files=skip_files)
        return dscovr_data


def get_dscovr_archive_data(starttime, endtime, resolution='min', skip_files=True):
    """Downloads and reads STEREO-A beacon data from CDF files. Files handling
    is done using heliosat, so files are downloaded to HELIOSAT_DATAPATH.
    Data sourced from: 
    https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/1542848400000;1554163200000/f1m;m1m

    Parameters
    ==========
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.
    resolution : str, (optional, 'min' (default) or 'hour')
        Determines which resolution data should be returned in.
    skip_files : bool (default=True)
        Heliosat get_data_raw var. Skips missing files in download folder.

    Returns
    =======
    dscovr : predstorm.SatData
        Object containing satellite data under keys.

    """

    logger = logging.getLogger(__name__)

    DSCOVR_ = heliosat.DSCOVR()

    logger.info("Reading archived DSCOVR data")

    # Magnetometer data
    magt, magdata = DSCOVR_.get_data_raw(starttime, endtime, 'mag', skip_files=skip_files)
    magt = [datetime.utcfromtimestamp(t) for t in magt]
    bx, by, bz = magdata[:,0], magdata[:,1], magdata[:,2]
    missing_value = -99999.
    bx[bx==missing_value] = np.NaN
    by[by==missing_value] = np.NaN
    bz[bz==missing_value] = np.NaN
    btot = np.sqrt(bx**2. + by**2. + bz**2.)

    if len(bx) == 0:
        logger.error("DSCOVR data is missing or masked in time range! Returning empty data object.")
        return SatData({'time': []})

    # Particle data
    pt, pdata = DSCOVR_.get_data_raw(starttime, endtime, 'proton', skip_files=skip_files)
    pt = [datetime.utcfromtimestamp(t) for t in pt]
    density, vtot, temperature = pdata[:,0], pdata[:,1], pdata[:,2]
    density[density==missing_value] = np.NaN
    vtot[vtot==missing_value] = np.NaN
    temperature[temperature==missing_value] = np.NaN

    if resolution == 'hour':
        stime = date2num(starttime) - date2num(starttime)%(1./24.)
        nhours = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60./60.
        tarray = np.array(stime + np.arange(0, nhours)*(1./24.))
    elif resolution == 'min':
        stime = date2num(starttime) - date2num(starttime)%(1./24./60.)
        nmins = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60.
        tarray = np.array(stime + np.arange(0, nmins)*(1./24./60.))

    # Interpolate variables to time:
    bx_int = np.interp(tarray, date2num(magt), bx)
    by_int = np.interp(tarray, date2num(magt), by)
    bz_int = np.interp(tarray, date2num(magt), bz)
    btot_int = np.interp(tarray, date2num(magt), btot)
    density_int = np.interp(tarray, date2num(pt), density)
    vtot_int = np.interp(tarray, date2num(pt), vtot)
    temp_int = np.interp(tarray, date2num(pt), temperature)

    # Pack into object:
    dscovr = SatData({'time': tarray,
                      'btot': btot_int, 'bx': bx_int, 'by': by_int, 'bz': bz_int,
                      'speed': vtot_int, 'density': density_int, 'temp': temp_int},
                      source='DSCOVR')
    dscovr.h['DataSource'] = "DSCOVR Level 1 (NOAA)"
    dscovr.h['SamplingRate'] = tarray[1] - tarray[0]
    dscovr.h['ReferenceFrame'] = DSCOVR_.spacecraft["data"]['mag'].get("frame")
    dscovr.h['HeliosatObject'] = DSCOVR_

    return dscovr


def get_dscovr_realtime_data():
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
    dscovr_data : ps.SatData object
        Object containing DSCOVR data under standard keys.
    """
    
    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    # Read plasma data:
    with urllib.request.urlopen(url_plasma) as url:
        dp = json.loads (url.read().decode())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dpn[1:]]
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dpn[1:])] 
        DSCOVR_P = np.array(dp_, dtype=dtype)
    # Read magnetic field data:
    with urllib.request.urlopen(url_mag) as url:
        dm = json.loads(url.read().decode())
        dmn = [[np.nan if x == None else x for x in d] for d in dm]     # Replace None w NaN
        dtype=[(x, 'float') for x in dmn[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dmn[1:]]
        dm_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dm[1:])] 
        DSCOVR_M = np.array(dm_, dtype=dtype)

    last_timestep = np.min([DSCOVR_M['time_tag'][-1], DSCOVR_P['time_tag'][-1]])
    first_timestep = np.max([DSCOVR_M['time_tag'][0], DSCOVR_P['time_tag'][0]])

    nminutes = int((num2date(last_timestep)-num2date(first_timestep)).total_seconds()/60.)
    itime = np.asarray([date2num(num2date(first_timestep) + timedelta(minutes=i)) for i in range(nminutes)], dtype=np.float64)

    rbtot_m = np.interp(itime, DSCOVR_M['time_tag'], DSCOVR_M['bt'])
    rbxgsm_m = np.interp(itime, DSCOVR_M['time_tag'], DSCOVR_M['bx_gsm'])
    rbygsm_m = np.interp(itime, DSCOVR_M['time_tag'], DSCOVR_M['by_gsm'])
    rbzgsm_m = np.interp(itime, DSCOVR_M['time_tag'], DSCOVR_M['bz_gsm'])
    rpv_m = np.interp(itime, DSCOVR_P['time_tag'], DSCOVR_P['speed'])
    rpn_m = np.interp(itime, DSCOVR_P['time_tag'], DSCOVR_P['density'])
    rpt_m = np.interp(itime, DSCOVR_P['time_tag'], DSCOVR_P['temperature'])

    # Pack into object
    dscovr_data = SatData({'time': itime,
                           'btot': rbtot_m, 'bx': rbxgsm_m, 'by': rbygsm_m, 'bz': rbzgsm_m,
                           'speed': rpv_m, 'density': rpn_m, 'temp': rpt_m},
                           source='DSCOVR')
    dscovr_data.h['DataSource'] = "DSCOVR (NOAA)"
    dscovr_data.h['SamplingRate'] = 1./24./60.
    dscovr_data.h['ReferenceFrame'] = 'GSM'
    DSCOVR_ = heliosat.DSCOVR()
    dscovr_data.h['HeliosatObject'] = DSCOVR_
    
    logger.info('get_dscovr_data_real: DSCOVR data read completed.')
    
    return dscovr_data


def get_icme_catalogue(filepath=None, spacecraft=None, starttime=None, endtime=None):
    """Downloads and returns the HELCATS ICME catalogue. 
    See https://www.helcats-fp7.eu/catalogues/wp4_icmecat.html for info.
    Details:

    ICME Catalogue
    This is the HELCATS interplanetary coronal mass ejection catalog, based on 
    magnetometer and plasma observations in the heliosphere. It is a product of 
    working package 4 of the EU HELCATS project (2014-2017).

    Number of events in ICMECAT: 668
    ICME observatories: Wind, STEREO-A, STEREO-B, VEX, MESSENGER, ULYSSES
    Time range: January 2007 - December 2015.

    This is version: 01 of the catalogue, released 2017-02-28.

    Parameters
    ==========
    filepath : str (default=None)
        If given, will read from ASCII file e.g. HELCATS_ICMECAT_v10_SCEQ.txt
    spacecraft : str (default=None)
        If given, data for single spacecraft will be returned.
    starttime : datetime.datetime object
        Start time (>=) of period with ICMEs.
    endtime : datetime.datetime object
        End time (<) of period with ICMEs.

    Returns
    =======
    icmes : np.array
        Array containing all ICME data. See icmes.dtype for keys.
    """

    icme_list_keys = {
        "ICMECAT_ID": "The unique identifier for the observed ICME. unit: string.",
        "SC_INSITU": "The name of the in situ observatory. unit: string.",
        "ICME_START_TIME": "The shock arrival or density enhancement time, can be similar to MO_START_TIME. unit: UTC.",
        "MO_START_TIME": "The start time of the magnetic obstacle (MO), including flux ropes, flux-rope-like, and ejecta signatures. unit: UTC.",
        "MO_END_TIME": "The end time of the magnetic obstacle. unit: UTC.",
        "ICME_END_TIME": "The end time of the ICME, can be similar to MO_END_TIME. unit: UTC.",
        "MO_BMAX": "The maximum total magnetic field in the magnetic obstacle. unit: nT.",
        "MO_BMEAN": "The mean total magnetic field of the magnetic obstacle. unit: nT.",
        "MO_BSTD": "The standard deviation of the total magnetic field of the magnetic obstacle. unit: nT.",
        "MO_BZMEAN": "The mean magnetic field Bz component in the magnetic obstacle. unit: nT.",
        "MO_BZMIN": "The minimum magnetic field Bz component of the magnetic obstacle. unit: nT.",
        "MO_DURATION":"Duration of interval between MO_START_TIME and MO_END_TIME. unit: hours.",
        "SC_HELIODISTANCE": "Average heliocentric distance of the spacecraft during the MO. unit: AU.",
        "SC_LONG_HEEQ": "Average heliospheric longitude of the spacecraft during the MO, range [-180,180]. unit: degree (HEEQ).",
        "SC_LAT_HEEQ": "Average heliospheric latitude of the spacecraft during the MO, range [-90,90]. unit: degree (HEEQ).",
        "MO_MVA_AXIS_LONG": "Longitude of axis from minimum variance analysis with magnetic field unit vectors (MVA): X=0 deg, Y=90 deg, range [0,360]. unit: degree (SCEQ).",
        "MO_MVA_AXIS_LAT": "Latitude of axis from MVA, +Z=-90 deg, -Z=-90, range [-90,90]. unit: degree (SCEQ).",
        "MO_MVA_RATIO": "Eigenvalue 2 over 3 ratio as indicator of reliability of MVA, must be > 2, otherwise NaN. unit: number.",
        "SHEATH_SPEED": "For STEREO-A/B and Wind, average proton speed from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: km/s.",
        "SHEATH_SPEED_STD": "For STEREO-A/B and Wind, standard deviation of proton speed from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: km/s.",
        "MO_SPEED": "For STEREO-A/B and Wind, average proton speed from MO_START_TIME to MO_END_TIME. unit: km/s.",
        "MO_SPEED_STD": "For STEREO-A/B and Wind, standard deviation of proton speed from MO_START_TIME to MO_END_TIME. unit: km/s.",
        "SHEATH_DENSITY": "For STEREO-A/B and Wind, average proton density from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: ccm^-3.",
        "SHEATH_DENSITY_STD": "For STEREO-A/B and Wind, standard deviation of proton density from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: cm^-3.",
        "MO_DENSITY": "For STEREO-A/B and Wind, average proton density from MO_START_TIME to MO_END_TIME. unit: cm^-3.",
        "MO_DENSITY_STD": "For STEREO-A/B and Wind, standard deviation of proton density from MO_START_TIME to MO_END_TIME. unit: cm^-3.",
        "SHEATH_TEMPERATURE": "For STEREO-A/B and Wind, average proton temperature from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: K.",
        "SHEATH_TEMPERATURE_STD": "For STEREO-A/B and Wind, standard deviation of proton temperature from ICME_START_TIME to MO_START_TIME, NaN if these times are similar. unit: K.",
        "MO_TEMPERATURE": "For STEREO-A/B and Wind, average proton temperature from MO_START_TIME to MO_END_TIME. unit: K.",
        "MO_TEMPERATURE_STD": "For STEREO-A/B and Wind, standard deviation of proton temperature from MO_START_TIME to MO_END_TIME. unit: K."
        }

    dt = [(s, 'object') if s in ['ICMECAT_ID', 'SC_INSITU'] else (s, 'float') for s in icme_list_keys.keys()]

    if filepath != None:
        # Read from file:
        icme_array = np.genfromtxt(filepath, dtype='str')
    else:
        url_helcats = "https://www.helcats-fp7.eu/catalogues/data/ICME_WP4_V10.json"
        # Read JSON from website:
        with urllib.request.urlopen(url_helcats) as url:
            icme_list = json.loads(url.read().decode())
        # Pack into array for easy handling:
        icme_array = np.array(icme_list['data'])

    cols = {}
    # Convert all data to relevant format:
    for i, col in enumerate(icme_list_keys.keys()):
        if col in ['ICMECAT_ID', 'SC_INSITU']:
            cols[col] = icme_array[:,i]
        elif 'TIME' in col:
            cols[col] = [date2num(datetime.strptime(x, "%Y-%m-%dT%H:%MZ")) if x[0] != '9' else None for x in icme_array[:,i]]
        else:
            cols[col] = [float(x) for x in icme_array[:,i]]
    alldata = [cols[x] for x in cols.keys()]
    # Pack into structure array:
    icmes = np.array(list(zip(*alldata)), dtype=dt)

    # Return only data for one spacecraft:
    if spacecraft != None:
        if spacecraft in set(icmes['SC_INSITU']):
            icmes = icmes[icmes['SC_INSITU']==spacecraft]
        else:
            logger.warning("{} is not a valid spacecraft! Returning all ICME data.")

    if starttime != None:
        icmes = icmes[icmes['ICME_START_TIME'] >= date2num(starttime)]
    if endtime != None:
        if spacecraft == 'Wind':
            icmes = icmes[icmes['ICME_END_TIME'] < date2num(endtime)]
        else:
            icmes = icmes[icmes['MO_END_TIME'] < date2num(endtime)] 

    return icmes


def get_l1_position(times, units='AU', refframe='HEEQ', observer='SUN'):
    """Uses Heliosat to return a position object for L1 with the provided times.

    Parameters
    ==========
    times : datetime (list/value)
        Array of times to return position data for.
    refframe : str (default=='HEEQ')
        observer reference frame
    observer : str (default='SUN')
        observer body name
    units : str (default='AU')
        output units - m / km / AU

    Returns
    =======
    L1Pos : PositionData object
        Object containing arrays of time and dst values.
    """

    Earth = heliosat._SpiceObject(None, "EARTH")
    Earth_traj = Earth.trajectory(times, frame=refframe, units=units, observer=observer)
    if type(times) == list:
        earth_r, elon, elat = Earth_traj[:,0], Earth_traj[:,1], Earth_traj[:,2]
    else:
        earth_r, elon, elat = Earth_traj[0], Earth_traj[1], Earth_traj[2]
    l1_r = earth_r - dist_to_L1/AU
    L1Pos = PositionData([l1_r, elon, elat], 'rlonlat')
    L1Pos.h['Units'] = units

    return L1Pos


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
        rdst_time[k]=date2num(datetime.strptime(rdst_time_str[k], "%Y-%m-%d %H:%M"))
    logger.info("NOAA real-time Dst data loaded.")

    dst_data = SatData({'time': rdst_time, 'dst': rdst},
                       source='KyotoDst')
    dst_data.h['DataSource'] = "Kyoto Dst (NOAA)"
    dst_data.h['SamplingRate'] = 1./24.

    return dst_data


def get_past_dst(filepath=None, starttime=None, endtime=None):
    """Will read Dst values from IAGA2002-format file. Data can be 
    downloaded from this webpage (in IAGA format):
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
    dst_time = np.array([date2num(datetime.strptime(d[0]+d[1], "%Y-%m-%d%H:%M:%S.%f")) for d in datastr])
    dst = np.array([float(d[-1]) for d in datastr])

    # Make sure no bad data is included:
    dst_time = dst_time[dst < 99999]
    dst = dst[dst < 99999]

    dst_data = SatData({'time': dst_time, 'dst': dst},
                       source='KyotoDst')
    dst_data.h['DataSource'] = "Kyoto Dst (Kyoto WDC)"
    dst_data.h['SamplingRate'] = 1./24.
    dst_data = dst_data.cut(starttime=starttime, endtime=endtime)
    
    return dst_data


def get_omni_data(starttime=None, endtime=None, filepath='', download=False, dldir='data'):
    """
    Reads OMNI2 .dat format data
    Will download and read OMNI2 data file (in .dat format).
    Variable definitions from OMNI2 dataset:
    see http://omniweb.gsfc.nasa.gov/html/ow_data.html

    FORMAT(2I4,I3,I5,2I3,2I4,14F6.1,F9.0,F6.1,F6.0,2F6.1,F6.3,F6.2, F9.0,F6.1,F6.0,2F6.1,F6.3,2F7.2,F6.1,I3,I4,I6,I5,F10.2,5F9.2,I3,I4,2F6.1,2I6,F5.1)
    1963   1  0 1771 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999. 999.9 9999. 999.9 999.9 9.999 99.99 9999999. 999.9 9999. 999.9 999.9 9.999 999.99 999.99 999.9  7  23    -6  119 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 999.9 999.9 99999 99999 99.9

    Parameters
    ==========
    starttime : datetime.datetime
        Start time of period of data to return.
    endtime : datetime.datetime
        End time of period of data to return.
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

    if download == False:
        if filepath != '' and not os.path.exists(filepath):
            raise Exception("get_omni_data: {} does not exist! Run get_omni_data(download=True) to download file.".format(filepath))
        if filepath == '':
            raise Exception("get_omni_data: no data source specified! Run get_omni_data(download=True) or get_omni_data(filepath=path) to specify source.".format(filepath))

    if download:
        omni2_url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_all_years.dat'
        logger.info("get_omni_data: downloading OMNI2 data from {}".format(omni2_url))
        tofile = os.path.join(dldir, 'omni2_all_years.dat')
        try: 
            urllib.request.urlretrieve(omni2_url, tofile)
            logger.info("get_omni_data: OMNI2 data successfully downloaded.")
            filepath = tofile
        except urllib.error.URLError as e:
            logger.error("get_omni_data: OMNI2 data download failed (reason: {})".format(e.reason))

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
    ae=np.zeros(dataset) #float
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
            ae[j]=line[41]
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
        times1[index] = date2num(timedum)

    omni_data = SatData({'time': times1,
                         'btot': btot, 'bx': bx, 'by': bygsm, 'bz': bzgsm,
                         'speed': speed, 'speedx': speedx, 'density': den, 'pdyn': pdyn,
                         'dst': dst, 'kp': kp, 'ae': ae},
                         source='OMNI')
    omni_data.h['DataSource'] = "OMNI (NASA OMNI2 data)"
    if download:
        omni_data.h['SourceURL'] = omni2_url
    omni_data.h['SamplingRate'] = times1[1] - times1[0]
    omni_data.h['ReferenceFrame'] = 'GSM'
    omni_data.h['HeliosatObject'] = heliosat._SpiceObject(None, "EARTH")

    if starttime != None or endtime != None:
        omni_data = omni_data.cut(starttime=starttime, endtime=endtime)
    
    return omni_data


def get_omni_data_new(starttime=None, endtime=None, filepath='', dldir='data'):
    """
    Downloads and read OMNI2 data files (in yearly .dat format).
    Variable definitions from OMNI2 dataset:
    see http://omniweb.gsfc.nasa.gov/html/ow_data.html

    Parameters
    ==========
    starttime : datetime.datetime
        Start time of period of data to return.
    endtime : datetime.datetime
        End time of period of data to return.
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

    startyear = starttime.strftime("%Y")
    endyear = endtime.strftime("%Y")
    dlyears = np.arange(int(startyear), int(endyear)+1, 1)
    omni_data_url = 'https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/'

    for dlyear in dlyears:
        omni2_url = omni_data_url+'omni2_{}.dat'.format(dlyear)
        logger.info("get_omni_data: retrieving OMNI2 data from {}".format(omni2_url))
        try: 
            omni_data_part = np.genfromtxt(omni2_url)
            if dlyear != dlyears[0]:
                omni_data_raw = np.hstack((omni_data_raw, omni_data_part))
            else:
                omni_data_raw = omni_data_part
        except Exception as e:
            logger.error("get_omni_data: OMNI2 data download failed (reason: {})".format(e))

    # Time variables:
    # ---------------
    year, day, hour = omni_data_raw[:,0], omni_data_raw[:,1], omni_data_raw[:,2]

    # Plasma variables:
    # -----------------
    speed = omni_data_raw[:,24]   # bulk speed
    speed[speed == 9999] = np.NaN
    speed_phi = omni_data_raw[:,25]   # speed angle phi
    speed_phi[speed_phi == 999.9] = np.NaN
    speed_theta = omni_data_raw[:,26] # speed angle theta
    speed_theta[speed_theta == 999.9] = np.NaN
    # Convert speed to GSE x see OMNI website footnote
    speedx = (-speed) * np.cos(np.radians(speed_theta)) * np.cos(np.radians(speed_phi))
    den = omni_data_raw[:,23] # proton density /ccm
    den[den == 999.9] = np.NaN
    pdyn = omni_data_raw[:,28] # pdyn in nPa
    pdyn[pdyn == 99.99] = np.NaN

    # Magnetic field variables:
    # -------------------------
    btot = omni_data_raw[:,9] # total field in nT
    btot[btot == 999.9] = np.NaN

    # GSE components in nT
    bx = omni_data_raw[:,12]
    bx[bx == 999.9] = np.NaN
    by = omni_data_raw[:,13]
    by[by == 999.9] = np.NaN
    bz = omni_data_raw[:,14]
    bz[bz == 999.9] = np.NaN

    # GSM components in nT
    bygsm = omni_data_raw[:,15]
    bygsm[bygsm == 999.9] = np.NaN
    bzgsm = omni_data_raw[:,16]
    bzgsm[bzgsm == 999.9] = np.NaN

    # Indices:
    # --------
    kp = omni_data_raw[:,38]    # kp index
    kp[kp == 99] = np.NaN
    dst = omni_data_raw[:,40]   # dst index
    dst[dst == 99999] = np.NaN
    ae = omni_data_raw[:,41]    # ae index
    ae[ae == 9999] = np.NaN
    spot = omni_data_raw[:,39]  # sunspot number
    spot[spot == 999] = np.NaN

    #convert time to matplotlib format
    times = np.zeros(len(year))
    for index in range(0,len(year)):
        times[index] = date2num(datetime(int(year[index]), 1, 1) + timedelta(day[index] - 1) +
                       timedelta(hours=hour[index]))

    omni_data = SatData({'time': times,
                         'btot': btot, 'bx': bx, 'by': bygsm, 'bz': bzgsm,
                         'speed': speed, 'speedx': speedx, 'density': den, 'pdyn': pdyn,
                         'dst': dst, 'kp': kp, 'ae': ae},
                         source='OMNI')
    omni_data.h['DataSource'] = "OMNI (NASA OMNI2 data)"
    omni_data.h['SourceURL'] = omni_data_url
    omni_data.h['SamplingRate'] = times[1] - times[0]
    omni_data.h['ReferenceFrame'] = 'GSM'
    omni_data.h['HeliosatObject'] = heliosat._SpiceObject(None, "EARTH")

    if starttime != None or endtime != None:
        omni_data = omni_data.cut(starttime=starttime, endtime=endtime)

    return omni_data


def get_predstorm_realtime_data(resolution='hour'):
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
    pred_data.h['ReferenceFrame'] = 'GSM'

    return pred_data
    

def get_sdo_realtime_image():
    """Downloads latest SDO image."""

    logger = logging.getLogger(__name__)

    sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg'
    #PFSS
    #sdo_latest='https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193pfss.jpg'
    try: urllib.request.urlretrieve(sdo_latest,'latest_1024_0193.jpg')
    except urllib.error.URLError as e:
        logger.error('Failed downloading ', sdo_latest,' ',e)
    #convert to png
    #check if ffmpeg is available locally in the folder or systemwide
    if os.path.isfile('ffmpeg'):
        os.system('./ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
        ffmpeg_avail=True
        logger.info('downloaded SDO latest_1024_0193.jpg converted to png')
        os.system('rm latest_1024_0193.jpg')
    else:
        os.system('ffmpeg -i latest_1024_0193.jpg latest_1024_0193.png -loglevel quiet -y')
        os.system('rm latest_1024_0193.jpg')


def get_stereo_beacon_data(starttime, endtime, which_stereo='ahead', resolution='min', skip_files=True):
    """Downloads and reads STEREO-A beacon data from CDF files. Files handling
    is done using heliosat, so files are downloaded to HELIOSAT_DATAPATH.

    Notes: 
        - HGRTN != RTN: http://www.srl.caltech.edu/STEREO/docs/coordinate_systems.html
        - Data before 2009-09-14 is in HGRTN, RTN after that.

    Parameters
    ==========
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.
    which_stereo : str ('ahead'/'a' or 'behind'/'b')
        Which stereo satellite should be used.
    resolution : str, (optional, 'min' (default) or 'hour')
        Determines which resolution data should be returned in.
    skip_files : bool (default=True)
        Heliosat get_data_raw var. Skips missing files in download folder.

    Returns
    =======
    sta : predstorm.SatData
        Object containing satellite data.
    """

    logger = logging.getLogger(__name__)

    if which_stereo.lower() in ['a', 'ahead']:
        STEREO_ = heliosat.STA()
    elif which_stereo.lower() in ['b', 'behind']:
        STEREO_ = heliosat.STB()
    else:
        logger.error("{} is not a valid STEREO type! Use either 'ahead' or 'behind'.".format(which_stereo))

    logger.info("Reading STEREO-{} beacon data".format(which_stereo.upper()))

    # Magnetometer data
    magt_ts, magdata = STEREO_.get_data_raw(starttime, endtime, 'mag_beacon')
    # Convert time
    magt = []
    for t in magt_ts: 
        try: 
            magt.append(date2num(datetime.fromtimestamp(t)))
        except: 
            magt.append(np.nan)
    magt = np.array(magt)
    nantimes = np.isnan(magt)
    if len(nantimes) != 0:
        magt[nantimes] = np.interp(nantimes.nonzero()[0], (~nantimes).nonzero()[0], magt[~nantimes])
    #magt = [datetime.fromtimestamp(t) for t in magt]
    magdata[magdata < -1e20] = np.nan
    br, bt, bn = magdata[:,0], magdata[:,1], magdata[:,2]
    btot = np.sqrt(br**2. + bt**2. + bn**2.)

    # Particle data
    if starttime <= datetime(2009, 9, 13):
        if endtime > datetime(2009, 9, 13):
            pt_s, pdata_s = STEREO_.get_data_raw(starttime, datetime(2009, 9, 13, 23, 59, 00), 'proton_beacon',
                                              extra_columns=["Velocity_HGRTN:4"])
            pt_e, pdata_e = STEREO_.get_data_raw(datetime(2009, 9, 14), endtime, 'proton_beacon',
                                              extra_columns=["Velocity_RTN:4"])
            pt_ts = np.vstack((pt_s, pt_e))
            pdata = np.vstack((pdata_s, pdata_e))
        else:
            pt_ts, pdata = STEREO_.get_data_raw(starttime, endtime, 'proton_beacon',
                                              extra_columns=["Velocity_HGRTN:4"])
    else:
        pt_ts, pdata = STEREO_.get_data_raw(starttime, endtime, 'proton_beacon', extra_columns=["Velocity_RTN:4"])

    pt = np.array([datetime.fromtimestamp(t) for t in pt_ts])
    pt = date2num(pt)
    pdata[pdata < -1e29] = np.nan
    data_cols = STEREO_.spacecraft['data']['sta_plastic_beacon']['columns']
    density = pdata[:,0]
    temperature = pdata[:,1]
    vx, vtot = pdata[:,2], pdata[:,5]

    # Check plasma data is reasonable:
    integrity = 2   # integrity of 0 means data is very dodgy
    if np.nanmean(np.diff(pt)) > 0.003: # 0.00069444 is mean diff in timesteps for min data
        integrity -= 1
    if np.nanstd(vtot) > 300.:  # quiet times should be 70ish
        integrity -= 1

    stime = date2num(starttime) - date2num(starttime)%(1./24.)
    # Roundabout way to get time_h ensures timings with full hours/mins:
    if resolution == 'hour':
        nhours = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60./60.
        tarray = np.array(stime + np.arange(0, nhours)*(1./24.))
    elif resolution == 'min':
        nmins = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60.
        tarray = np.array(stime + np.arange(0, nmins)*(1./24./60.))

    # Interpolate variables to time:
    br_int = np.interp(tarray, magt, br)
    bt_int = np.interp(tarray, magt, bt)
    bn_int = np.interp(tarray, magt, bn)
    btot_int = np.interp(tarray, magt, btot)
    density_int = np.interp(tarray, pt, density)
    vx_int = np.interp(tarray, pt, vx)
    vtot_int = np.interp(tarray, pt, vtot)
    temp_int = np.interp(tarray, pt, temperature)

    # Pack into object:
    stereo = SatData({'time': tarray,
                   'btot': btot_int, 'br': br_int, 'bt': bt_int, 'bn': bn_int,
                   'speed': vtot_int, 'speedx': vx_int, 'density': density_int, 'temp': temp_int},
                   source='STEREO-A')
    stereo.h['DataSource'] = "STEREO-A Beacon"
    stereo.h['SamplingRate'] = tarray[1] - tarray[0]
    stereo.h['ReferenceFrame'] = STEREO_.spacecraft["data"]['sta_impact_beacon'].get("frame")
    stereo.h['HeliosatObject'] = STEREO_
    stereo.h['Instruments'] = ['PLASTIC', 'IMPACT']
    stereo.h['PlasmaDataIntegrity'] = integrity

    return stereo


def get_position_data(filepath, times, rlonlat=False, l1_corr=False):
    """Reads position data from pickled heliopy.spice.Trajectory into
    PositionData object.
    Note: Interpolates position data, which is faster but not as exact.

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
    posx = np.interp(times, date2num(posdata.times), posdata.x)
    posy = np.interp(times, date2num(posdata.times), posdata.y)
    posz = np.interp(times, date2num(posdata.times), posdata.z)


    if rlonlat:
        r, theta, phi = cart2sphere(posx, posy, posz)
        if l1_corr:
            r = r - dist_to_L1
        Positions = PositionData([r, phi, theta], 'rlonlat')
    else:
        if l1_corr:
            posx = posx - dist_to_L1
        Positions = PositionData([posx, posy, posz], 'xyz')
    Positions.h['Units'] = 'km'
    Positions.h['ReferenceFrame'] = refframe
    
    return Positions


def save_to_file(filepath, wind=None, dst=None, aurora=None, kp=None, ec=None):
    """Produces output in PREDSTORM realtime format."""

    out = np.zeros([np.size(wind['time']),17])

    #get date in ascii
    for i in np.arange(np.size(wind['time'])):
       time_dummy = num2date(wind['time'][i])
       out[i,0] = time_dummy.year
       out[i,1] = time_dummy.month
       out[i,2] = time_dummy.day
       out[i,3] = time_dummy.hour
       out[i,4] = time_dummy.minute
       out[i,5] = time_dummy.second

    out[:,6] = wind['time']
    out[:,7] = wind['btot']
    out[:,8] = wind['bx']
    out[:,9] = wind['by']
    out[:,10] = wind['bz']
    out[:,11] = wind['density']
    out[:,12] = wind['speed']
    out[:,13] = dst['dst']
    out[:,14] = kp['kp']
    out[:,15] = aurora['aurora']
    out[:,16] = ec['ec']/4421.

    #description
    column_vals = '{:>17}{:>16}{:>7}{:>7}{:>7}{:>7}{:>9}{:>9}{:>8}{:>7}{:>8}{:>12}'.format(
        'Y  m  d  H  M  S', 'matplotlib_time', 'B[nT]', 'Bx', 'By', 'Bz', 'N[ccm-3]', 'V[km/s]',
        'Dst[nT]', 'Kp', 'AP[GW]', 'Ec/4421[(km/s)**(4/3)nT**(2/3)]')
    time_cols_fmt = '%4i %2i %2i %2i %2i %2i %15.6f'
    b_cols_fmt = 4*'%7.2f'
    p_cols_fmt = '%9.0i%9.0i'
    indices_fmt = '%8.0f%7.2f%8.1f%12.1f'
    float_fmt = time_cols_fmt + b_cols_fmt + p_cols_fmt + indices_fmt
    np.savetxt(filepath, out, delimiter='',fmt=float_fmt, header=column_vals)

    return


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
    if np.abs((satdata1.h['SamplingRate'] - 1./24.)) < 0.0001:
        timestep = 1./24.
    elif np.abs((satdata1.h['SamplingRate'] - 1./24./60.)) < 0.0001:
        timestep = 1./24./60.
    else:
        timestep = satdata1.h['SamplingRate']
    n_timesteps = len(np.arange(satdata1['time'][-1] + timestep, satdata2['time'][-1], timestep))  
    # Make time array with matching steps
    new_time = np.array(satdata1['time'][-1] + np.arange(1, n_timesteps) * timestep)

    datadict = {}
    datadict['time'] = np.concatenate((satdata1['time'], new_time))
    for k in keys:
        # Interpolate dataset #2 to array matching dataset #1
        int_var = np.interp(new_time, satdata2['time'], satdata2[k])
        # Make combined array data
        datadict[k] = np.concatenate((satdata1[k], int_var))

    MergedData = SatData(datadict, source=satdata1.source+'+'+satdata2.source)
    tf = "%Y-%m-%d %H:%M:%S"
    if satdata1.h['DataSource'] == satdata2.h['DataSource']:
        MergedData.h['DataSource'] = satdata1.h['DataSource']
        MergedData.h['Instruments'] = satdata1.h['Instruments']
        MergedData.h['FileVersion'] = satdata1.h['FileVersion']
    else:
        MergedData.h['DataSource'] = '{} ({} - {}) & {} ({} - {})'.format(satdata1.h['DataSource'],
                                                    datetime.strftime(num2date(satdata1['time'][0]), tf),
                                                    datetime.strftime(num2date(satdata1['time'][-1]), tf),
                                                    satdata2.h['DataSource'],
                                                    datetime.strftime(num2date(new_time[0]), tf),
                                                    datetime.strftime(num2date(new_time[-1]), tf))
        MergedData.h['Instruments'] = satdata1.h['Instruments'] + satdata2.h['Instruments']
        MergedData.h['FileVersion'] = {**satdata1.h['FileVersion'], **satdata2.h['FileVersion']}
    MergedData.h['SamplingRate'] = timestep
    MergedData.h['ReferenceFrame'] = satdata1.h['ReferenceFrame']

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
    #http://matplotlib.org/examples/pylab_examples/date_demo2.html
  
    j=0
    #time_str=np.empty(np.size(time_in),dtype='S19')
    time_str= ['' for x in range(len(time_in))]
    #=np.chararray(np.size(time_in),itemsize=19)
    time_num=np.zeros(np.size(time_in))
   
    for i in time_in:
        #convert from bytes (output of scipy.readsav) to string
        time_str[j]=time_in[j][0:16].decode()+':00'
        time_num[j]=date2num(datetime.strptime(time_str[j], "%Y-%m-%dT%H:%M:%S"))
        j=j+1  
    return time_num


def converttime():
    """
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


def parse_satellite_name(sat_str):
    """Parses satellite name from str."""

    sat_str_ = sat_str.replace('-','').replace('_','').replace(' ','')

    if sat_str_.lower() in ['sta', 'stereoa', 'stereoahead']:
        return 'stereo-ahead'
    elif sat_str_.lower() in ['stb', 'stereob', 'stereobehind']:
        return 'stereo-behind'





