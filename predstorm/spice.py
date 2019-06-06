#!/usr/bin/env python
"""
For planetary ephemeris data (de4##.dsp):
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

For planetary constants (pck00##.tpc):
https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/

For HEEQ frame (RSSD0002.tf):
https://naif.jpl.nasa.gov/pub/naif/VEX/kernels/fk/
https://naif.jpl.nasa.gov/pub/naif/pds/data/nh-j_p_ss-spice-6-v1.0/nhsp_1000/data/fk/heliospheric_v004u.tf

For STEREO ephemeris data:
https://stereo-ssc.nascom.nasa.gov/data/moc_sds/ahead/data_products/ephemerides/
https://sohowww.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/depm/ahead/
"""

import os
import logging
from datetime import datetime, timedelta
import heliopy.data.spice as spicedata
import heliopy.spice as hspice
import numpy as np
import pickle
import spiceypy

logger = logging.getLogger(__name__)

required_kernels = {'STEREO AHEAD': 
                        ['ahead_2017_061_5295day_predict.epm.bsp',
                         'naif0012.tls',
                         'de430.bsp',
                         'pck00010.tpc',
                         'heliospheric_v004u.tf'],
                    'EARTH' :
                        ['de430.bsp',
                         'naif0012.tls',
                         'heliospheric_v004u.tf']
                                     }


#@jit(nopython=True)
def cart2sphere(x,y,z):
    r = np.sqrt(x**2. + y**2. + z**2.)
    theta = np.arctan2(z,np.sqrt(x**2. + y**2.))
    phi = np.arctan2(y,x)
    return (r, theta, phi)


def get_satellite_position(satname, timestamp, kernelpath=None, kernellist=None, refframe="J2000", refobject='SUN', rlonlat=False):
    """
    Returns satellite position from the reference frame of the Sun.
    Files here:
    https://sohowww.nascom.nasa.gov/solarsoft/stereo/gen/data/spice/depm/ahead/
    and
    https://stereo-ssc.nascom.nasa.gov/data/moc_sds/ahead/data_products/ephemerides/

    Parameters
    ==========
    satname : str
        Name of satellite. Recognised strings: 'stereo'
    timestamp : datetime.datetime object / list of dt objs
        Times of positions to return.
    kernelpath : str (default=None)
        Optional path to directory containing kernels, else local "kernels" is used.
    kernellist : str (default=None)
        Optional list of kernels in directory kernelpath, else local list is used.
    refframe : str (default='J2000')
        See SPICE Required Reading Frames. J2000 is standard, HEE/HEEQ are heliocentric.
    refobject : str (default='Sun')
        String for reference onject, e.g. 'Sun' or 'Earth'
    rlonlat : bool (default=False)
        If True, returns coordinates in (r, lon, lat) format, not (x,y,z).

    Returns
    =======
    position : array(x,y,z), list of arrays for multiple timestamps
        Position of satellite in x, y, z with reference frame refframe and Earth as
        observing body. Returns (r,lon,lat) if rlonlat==True.
    """

    if 'stereo' in satname.lower():
        if 'ahead' in satname.lower() or 'a' in satname.lower():
            satstr = 'STEREO AHEAD'
        if 'behind' in satname.lower() or 'b' in satname.lower():
            satstr = 'STEREO BEHIND'
    elif 'dscovr' in satname.lower():
        satstr = 'DSCOVR'
        logger.error("get_satellite_position: DSCOVR kernels not yet implemented!")
    else:
        satstr = satname.upper()
        logger.warning("get_satellite_position: No specific SPICE kernels for {} satellite!".format(satname))

    if kernellist == None:
        kernellist = required_kernels[satstr]

    if kernelpath == None:
        kernelpath = os.path.join('data/kernels')

    logger.info("get_satellite_position: Reading SPICE kernel files from {}".format(kernelpath))

    for k in kernellist:
        spiceypy.furnsh(os.path.join(kernelpath, k))
    time = spiceypy.datetime2et(timestamp)

    position, lighttimes = spiceypy.spkpos(satstr, time, refframe, 'NONE', refobject)

    if rlonlat:
        if len(position) > 3:
            return np.asarray([spiceypy.reclat(x) for x in position])
        return spiceypy.reclat(position)
    else:
        return position


def get_satellite_position_heliopy(satname, timestamp, refframe='J2000', refobject='Sun', rlonlat=False, returnobj=False):
    """Uses Heliopy's spice to get position information. Will automatically download
    required kernels.

    Parameters
    ==========
    satname : str
        Satellite name. Currently available: ['stereo_a', 'stereo_a_pred', 'earth']
    timestamp : datetime / list of datetimes
        Datetime objects to iterate through and return positions for.
    refframe : str
        String denoting reference frame to use for position.
    refobject : str (default='Sun')
        String for reference onject, e.g. 'Sun' or 'Earth'
    rlonlat : bool (default=False)
        If True, returns coordinates in (r, lon, lat) format, not (x,y,z).
    returnobj : bool (default=False)
        If True, returns heliopy.Trajectory object instead of arrays.

    Returns
    =======
    None - saves pickled file to posdir with file format SATNAME_TIMERANGE_REFFRAME.p
    """

    if isinstance(timestamp, datetime):
        timestamp = [timestamp]
    elif isinstance(timestamp, list):
        pass
    else:
        logger.warning("get_satellite_position_heliopy: Don't recognise input timestamp format!")

    if 'stereoa' in satname.lower().replace('-','').replace('_',''):
        if 'pred' in satname.lower():
            heliostr = 'stereo_a_pred'
        else:
            heliostr = 'stereo_a'
        satstr = 'STEREO AHEAD'
    elif 'stereob' in satname.lower().replace('-','').replace('_',''):
        if 'pred' in satname.lower():
            heliostr = 'stereo_b_pred'
        else:
            heliostr = 'stereo_b'
        satstr = 'STEREO BEHIND'
    elif satname.lower() in ['mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
        heliostr = 'planet_trajectories'
        satstr = satname

    hspice.furnish(spicedata.get_kernel(heliostr))
    pos = hspice.Trajectory(satstr)
    pos.generate_positions(timestamp, refobject, refframe)

    if returnobj:
        return pos

    pos.change_units('AU')
    if rlonlat:
        return cart2sphere(pos.x, pos.y, pos.z)
    else:
        return (pos.x, pos.y, pos.z)


def make_position_file(satname, timerange, refframe, refobject='Sun', samprate=6, posdir='data/positions', heliopy=True):
    """Makes pickled version of position data. Uses either heliopy's spice.get_kernel 
    function so kernels will be downloaded into local heliopy file.

    Parameters
    ==========
    satname : str
        Satellite name. Currently available: ['stereo_a', 'stereo_a_pred', 'earth']
    timerange : [datetime, datetime]
        Datetime objects to iterate through and return positions for.
    refframe : str
        String denoting reference frame to use for position.
    refobject : str (default='Sun')
        String for reference onject, e.g. 'Sun' or 'Earth'
    samprate : int (default=6)
        Sampling rate (in hours) for output.
    posdir : str (default='data/positions')
        Path to where file will be stored.
    heliopy : bool (default=True)
        If True, heliopy method is used.

    Returns
    =======
    None - saves pickled file to posdir with file format SATNAME_TIMERANGE_REFFRAME.p
    """

    if os.path.isdir(posdir) == False:
        os.mkdir(posdir)

    times = [timerange[0] + timedelta(hours=n) for n in np.arange(0., (timerange[1]-timerange[0]).days*24, samprate)]

    if heliopy:
        logger.info("make_position_file: Using heliopy to generate position file for {} for times {} till {} with resolution {} hours".format(
            satname, times[0], times[-1], samprate))
        posdata = get_satellite_position_heliopy(satname, times, refframe=refframe, 
                                                 refobject=refobject, rlonlat=False)
    else:
        print("Not yet implemented.")
        posdata = get_satellite_position(satname, times, refframe=refframe, rlonlat=False)

    s_timerange = datetime.strftime(timerange[0], "%Y%m%d")+'-'+datetime.strftime(timerange[-1], "%Y%m%d")
    savefile = os.path.join(posdir, "{}_{}_{}_{:d}h.p".format(satname, s_timerange, refframe, samprate))
    pickle.dump(pos, open(savefile, "wb"))

    logger.info("make_position_file: File saved to {}".format(savefile))

    return


def main():
    """Automatically generate all position files."""

    refframe = 'HEEQ'
    dates = [datetime.strptime("20000101","%Y%m%d"), datetime.strptime("20250101","%Y%m%d")]
    make_position_file('Earth', dates, refframe)
    try:
        # Load older dates first to get all historical kernels:
        stereoa_dates = [datetime.strptime("20070101","%Y%m%d"), datetime.strptime("20190501","%Y%m%d")]
        make_position_file('STEREOA', stereoa_dates, refframe)
        # Using those, make predictions:
        stereoa_dates = [datetime.strptime("20070101","%Y%m%d"), datetime.strptime("20250101","%Y%m%d")]
        make_position_file('STEREOA-pred', stereoa_dates, refframe)
        stereob_dates = [datetime.strptime("20070101","%Y%m%d"), datetime.strptime("20140928","%Y%m%d")]
        make_position_file('STEREOB', stereob_dates, refframe)
    except Exception as e:
        print(e)
        print('\n'+"ERROR: Heliopy version may be old. Try installing most recent version: pip install heliopy==0.7.0")


if __name__ == '__main__':

    main()

