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
from datetime import datetime
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


def get_satellite_position(satname, timestamp, kernelpath=None, kernellist=None, refframe="J2000", rlonlat=False):
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

    position, lighttimes = spiceypy.spkpos(satstr, time, refframe, 'NONE', 'SUN')

    if rlonlat:
        return spiceypy.reclat(position)
    else:
        return position


def get_satellite_position_heliopy(satname, timestamp, refframe='J2000', rlonlat=False):
    """Uses Heliopy's spice to get position information. Will automatically download
    required kernels.
    """

    import heliopy.data.spice as spicedata
    import heliopy.spice as hspice

    hspice.furnish(spicedata.get_kernel('stereo_a_pred'))
    sta = hspice.Trajectory(satname)
    sta.generate_positions(timestamp, 'Sun', refframe)

    sta.change_units('AU')
    if rlonlat:
        return cart2sphere(sta.x, sta.y, sta.z)
    else:
        return (sta.x, sta.y, sta.z)


if __name__ == '__main__':

    position = get_satellite_position('STEREO-A', [datetime.strptime("2019-06-22", "%Y-%m-%d"), datetime.strptime("2019-06-23", "%Y-%m-%d")])
    print("Stereo was at {}".format(position))


