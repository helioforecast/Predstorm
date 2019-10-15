'''

predict_ec.py


Errors in predicting the Newell coupling with a spacecraft east of the Sun-Earth line


Rachel Bailey, Christian Moestl
IWF-helio, Graz, Austria

October 2019



'''


import os
import sys
import copy
from datetime import datetime, timedelta
import getpass
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import num2date, date2num
import numpy as np
import sunpy.time
import pickle
import scipy.io
from scipy import stats
import scipy
import seaborn as sns
import pdb


import heliosat
from heliosat.spice import transform_frame
import predstorm as ps
from predstorm.config.constants import AU, dist_to_L1
from predstorm.predict import dst_loss_function
# Old imports (remove later)
from predstorm.data import SatData

#logger = ps.init_logging(verbose=True)







# STEREO-B read function
def get_stereob_data(starttime, endtime, resolution='min', source='', stride=1):
    """Read STEREO-B data into SatData object.

    Notes: 
        - HGRTN != RTN: http://www.srl.caltech.edu/STEREO/docs/coordinate_systems.html"""

    logger = logging.getLogger(__name__)
    logger.info("Reading STEREO-B data")

    stb = heliosat.STB(data_source=source)

    # Magnetometer data
    magt, magdata = stb.get_mag(starttime, endtime, use_threading=False,
                                ignore_missing_files=True, stride=60)
    magt = [datetime.fromtimestamp(t) for t in magt]
    magdata[magdata < -1e20] = np.nan
    br, bt, bn = magdata[:,0], magdata[:,1], magdata[:,2]
    btot = np.sqrt(br**2. + bt**2. + bn**2.)

    # Particle data
    if source == 'beacon':
        if starttime <= datetime(2009, 9, 13):
            stb._fc_cdf_keys = ["Epoch1", "Density", "Velocity_HGRTN", "Temperature_Inst"]
            if endtime > datetime(2009, 9, 13):
                fct_s, fcdata_s = stb.get_fc(starttime, datetime(2009, 9, 13, 23, 59, 00), use_threading=False,
                                         ignore_missing_files=True, stride=stride)
                temp_total = np.sqrt(fcdata_s[:,-3]**2. + fcdata_s[:,-2]**2. + fcdata_s[:,-1]**2.)
                fcdata_s = np.hstack((fcdata_s[:,:-3], temp_total))
                stb._fc_cdf_keys = ["Epoch1", "Density", "Velocity_RTN", "Temperature_Inst"]
                fct_e, fcdata_e = stb.get_fc(datetime(2009, 9, 14), endtime, use_threading=False,
                                         ignore_missing_files=True, stride=stride)
                fct = np.vstack((fct_s, fct_e))
                fcdata = np.vstack((fcdata_s, fcdata_e))
            else:
                fct, fcdata = stb.get_fc(starttime, endtime, use_threading=False,
                                         ignore_missing_files=True, stride=stride)
        else:
            fct, fcdata = stb.get_fc(starttime, endtime, use_threading=False,
                                     ignore_missing_files=True, stride=stride)
    else:
        fct, fcdata = stb.get_fc(starttime, endtime, use_threading=False,
                                 ignore_missing_files=True, stride=stride)

    fct = [datetime.fromtimestamp(t) for t in fct]
    fcdata[fcdata < -1e29] = np.nan
    density = fcdata[:,0]
    if source == 'beacon':
        vx, vtot = fcdata[:,1], fcdata[:,4]
        temperature = fcdata[:,5]
    else:
        vx, vtot = fcdata[:,1], fcdata[:,2]
        temperature = fcdata[:,3]

    stime = date2num(starttime) - date2num(starttime)%(1./24.)
    # Roundabout way to get time_h ensures timings with full hours/mins:
    if resolution == 'hour':
        nhours = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60./60.
        tarray = np.array(stime + np.arange(0, nhours)*(1./24.))
    elif resolution == 'min':
        nmins = (endtime.replace(tzinfo=None) - num2date(stime)).total_seconds()/60.
        tarray = np.array(stime + np.arange(0, nmins)*(1./60.))

    # Interpolate variables to time:
    br_int = np.interp(tarray, date2num(magt), br)
    bt_int = np.interp(tarray, date2num(magt), bt)
    bn_int = np.interp(tarray, date2num(magt), bn)
    btot_int = np.interp(tarray, date2num(magt), btot)
    density_int = np.interp(tarray, date2num(fct), density)
    vx_int = np.interp(tarray, date2num(fct), vx)
    vtot_int = np.interp(tarray, date2num(fct), vtot)
    temp_int = np.interp(tarray, date2num(fct), temperature)

    # Pack into object:
    stbf = SatData({'time': tarray,
                    'btot': btot_int, 'br': br_int, 'bt': bt_int, 'bn': bn_int,
                    'speed': vtot_int, 'speedx': vx_int, 'density': density_int, 'temp': temp_int},
                    source='STEREO-B')
    stbf.h['DataSource'] = "STEREO-B Beacon"
    stbf.h['SamplingRate'] = tarray[1] - tarray[0]
    stbf.h['CoordinateSystem'] = 'SCEQ'

    return stbf
    
    
    
    # Function for printing statistics on dst diffs
def get_statistics(dst_pred, dst_real, source='L1', printtext=True):
    """Prints some nice statistics, that's all."""

    dst_diff = dst_real - dst_pred
    dst_diff_mean = np.nanmean(dst_diff)
    dst_diff_std = np.nanstd(dst_diff)
    t = np.linspace(0.0, stbh['time'][-1] - stbh['time'][0], len(dst_real), endpoint=False)
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
    stat_dict['xcorr'] = 24.*dt[xcorr.argmax()]
    stat_dict['ppmc'] = ppmc
    stat_dict['mae'] = mae

    return stat_dict
    
    
    
    
    
    
    
    
    
    
    
# Starting parameters
nfig = 1


# Load STEREO-B data
pickle_path = 'data/stb_satdata_h.p'
if not os.path.exists(pickle_path):
    #heliosat.configure_logging()
    starttime = datetime(2007,3,20)#
    endtime = datetime(2014,9,1)#
    stb = ps.get_stereo_beacon_data(starttime, endtime, resolution='hour', which_stereo='behind')
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(stb, pickle_file)
else:
    stb = pickle.load(open(pickle_path, 'rb'))
    # Converting RTN to quasi-GSE:
    stb['bx'], stb['by'], stb['bz'] = -stb['br'], -stb['bt'], stb['bn']
    stb.h['HeliosatObject'] = heliosat.STB()
stb = stb.interp_nans()
# Converting quasi-GSE to quasi-GSM:
stb.convert_GSE_to_GSM()
print(stb)




# Load OMNI data
omni = ps.get_omni_data()
omni = omni.interp_nans()
print(omni)



# Copy data for 27-day 'persistence' model
pers = copy.deepcopy(omni)
pers['time'] -= 26.27 # subtract 27 days
pers = pers.make_hourly_data()
print(pers)



# Load position data
print('Load spacecraft and planetary positions')        
stb.load_positions()




# Extract intervals from STEREO-B data
angle_bracket = [-0, -110] # from 1 till 2 in time (can't be backwards)
min_time = np.round(stb['time'][np.where(stb.pos['lon']*180/np.pi < angle_bracket[0])[0][0]], 0)
max_time = np.round(stb['time'][np.where(stb.pos['lon']*180/np.pi < angle_bracket[1])[0][0]], 0)
print('Time range covered for analysis in angle bracket {}:'.format(angle_bracket))
print('start:', num2date(min_time))
print('end:  ', num2date(max_time))

# Cut data
stbh = stb.make_hourly_data()
stbh = stbh.cut(starttime=num2date(min_time), endtime=num2date(max_time))


# Shift time from L5 to L1
t_unmapped = copy.deepcopy(stbh['time'])
stbh = stbh.shift_time_to_L1(method='new', sun_syn=26.24)
stbh = stbh.make_hourly_data()



# Final cut
stime_new = num2date(max((omni['time'][0], stbh['time'][0], pers['time'][0])))
etime_new = num2date(min((omni['time'][-1], stbh['time'][-1], pers['time'][-1])))
print()

print('Time range correction after time shift:')
print('start:', stime_new)
print('end:  ', etime_new)

stbh = stbh.cut(starttime=stime_new, endtime=etime_new)
omni = omni.cut(starttime=stime_new, endtime=etime_new)
pers = pers.cut(starttime=stime_new, endtime=etime_new)



# Final cut (AGAIN so it finally matches up for whatever reason)
stime_new = num2date(max((omni['time'][0], stbh['time'][0], pers['time'][0])))
etime_new = num2date(min((omni['time'][-1], stbh['time'][-1], pers['time'][-1])))
print()

print('Time range correction after time shift:')
print('start:', stime_new)
print('end:  ', etime_new)

stbh = stbh.cut(starttime=stime_new, endtime=etime_new)
omni = omni.cut(starttime=stime_new, endtime=etime_new)
pers = pers.cut(starttime=stime_new, endtime=etime_new)


# Shift wind from L5 to L1
stbh = stbh.shift_wind_to_L1()
stb_r_1h = stbh.pos['r']
stb_long_1h = stbh.pos['lon']*180/np.pi
stb_lat_1h = stbh.pos['lat']*180/np.pi
# Position of Earth L1
dttime = [num2date(t).replace(tzinfo=None) for t in stbh['time']]
L1Pos = ps.get_l1_position(dttime, units=stbh.pos.h['Units'], refframe=stbh.pos.h['ReferenceFrame'])
l1_r_1h = L1Pos['r']
l1_lon_1h = L1Pos['lon']*180/np.pi
l1_lat_1h = L1Pos['lat']*180/np.pi










###################################### Newell Coupling Metrics



ec_omni = omni.get_newell_coupling()['ec']/4421.
ec_stb = stbh.get_newell_coupling()['ec']/4421.
ec_pers = pers.get_newell_coupling()['ec']/4421.

print('mean Ec STB', np.mean(ec_stb),np.std(ec_stb))
print('mean Ec OMNI', np.mean(ec_omni),np.std(ec_omni))
print('mean Ec pers', np.mean(ec_pers), np.std(ec_pers))


print()

small_ec=np.where(np.logical_and(ec_omni> 0, ec_omni <3))

print('O-C selected mean ', np.mean(ec_omni[small_ec]-ec_stb[small_ec]))
print('O-C selected std', np.std(ec_omni[small_ec]-ec_stb[small_ec]))

print(np.size(ec_omni[small_ec]))

print()
print('O-C mean', np.mean(ec_omni[0:10000]-ec_pers[0:10000]))
print('O-C std', np.std(ec_omni[0:10000]-ec_pers[0:10000]))
print(np.max(ec_stb))


plt.plot(ec_omni[small_ec]-ec_stb[small_ec])


plt.ion()






##average 4 hour weights ec_omni and ec_stb


ave_hours=4                # hours previous to integrate over, usually 4
prev_hour_weight = 0.65    # reduce weighting by factor with each hour back
 
weights = np.ones(ave_hours)  #make array with weights     
for k in np.arange(1,ave_hours,1):  weights[k] = weights[k-1]*prev_hour_weight
    
weights=np.flip(weights)


ec_omni_ave4=np.zeros(len(ec_omni)   )
ec_stb_ave4=np.zeros(len(ec_omni) )
  
#go through all times:
for i in np.arange(4,len(ec_stb)):
    ec_stb_ave4[i] = np.round(np.nansum(ec_stb[i-4:i]*weights)/ np.nansum(weights),1)
    ec_omni_ave4[i] = np.round(np.nansum(ec_omni[i-4:i]*weights)/ np.nansum(weights),1)
   


print('O-C ave4 selected std', np.std(ec_omni_ave4[small_ec]-ec_stb_ave4[small_ec]))
    

sys.exit()

















lonstep = -1
lonrange = 5
testangles = range(angle_bracket[0], angle_bracket[1], lonstep)
diff_stds, diff_stds_omni = [], []
label1, label2 = 'OMNI', 'STEREO'
ec_stat_dicts = {'omni': [], 'stb': [], 'pers': [], 'angles': [], 'latdiffs': []}
for dtest in testangles:
    t_inds = np.where((stbh.pos['lon']*180/np.pi > (dtest-lonrange/2.)) & (stbh.pos['lon']*180/np.pi < (dtest+lonrange/2.)))
    stat_dict_omni = get_statistics(ec_pers[t_inds], ec_omni[t_inds], printtext=False)
    stat_dict = get_statistics(ec_stb[t_inds], ec_omni[t_inds], printtext=False)
    ec_stat_dicts['omni'].append(stat_dict_omni)
    ec_stat_dicts['stb'].append(stat_dict)
    ec_stat_dicts['angles'].append(dtest)
    ec_stat_dicts['latdiffs'].append(np.mean(stbh.pos['lat'][t_inds]*180./np.pi - l1_lat_1h[t_inds]))
    diff_stds_omni.append(stat_dict_omni['diff_std'])
    diff_stds.append(stat_dict['diff_std'])

# Plot correlation values between variables:
sns.set(style="darkgrid")
nfig += 1
fig, (ax1, ax2) = plt.subplots(1, 2, num=nfig, figsize=(12, 4.5))

stat_dicts = ec_stat_dicts

# Get arrays of values:
dlons = np.array(stat_dicts['angles'])
ppmc_omni = np.array([x['ppmc'] for x in stat_dicts['omni']])
ppmc_stb = np.array([x['ppmc'] for x in stat_dicts['stb']])
mean_omni = np.array([x['diff_mean'] for x in stat_dicts['omni']])
mean_stb = np.array([x['diff_mean'] for x in stat_dicts['stb']])
std_omni = np.array([x['diff_std'] for x in stat_dicts['omni']])
std_stb = np.array([x['diff_std'] for x in stat_dicts['stb']])

# First plot: correlation of Dst(OMNI) and Dst(STB) with Kyoto-Dst
ax1.plot(dlons, ppmc_omni, 'rx', label='PERS')
ax1.plot(dlons, ppmc_stb, 'bx', label='STB')

# Line denoting position of L5:
ax1.axvline(-60, c='k', lw=1)
ax1.annotate('L5', (-62, 0.68))

# Add linear functions:
slope, intercept, lo_slope, up_slope = stats.theilslopes(ppmc_omni, dlons)
print("PERS: slope={:.4f}, intercept={:.2f}".format(slope, intercept))
ax1.plot(dlons, slope*dlons+intercept, 'r-', alpha=0.7)
slope, intercept, lo_slope, up_slope = stats.theilslopes(ppmc_stb, dlons)
print("STB:  slope={:.2f}, intercept={:.2f}".format(slope, intercept))
ax1.plot(dlons, slope*dlons+intercept, 'b-', alpha=0.7)

# Third plot: mean average error and stddev of Dst(OMNI) and Dst(STB) with Kyoto-Dst
ax2.plot(dlons, mean_omni, color='r', ls='-', lw=2, label='PERS')
ax2.fill_between(dlons, mean_omni-std_omni, mean_omni+std_omni,
                 facecolor='r', alpha=0.5)
ax2.plot(dlons, mean_stb, color='b', ls='-', lw=2, label='STB')
ax2.fill_between(dlons, mean_stb-std_stb, mean_stb+std_stb,
                 facecolor='b', alpha=0.5)
        
# Formatting:
ylabelpad = -3
ax1.set_title("CC of pred. EC with OMNI EC")
ax1.set_xlabel(r"$\Delta lon$ Earth-STB")
ax1.set_ylabel("correlation coefficient", labelpad=ylabelpad)
ax1.set_xlim([angle_bracket[0]+2, angle_bracket[0]-2])
ax1.set_xlim([angle_bracket[0], angle_bracket[1]])
ax1.legend()
ax2.set_title("MAE of pred. EC with OMNI EC")
ax2.set_xlabel(r"$\Delta lon$ Earth-STB")
ax2.set_ylabel("mean average error", labelpad=ylabelpad)
ax2.set_xlim([angle_bracket[0]+2, angle_bracket[1]-2])
ax2.legend(loc='lower left')

plt.tight_layout()
plt.subplots_adjust(wspace=0.2)
plt.show()













