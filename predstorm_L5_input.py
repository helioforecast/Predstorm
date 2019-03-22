# PREDSTORM L5 input parameters file
# ----------------------------------

# If True, show interpolated data points on the DSCOVR input plot
showinterpolated = True

# Time interval for both the observed and predicted windDelta T (hours), start with 24 hours here (covers 1 night of aurora)#the , 
deltat = 24

# Time range of training data (four solar minimum years as training data for 2018)      # TODO remove this, not in use?
trainstart = '2006-Jan-01 00:00'
trainend =   '2010-Jan-01 00:00'

# Synodic solar rotation (= 26.24 days (equator), 27.28 days for Carrington rotation, 26 deg latitude)
# --> use other values for equatorial coronal holes?
sun_syn = 26.24

# Number of days on right side of plot (future data from STEREO-A)
plot_future_days = 7

# Number of days on left side of plot (past data)
plot_past_days = 3.5

# If True, save older data for plotting Burton/OBrien Dst for verification
verification_mode = False

# File to save data to for later verification
verify_filename = 'real/savefiles/predstorm_realtime_stereo_l1_save_v1_2018-09-10-10_02.p'

# Intervals for verification
verify_int_start = '2018-09-09 12:00:00'
verify_int_end =   '2018-09-12 23:00:00'

# Dst method to be plotted (options = 'temerin_li', 'burton', 'obrien')
dst_method = 'temerin_li'

# Offset for Dst in plot
dst_offset = 0
