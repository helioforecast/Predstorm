# PREDSTORM L5 input parameters file
# ----------------------------------

# Synodic solar rotation (= 26.24 days (equator), 27.28 days for Carrington rotation, 26 deg latitude)
# --> use other values for equatorial coronal holes?
sun_syn = 26.24

# Number of days on right side of plot (future data from STEREO-A)
plot_future_days = 6.0

# Number of days on left side of plot (past data)
plot_past_days = 3.5

# If True, save older data for plotting Burton/OBrien Dst for verification
verification_mode = False

# Dst method to be plotted (options = 'ml', 'mldstdiff', 'temerin_li', 'temerin_li_2006', 'burton', 'obrien')
dst_method = 'ml'

# Offset for Dst in plot
dst_offset = 0
