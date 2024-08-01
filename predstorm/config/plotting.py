import matplotlib.pyplot as plt

# GENERAL
# -------
lw = 1
fs = 11
# Figure size (in)
figsize=(14,12)
# Seaborn settings
sns_context = 'talk'
sns_style = 'darkgrid'
# Font sizes
fs_legend = 9
fs_levels = 10
fs_title = 16
fs_ylabel = 13
# X-axis labels
date_fmt = '%b %d %Hh'
# Y-axis labels:

# DATA-SPECIFIC
# -------------
# Colour for DSCOVR data
c_dis = 'black'
c_dis_dst = 'purple'
# Colour for STEREO data
c_sta = 'red'
c_sta_dst = 'blue'
# Colour for magnetic fields
c_btot = 'black'
c_bx, c_by, c_bz = 'teal', 'orange', 'purple'
# Colour for Dst
c_dst = 'black'
ms_dst = 4
# Colour for kp
c_kp = 'lightgreen'
# Colour for aurora power
c_aurora = 'orange'
# Colour for newell coupling
c_ec = 'teal'

# FUNCTIONS
# ---------

def plot_dst_activity_lines(xlims=None, ax=None, color='k'):
    if ax == None:
        ax = plt.gca()
    if xlims == None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlims
    ax.plot_date([xmin, xmax], [0, 0],'--k', alpha=0.3, linewidth=1)
    for hline, linetext in zip([-50, -100, -250], ['moderate', 'intense', 'super-storm']):
        ax.plot_date([xmin, xmax], [hline, hline],'--', color=color, alpha=0.3, lw=1)
        ax.annotate(linetext, xy=(xmin, hline+2), xytext=(xmin+(xmax-xmin)*0.005, hline+2), color=color, fontsize=fs_levels)

def plot_speed_lines(xlims=None, ax=None, color='k'):
    if ax == None:
        ax = plt.gca()
    if xlims == None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlims
    for hline, linetext in zip([400, 800], ['slow', 'fast']):
        ax.plot_date([xmin, xmax], [hline, hline],'--', color=color, alpha=0.3, linewidth=1)
        ax.annotate(linetext,xy=(xmin,hline), xytext=(xmin+(xmax-xmin)*0.005,hline+5), color=color, fontsize=fs_levels)

def liability_text():
    plt.figtext(0.01,0.03,'We take no responsibility or liability for the frequency of provision and accuracy of this forecast.' , fontsize=8, ha='left')
    plt.figtext(0.01,0.01,'We will not be liable for any losses and damages in connection with using the provided information.' , fontsize=8, ha='left')

def group_info_text():
    plt.figtext(0.99,0.05,'Austrian Space Weather Office, GeoSphere Austria, https://helioforecast.space', fontsize=12, ha='right')
    plt.figtext(0.99,0.025,'https://x.com/ASWOGeoSphere', fontsize=12, ha='right')

def group_info_text_small():
    plt.figtext(0.90,0.020,'Austrian Space Weather Office, GeoSphere Austria, https://helioforecast.space', fontsize=12, ha='right')
