#!/usr/bin/env python
"""
--------------------------------------------------------------------
Creates config file for use in PREDSTORM scripts. Adjust values to 
local paths and modelling preferences.
https://docs.python.org/3/library/configparser.html
--------------------------------------------------------------------
"""

import sys
import configparser
import getopt

config = configparser.ConfigParser()

# ------------
# ALL SERVERS
# ------------

config['DEFAULT'] =            {'Location': 'Home'
                                'ScriptPath': '/home/user/Scripts/PREDSTORM'}

# Constants and variables
config['Parameters'] =         {'SynodicRotation' : '26.24', # synodic sun rotation
                                }

# Variables and paths needed for predictions based on NOAA RTSW data
config['RealTimePred'] =       {'DstPredMethod'   : 'ml_dstdiff',
                                # Offset to the resulting Dst data (nT)
                                'DstOffset'       : '0',
                                # Path to Dst prediction model to load
                                'DstModelPath'    : '/home/user/Scripts/PREDSTORM/dst_pred_model_final.pickle',
                                # Number of days into the past to plot
                                'PlotPastDays'    : '3.5',
                                # Number of days into the future to plot
                                'PlotFutureDays'  : '3',
                                # Number of days into the future to save as prediction
                                'SaveFutureDays'  : '14',
                                # Source of data for next days (options are 'recurrence' or 'STEREO-A')
                                'FutureSource'    : 'Recurrence',
                                # Location to save input data e.g. real-time solar wind
                                'InputDataPath'   : '/home/user/Scripts/PREDSTORM/data',
                                # Location to save realtime plots and data
                                'RealtimePath'    : '/home/user/Scripts/PREDSTORM/realtime',
                                # Location to archive plots and data
                                'ArchivePath'     : '/home/user/Scripts/PREDSTORM/archive'}

# CREATE CONFIG FILE
# ===================
with open('config.ini', 'w') as configfile:
    config.write(configfile)
