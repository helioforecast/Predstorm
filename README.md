# PREDSTORM

This is a python code for space weather prediction research.

The package is used for research on predicting the background solar wind, 
high speed solar wind streams and solar storm magnetic flux ropes, 
based on data from solar wind monitors at the Sun-Earth L1 and L5 points, 
as well as from any spacecraft positioned east of the Sun-Earth line around or < 1 AU.

Status: work in progress, November 2018 
issues: 
- predstorm_L5_stereob_errors.py needs to be rewritten for new data structures (recarrays)
- rewrite verification in predstorm_L5.py  
- use astropy instead of ephem in predstorm_module.py

If you plan to use this code for generating results for 
peer-reviewed scientific publications, please contact me (see bio).


## Dependencies

* seaborn, sunpy, urllib, json, cdflib (https://github.com/MAVENSDC/cdflib)

## Running the code

* ....