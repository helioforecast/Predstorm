# PREDSTORM

This is a python code for space weather prediction research.

The package is used for research on predicting the background solar wind, 
high speed solar wind streams and solar storm magnetic flux ropes, 
based on data from solar wind monitors at the Sun-Earth L1 and L5 points, 
as well as from any spacecraft positioned east of the Sun-Earth line around or < 1 AU.
We also derive predictions of the geomagnetic Dst index, Kp and auroral power.

Status: work in progress, April 2019.

If you want to use parts of these codes for generating results for 
peer-reviewed scientific publications, please contact us per email (see contributor biographies)
or via twitter @chrisoutofspace (Christian Moestl).

## Dependencies

To install the packages that need to be added to an existing anaconda python 3.7 installation (https://www.anaconda.com/distribution/), 

* sunpy (https://github.com/sunpy/sunpy), cdflib (https://github.com/MAVENSDC/cdflib),

use this on a command line:

```
conda config --append channels conda-forge
conda install sunpy
pip install cdflib
```

* predstorm_l5.py checks for an ffmpeg executable 
in the current directory (for converting images, making movies), otherwise the system-wide available version is used.

## Running the code

* On the command line:

```
python predstorm_l1.py
python predstorm_l5.py
python mfr_predict.py
```

* use 
```
python predstorm_l5.py --server 
```
for Agg backend. 

* In ipython:

```
run predstorm_l1
run predstorm_l5
run mfr_predict
```



* Folder "data/" contains a position file for planets and spacecraft 
(to be replaced in the future with positions obtained via spiceypy/heliopy) and an unpublished v2.0 of the HELCATS ICMECAT catalog.
OMNI2 data are automatically downloaded in this folder.


