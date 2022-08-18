# PREDSTORM

This is Python3 code for space weather prediction research.

The package is used for predicting the background solar wind, high speed solar wind streams and solar storm magnetic flux ropes based on data from solar wind monitors at the Sun-Earth L1 and L5 points, as well as from any spacecraft positioned east of the Sun-Earth line around or < 1 AU. We also derive predictions of the geomagnetic Dst index, Kp and auroral power.

Status in 2019: work in progress.

If you want to use parts of this code for generating results for peer-reviewed scientific publications, please contact us per email (see contributor biographies) or via twitter @chrisoutofspace (Christian Moestl).

## Installation

PREDSTORM is written in Python 3. Most dependencies (numpy, scipy, matplotlib) can be downloaded and used in an anaconda environment (https://www.anaconda.com/distribution/) and can be installed using the following lines:

	  conda install scipy numpy matplotlib scikit-learn seaborn requests-ftp beautifulsoup4
	  conda install -c conda-forge spiceypy cftime
	  conda install -c numba numba

Remaining dependencies (particularly those for CDF handling) can be downloaded using pip:

    pip install cdflib spacepy astropy

PREDSTORM also relies on the HelioSat package for all heliospheric data downloads and SPICE kernel handling. Currently it can be downloaded from GitHub:

    git clone https://github.com/ajefweiss/HelioSat.git
    cd HelioSat
	  python setup.py install

HelioSat automatically downloads kernels and all required satellite files (e.g. STEREO, DSCOVR, ...). To set the path where these files are downloaded to, set the following environment variable (in .bashrc on Linux, .bash_profile on Mac):

    export HELIOSAT_DATAPATH="/path/to/data/heliosat"
    
It's a good idea to import the package after first installation to handle first download of all required SPICE kernels.

## Running the code

In the command line:

    python predstorm_l1.py
    python predstorm_l5.py

Use the following option for the Agg backend in matplotlib:

    python predstorm_l5.py --server 

### Notes

* Running the scripts creates the local folder /data. OMNI2 data, among other things, are automatically downloaded into this folder.

* Results are saved in the folder /results.

* Logs of the scripts are saved in predstorm.log.

## Contributors

IWF-Helio Group, Space Research Institute (ÖAW), Graz, Austria:
* Christian Möstl
* Rachel L. Bailey
