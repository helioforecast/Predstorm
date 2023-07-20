# PREDSTORM

This is Python3 code for space weather prediction research.

The package is used for predicting the background solar wind, high speed solar wind streams and solar storm magnetic flux ropes based on data from solar wind monitors at the Sun-Earth L1 and L5 points, as well as from any spacecraft positioned east of the Sun-Earth line around or < 1 AU. We also derive predictions of the geomagnetic Dst index, Kp and auroral power.

If you want to use parts of this code for generating results for peer-reviewed scientific publications, please contact us per email (see contributor biographies) or via twitter @chrisoutofspace (Christian Moestl).

To see this code running in real-time, visit https://helioforecast.space/solarwind.

The code was last updated in 2023.

## Installation

PREDSTORM is written in Python 3. Most dependencies (numpy, scipy, matplotlib) can be downloaded and used in an anaconda environment (https://www.anaconda.com/distribution/), and a full environment can be installed using the following lines:

	conda create --name predstorm python=3.7
	conda activate predstorm
	conda install matplotlib numpy scikit-learn pandas
	conda install h5py seaborn numba
	conda install -c conda-forge spiceypy
	conda install -c conda-forge astropy
	conda install -c anaconda requests
	pip install mplcyberpunk

A list of specific packages for the environment can be founds in the envs/ folder.

This code also currently relies on a local installation of the Helio4Cast package for downloading and saving NOAA RTSW files, and this package can be downloaded here: https://github.com/helioforecast/helio4cast.

## Running the code

Make sure to create a config file using create_config.py (with local file paths):

	python create_config.py

To run the code for the version created for L5/recurrence data (originally using STEREO-A in 2019-2021), run this in the command line:

	python run_predstorm_l5.py

### Notes

* You will need at least 30 days of past L1 data to run with the recurrence model. A Python pickle file of this data can be found here: https://helioforecast.space/static/sync/insitu_python/noaa_rtsw_last_30files_now.p
* Dst predictions from L1 solar wind rely on a scikit-learn-based GradientBooestingRegressor algorithm. The code for training this model will be uploaded in the near future.
* Running the scripts creates the local folder /data. OMNI2 data, among other things, are automatically downloaded into this folder.
* Realtime results are saved in the folder predstorm/realtime.
* Logs of the scripts are saved in predstorm.log.

## Contributors

Austrian Space Weather Office (Helio4Cast) and Conrad Observatory, GeoSphere Austria, Graz, Austria:
* Christian Möstl
* Rachel L. Bailey
