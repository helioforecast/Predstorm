import os, sys
from datetime import datetime, timedelta
try:
    from datetime import UTC
except:
    pass
import json
import logging
import numpy as np
from matplotlib.dates import date2num, num2date
import urllib
import urllib.error

try:
    import h5py
except:
    pass
import pandas as pd


def get_today_utc():
    try: today = datetime.now(UTC).strftime("%Y-%m-%d")
    except: today = datetime.utcnow().strftime("%Y-%m-%d")
    return today


def download_noaa_rtsw_data(save_path):
    """Downloads NOAA real-time solar wind data (plasma, mag and dst)
    from the following paths:
    https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json

    Parameters
    ==========
    save_path : str
        String of directory to save files to.
    datestr : str
        Date format added to filename as e.g. mag_{datestr}.json.

    Returns
    =======
    (get_plas, get_mag) : (bool, bool)
        Both True if both files were successfully downloaded.

    Example
    =======
    >>> pla_success, mag_success = download_noaa_rtsw_data("data")
    """

    dst_noaa =    "http://services.swpc.noaa.gov/products/kyoto-dst.json"
    plasma_noaa = "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json"
    mag_noaa =    "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json"

    logging.info('downloading NOAA real time solar wind plasma, mag and dst data.')

    get_plas, get_mag, get_dst = True, True, True

    today = get_today_utc()

    os.makedirs(save_path, exist_ok=True)

    try:
        urllib.request.urlretrieve(plasma_noaa, os.path.join(save_path, f'plasma_{today}.json'))
    except urllib.error.URLError as e:
        logging.error("Failed to download %s: %s", plasma_noaa, e.reason)
        get_plas = False

    try:
        urllib.request.urlretrieve(mag_noaa, os.path.join(save_path, f'mag_{today}.json'))
    except urllib.error.URLError as e:
        logging.error("Failed to download %s: %s", mag_noaa, e.reason)
        get_mag = False

    try:
        urllib.request.urlretrieve(dst_noaa, os.path.join(save_path, f'dst_{today}.json'))
    except urllib.error.URLError as e:
        logging.error("Failed to download %s: %s", dst_noaa, e.reason)
        get_dst = False

    return get_plas, get_mag, get_dst


def archive_noaa_rtsw_data(json_path, archive_path, suffix='', archive_ndays=100, fake_recurrence=False):
    """
    Creates a running archive file containing NOAA real-time solar wind
    data files in hdf5 format. The length of data archived is

    Parameters
    ==========
    json_path : str
        String of directory containing plasma data files.
    archive_path : str
        String of directory to save plasma data file.
    archive_ndays : int (default=100)
        Number of days to save in file.
    fake_recurrence : bool (default=False)
        When creating this file for the first time, call this to make a
        FAKE recurrence file that will gradually be filled with satellite
        data.

    Returns
    =======
    True if completed.

    Example
    =======
    >>> archive_noaa_rtsw_data('rtswdata', 'archive')
    """

    pla_keys = ['proton_density', 'proton_speed', 'proton_temperature']
    mag_keys = ['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    dst_keys = ['dst']
    all_keys = pla_keys + mag_keys + dst_keys

    today = get_today_utc()

    # Read latest NOAA RTSW data files:
    df_pla = pd.read_json(os.path.join(json_path, f"plasma_{today}.json"))
    df_mag = pd.read_json(os.path.join(json_path, f"mag_{today}.json"))
    df_dst = pd.read_json(os.path.join(json_path, f"dst_{today}.json"))

    # Sort by increasing time:
    df_pla = df_pla.set_index('time_tag').sort_index()
    df_mag = df_mag.set_index('time_tag').sort_index()
    df_dst = df_dst.set_index('time_tag').sort_index()

    # Make sure index is in datetime format:
    df_pla.index = pd.to_datetime(df_pla.index)
    df_mag.index = pd.to_datetime(df_mag.index)
    df_dst.index = pd.to_datetime(df_dst.index)

    # Only use active observer data:
    df_pla = df_pla[df_pla['active'] == True]
    df_mag = df_mag[df_mag['active'] == True]

    # Only save variables needed for PREDSTORM:
    df_pla = df_pla[['proton_speed', 'proton_density', 'proton_temperature']]
    df_mag = df_mag[['bx_gsm', 'by_gsm', 'bz_gsm', 'bt']]

    # Cut Dst down to past 24 hours to match pla/mag data:
    df_dst = df_dst.loc[df_dst.index[-1]-pd.Timedelta(hours=23):df_dst.index[-1]]

    # Remove possible duplicate timesteps:
    df_pla = df_pla[~df_pla.index.duplicated(keep='first')]
    df_mag = df_mag[~df_mag.index.duplicated(keep='first')]
    df_dst = df_dst[~df_dst.index.duplicated(keep='first')]

    df_MIN = pd.concat([df_pla, df_mag, df_dst], axis=1)

    if fake_recurrence:
        new_index = pd.date_range(
            df_MIN.index.min() - pd.Timedelta(days=30),
            df_MIN.index.min() - pd.Timedelta(minutes=1),
            freq="1min"
        )
        df_fake = pd.DataFrame(index=new_index, columns=df_MIN.columns)
        df_fake[:] = df_MIN.mean()

        df_MIN = pd.concat([df_fake, df_MIN])

    for key in pla_keys+mag_keys+dst_keys:
        df_MIN[key] = pd.to_numeric(df_MIN[key], errors="coerce")

    df_HOUR = df_MIN.resample("h").mean()

    hdf_file_min =  os.path.join(archive_path, f'rtsw_min_last100days{suffix}.h5')
    hdf_file_hour = os.path.join(archive_path, f'rtsw_hour_last100days{suffix}.h5')

    append_to_hdf(df_MIN,  hdf_file_min,  all_keys, archive_ndays)
    append_to_hdf(df_HOUR, hdf_file_hour, all_keys, archive_ndays)

    logging.info('Archiving of NOAA data complete.')


def archive_noaa_rtsw_data_historic(json_path, archive_path, datenow="", limit_by_ndays=100):
    """Archives the NOAA real-time solar wind data files in hdf5 format.

    Parameters
    ==========
    json_path : str
        String of directory containing plasma data files.
    archive_path : str
        String of directory to save plasma data file.

    Returns
    =======
    True if completed.

    Example
    =======
    >>> archive_noaa_rtsw_data("rtswdata", "archive")
    """

    logging.info('Archive NOAA real time solar wind data as h5 file')

    items = os.listdir(json_path)
    pla_list, mag_list, dst_list = [], [], []
    for name in items:
       if name.startswith("mag") and name.endswith(".json"):
            mag_list.append(name)
       if name.startswith("pla") and name.endswith(".json"):
            pla_list.append(name)
       if name.startswith("dst") and name.endswith(".json"):
            dst_list.append(name)

    pla_keys = ['time_tag', 'density', 'speed', 'temperature']
    mag_keys = ['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    dst_keys = ['time_tag', 'dst']
    rtsw_pla = np.zeros((5000000, len(pla_keys)))
    rtsw_mag = np.zeros((5000000, len(mag_keys)))
    rtsw_dst = np.zeros((5000000, len(dst_keys)))

    # READ FILES
    # ----------

    # Go through plasma files:
    kp = 0
    for json_file in pla_list:
        try:
            pla_data = read_noaa_rtsw_json(os.path.join(json_path, json_file))
            for ip, pkey in enumerate(pla_keys):
                rtsw_pla[kp:kp+np.size(pla_data),ip] = pla_data[pkey]
            kp = kp + np.size(pla_data)
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_pla_cut = rtsw_pla[0:kp]
    rtsw_pla_cut = rtsw_pla_cut[rtsw_pla_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_pla_cut[:,0], return_index=True)
    rtsw_pla_fin = rtsw_pla_cut[ind] # remove multiples of timesteps

    # Go through magnetic files:
    km = 0
    for json_file in mag_list:
        try:
            mag_data = read_noaa_rtsw_json(os.path.join(json_path, json_file))
            for ip, pkey in enumerate(mag_keys):
                rtsw_mag[km:km+np.size(mag_data),ip] = mag_data[pkey]
            km = km + np.size(mag_data)
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_mag_cut = rtsw_mag[0:km]
    rtsw_mag_cut = rtsw_mag_cut[rtsw_mag_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_mag_cut[:,0], return_index=True)
    rtsw_mag_fin = rtsw_mag_cut[ind] # remove multiples of timesteps

    # Go through Dst files:
    kd = 0
    for json_file in dst_list:
        try:
            dst_data = read_noaa_rtsw_json(os.path.join(json_path, json_file),
                                           timef="%Y-%m-%d %H:%M:%S")
            for ip, pkey in enumerate(dst_keys):
                rtsw_dst[kd:kd+np.size(dst_data),ip] = dst_data[pkey]
            kd = kd + np.size(dst_data)
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_dst_cut = rtsw_dst[0:kd]
    rtsw_dst_cut = rtsw_dst_cut[rtsw_dst_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_dst_cut[:,0], return_index=True)
    rtsw_dst_fin = rtsw_dst_cut[ind] # remove multiples of timesteps

    # Interpolate onto minute and hour timesteps (since both files are mismatched/missing timesteps):
    #import IPython
    #IPython.embed()
    rtsw_dst_fin = rtsw_mag_fin[:,0:2]
    first_timestamp_min = num2date(np.max((rtsw_pla_fin[0,0], rtsw_mag_fin[0,0])))
    first_timestamp_min = first_timestamp_min - timedelta(seconds=first_timestamp_min.second)
    first_timestamp_hour = num2date(np.max((rtsw_pla_fin[0,0], rtsw_mag_fin[0,0], rtsw_dst_fin[0,0])))
    first_timestamp_hour = first_timestamp_hour - timedelta(seconds=first_timestamp_hour.second)
    last_timestamp_min = num2date(np.min((rtsw_pla_fin[-1,0], rtsw_mag_fin[-1,0])))
    last_timestamp_min = last_timestamp_min - timedelta(seconds=last_timestamp_min.second)
    last_timestamp_hour = num2date(np.min((rtsw_pla_fin[-1,0], rtsw_mag_fin[-1,0], rtsw_dst_fin[-1,0])))
    last_timestamp_hour = last_timestamp_hour - timedelta(seconds=last_timestamp_hour.second)
    n_min = int((last_timestamp_min-first_timestamp_min).total_seconds() / 60)
    n_hour = int(np.round(int((last_timestamp_hour-first_timestamp_hour).total_seconds() / 60) / 60, 0))

    min_steps = np.array([date2num(first_timestamp_min+timedelta(minutes=n)) for n in range(n_min)])
    hour_steps = np.array([date2num(first_timestamp_hour+timedelta(hours=n)) for n in range(n_hour)])

    # DEFINE HEADER
    # -------------
    metadata = {
        "Description": "Real time solar wind magnetic field and plasma data from NOAA",
        "TimeRange": "{} - {}".format(first_timestamp_min.strftime("%Y-%m-%dT%H:%M"), last_timestamp_min.strftime("%Y-%m-%d %H:%M")),
        "SourceURL": "https://services.swpc.noaa.gov/products/solar-wind/",
        "CompiledBy": "Helio4Cast code, https://github.com/helioforecast/helio4cast",
        "Authors": "C. Moestl (twitter @chrisoutofspace) and R. L. Bailey (GitHub bairaelyn)",
        "FileCreationDate": datetime.utcnow().strftime("%Y-%m-%dT%H:%M")+' UTC',
        "Units": "B-field: nT, Density: cm^-3, Temperature: K, Speed: km s^-1",
        "Notes": "None in data have been replaced with np.NaNs.",
    }

    # WRITE DATA: LAST 100 DAYS
    # -------------------------
    if datenow == "":
        past_100days = datetime.utcnow() - timedelta(days=limit_by_ndays)
    else:
        past_100days = datenow - timedelta(days=limit_by_ndays)

    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # Write to file (minute timesteps):
    min_steps_100 = min_steps[min_steps > date2num(past_100days)]
    hour_steps_100 = hour_steps[hour_steps > date2num(past_100days)]
    hdf5_file = os.path.join(archive_path, 'rtsw_min_last100days_historic.h5')
    hf = h5py.File(hdf5_file, mode='w')

    hf.create_dataset('time', data=min_steps_100)
    print("---- KEYS")
    print(list(hf.keys()))
    print(hf['time'])
    for key in pla_keys[1:]:
        data_interp = np.interp(min_steps_100, rtsw_pla_fin[:,0], rtsw_pla_fin[:,pla_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in mag_keys[1:]:
        data_interp = np.interp(min_steps_100, rtsw_mag_fin[:,0], rtsw_mag_fin[:,mag_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    metadata['SamplingRate'] = 1./24./60.
    for k, v in metadata.items():
        hf.attrs[k] = v
    hf.close()

    # Write to file (hour timesteps):
    metadata["TimeRange"] = "{} - {}".format(first_timestamp_hour.strftime("%Y-%m-%dT%H:%M"), last_timestamp_hour.strftime("%Y-%m-%d %H:%M"))
    hdf5_file = os.path.join(archive_path, 'rtsw_hour_last100days_historic.h5')
    hf = h5py.File(hdf5_file, mode='w')

    hf.create_dataset('time', data=hour_steps_100)
    for key in pla_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_pla_fin[:,0], rtsw_pla_fin[:,pla_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in mag_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_mag_fin[:,0], rtsw_mag_fin[:,mag_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in dst_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_dst_fin[:,0], rtsw_dst_fin[:,dst_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    metadata['SamplingRate'] = 1./24.
    for k, v in metadata.items():
        hf.attrs[k] = v
    hf.close()

    logging.info('Archiving of NOAA data done')

    return True


def read_noaa_rtsw_json(json_file, timef="%Y-%m-%d %H:%M:%S.%f"):
    """Reads NOAA real-time solar wind data JSON files (already downloaded).

    Parameters
    ==========
    json_file : str
        String of direct path to plasma data file.

    Returns
    =======
    rtsw_data : np.array
        Numpy array with JSON keys accessible as keys or under rtsw_data.dtype.names.

    Example
    =======
    >>> json_file = 'data/plasma-7-day_2020_Mar_28_17_00.json'
    >>> pla_data = read_noaa_rtsw_json(json_file)
    """

    # Read JSON file:
    with open(json_file, 'r') as jdata:
        dp = json.loads(jdata.read())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        datesp = [datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")  for x in dpn[1:]]
        #convert datetime to matplotlib times
        mdatesp = date2num(datesp)
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(mdatesp, dpn[1:])]
        rtsw_data = np.array(dp_, dtype=dtype)

    return rtsw_data


def append_to_hdf(df, hdf_file, all_keys, archive_ndays=100):
    """
    Append new data to an HDF5 file, one key per column.

    Keeps a rolling archive of roughly archive_ndays days by trimming
    only when the stored span exceeds that limit. Trimming removes the
    oldest full day, so rewriting happens at most about once per day.
    """

    with pd.HDFStore(hdf_file, mode="a") as store:

        for key in all_keys:
            if key not in df.columns:
                continue

            df_key = df[[key]]

            # Since Dst is often updated retrospectively, update and don't append:
            if key == "dst":
                update_hourly_key(store, df_key, key, rewrite_days=2)
                continue

            if f"/{key}" in store.keys():
                nrows = store.get_storer(key).nrows
                last = store.select(key, start=nrows-1)
                last_time = last.index[0]
                df_new = df_key[df_key.index > last_time]
            else:
                df_new = df_key

            if not df_new.empty:
                store.append(
                    key,
                    df_new,
                    format="table",
                    data_columns=True
                )

            # Nothing stored for this key yet
            if "/{}".format(key) not in store.keys():
                continue

            # Check current stored span; only trim if it exceeds archive_ndays
            first = store.select(key, start=0, stop=1)
            last = store.select(key, start=store.get_storer(key).nrows - 1)

            first_time = first.index[0]
            last_time = last.index[0]

            if (last_time - first_time) > pd.Timedelta(days=archive_ndays):
                # Remove the oldest full day only
                cutoff_day = first_time.normalize() + pd.Timedelta(days=1)

                df_keep = store[key]
                df_keep = df_keep[df_keep.index >= cutoff_day]

                store.remove(key)
                store.append(
                    key,
                    df_keep,
                    format="table",
                    data_columns=True
                )

        metadata = {
            "Description": "Real time solar wind magnetic field and plasma data from NOAA",
            "TimeRange": f"{first_time:%Y-%m-%dT%H:%M} - {last_time:%Y-%m-%dT%H:%M}",
            "SourceURL": "https://services.swpc.noaa.gov/products/solar-wind/",
            "CompiledBy": "Helio4Cast code, https://github.com/helioforecast/helio4cast",
            "Authors": "C. Moestl (twitter @chrisoutofspace) and R. L. Bailey (GitHub bairaelyn)",
            "FileCreationDate": datetime.now().strftime("%Y-%m-%dT%H:%M")+' UTC',
            "Units": "B-field: nT, Density: cm^-3, Temperature: K, Speed: km s^-1",
            "Notes": "Takes only data from active observer as defined by NOAA.",
        }

        # Write metadata:
        root_attrs = store._handle.root._v_attrs
        for k, v in metadata.items():
            setattr(root_attrs, k, v)


def update_hourly_key(store, df_key, key, rewrite_days=2):
    """
    Update a key whose recent values may change from NaN to real values.
    Rewrites only a recent window.
    """
    if df_key.empty:
        return

    window_start = df_key.index.max() - pd.Timedelta(days=rewrite_days)

    # New data only for recent window
    df_recent_new = df_key[df_key.index >= window_start]

    if "/{}".format(key) in store.keys():
        df_old = store[key]
        df_old_keep = df_old[df_old.index < window_start]
        df_old_recent = df_old[df_old.index >= window_start]

        # Combine old + new, keeping newer non-NaN values where available
        combined_recent = df_old_recent.combine_first(df_recent_new)
        combined_recent.update(df_recent_new)

        df_final = pd.concat([df_old_keep, combined_recent]).sort_index()
        df_final = df_final[~df_final.index.duplicated(keep="last")]

        store.remove(key)
        store.append(key, df_final, format="table", data_columns=True)
    else:
        store.append(key, df_recent_new.sort_index(), format="table", data_columns=True)


def load_all_keys(hdf_file):
    """
    Loads pandas-created HDF5 file into DataFrame + metadata.
    """

    dfs = []

    with pd.HDFStore(hdf_file, mode="r") as store:
        # read all keys
        for key in store.keys():
            df = store[key]
            dfs.append(df)

        # Get metadata from root
        attrs = store._handle.root._v_attrs
        metadata = {k: getattr(attrs, k) for k in attrs._v_attrnames}

    # combine columns
    df_all = pd.concat(dfs, axis=1).sort_index()

    return df_all, metadata



json_path = "data"
datestrf = "%Y-%m-%d"

# NORMAL RUNS
get_plas, get_mag, get_dst = download_noaa_rtsw_data(json_path)
archive_noaa_rtsw_data(json_path, 'data', suffix='_TEST')

# WHEN FIRST CREATING A FILE, run with fake_recurrence=True
#get_plas, get_mag, get_dst = download_noaa_rtsw_data(json_path)
#archive_noaa_rtsw_data(json_path, 'data', fake_recurrence=True)

# CREATING A FILE USING DATA WITH OLD FORMAT (pre-March 2026)
#archive_noaa_rtsw_data_historic('NOAA-Data', 'data', datenow=datetime(2024,10,31))

# TEST A FILE WRITTEN BY THIS CODE
#df_test, metadata = load_all_keys("data/rtsw_min_last100days.h5")




