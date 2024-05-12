import json
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta, datetime
from ocf_blosc2 import Blosc2
from typing import List, Dict, Union, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
from itertools import accumulate, repeat, chain, pairwise
import os
import pickle


class ClimatehackDataset(Dataset):
    """Dataset for ClimateHack.AI 2023
    Timepoints are built as the product of input dates and a time range.
    Retreives data from the relevant year/month files for a given timepoint. 
    A batch is built by selecting data for a certain time then randomly selecting sites for the batch dimension.
    
    Uses IterableDataset as superclass, since __getitem__ can fail on data size and NaN validation. We only yield successful batches.
    Tracks failed timepoints and sites to not waste time during future iterations, so the length of the dataset may vary.
    """
    def __init__(
        self,
        data: Dict[str, Any],
        metadata: pd.DataFrame,
        dates: List[datetime],
        features: Dict[str, Any],
        failed_items: Dict[str, Any],
        batch_size: int = 32,
        seed: Optional[int] = None,
        shuffle: bool = True,
        pv_features: Optional[dict] = None,
        hrv_crop: int = 128,
        nonhrv_crop: int = 128,
        weather_crop: int = 128,
        offset_start_time: bool = False
    ):
        """Dataset contstructor. Shouldn't be called directly, use convenience function get_datasets instead.

        Parameters
        ----------
        data : dict[tuple[int, int], dict[str, Any]]
            Dict containing keys as tuple of (year, month). Values are a dict with feature names as keys and
            datasets as values.
        metadata : pd.DataFrame
            ClimateHack.AI provided metadata for each site. Includes (x, y) indices for each feature type and site.
        dates : list[datetime]
            Dates used for generating iterable dataset. Train/test splits have different dates.
        features : dict[str, Any]
            Features to be extracted from xarray.Dataset. For each key, structure may vary.
            True, False or None are always accepted. True means all features will be used.
                "hrv": True, False, or None
                "nonhrv": list with possible feature names
                "weather": list with possible feature names
                "aerosols": list of tuples (feature, altitude)
        failed_items : dict[str, Any]
            Info about which dates, times, and sites are invalid due to shape and NaN checking.
        batch_size : int, optional
            Number of sites to use when creating a batch for a given timepoint, by default 32.
        seed : Optional[int], optional
            Seed to use for random shuffling, by default None
        shuffle : bool, optional
            Whether to shuffle the dataset each __iter__ call, by default True
        hrv_crop: int, by default 128
            Height/width of HRV data to crop.
        weather_crop: int, by default 128
            Height/width of weather data to crop.
        """
        super().__init__()
        self.data = data
        self.metadata = metadata
        self.pv_features = pv_features
        self.dates = dates
        self.features = features
        self.failed_items = failed_items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.offset_start_time = offset_start_time
        if self.pv_features:
            assert self.features["metadata"] is True, "Must have metadata=True to use pv features"

        self._weather_ub = int((weather_crop + 1) // 2)
        self._weather_lb = int(weather_crop // 2)
        
        self._hrv_ub = int((hrv_crop + 1) // 2)
        self._hrv_lb = int(hrv_crop // 2)
        
        self._nonhrv_ub = int((nonhrv_crop + 1) // 2)
        self._nonhrv_lb = int(nonhrv_crop // 2)

        self.seed = seed
        np.random.seed(seed)
        
        initial_times = _get_times(self.dates)
        self.initial_times = [t for t in initial_times if t not in self.failed_items["times"]]
        if self.offset_start_time:
            # As long as the next hour is valid, then we can do offsets
            ts = sorted(self.initial_times)
            valid_ts = [t0 for t0, t1 in pairwise(ts) if t0.hour + 1 == t1.hour]
            self.initial_times = valid_ts
        if self.shuffle:
            np.random.shuffle(self.initial_times)

    def __len__(self) -> int:
        return len(self.initial_times)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Retreives one item given an index to the initial times instance variable.
        
        Parameters
        ----------
        index : int
            Index for self.initial_times

        Returns
        -------
        Dict[str, torch.Tensor]
            Data for network, tensors of shape (B, C, N, H, W)
        """
        t0 = self.initial_times[index]
        if self.offset_start_time:
            # Randomly pick interval within the hour
            mins = pd.Timedelta(minutes=np.random.choice(12) * 5)
            t0 = t0 + mins

        # Read data into memory
        data = self._get_features(t0)
        # Randomly select sites based on batch size
        all_sites = set(self.metadata.index)
        # Exclude sites that are not valid
        invalid_sites = set(self.failed_items["sites"][t0.floor("60min")])
        if self.offset_start_time:
            # Also remove sites from the next hour
            invalid_sites = invalid_sites | set(self.failed_items["sites"][t0.floor("60min") + pd.Timedelta(hours=1)])

        valid_sites = list(all_sites - invalid_sites)
        sites = np.random.choice(valid_sites, size=self.batch_size, replace=False)

        batch = {}
        for feature, feature_data in data.items():
            if feature in ["pv", "targets"]:
                pv = feature_data[sites].values.T
                batch[feature] = pv
            elif feature == "time":
                batch["time"] = np.repeat(feature_data, self.batch_size)
            elif feature == "metadata":
                batch["metadata"] = feature_data.loc[sites].values
            elif feature in ["hrv", "weather", "nonhrv"]:
                batch_data = []
                if feature == "weather":
                    ub = self._weather_ub
                    lb = self._weather_lb
                elif feature == "hrv":
                    ub = self._hrv_ub
                    lb = self._hrv_lb
                elif feature == "nonhrv":
                    ub = self._nonhrv_ub
                    lb = self._nonhrv_lb
                else:
                    ub = 64
                    lb = 64
                for site_id in sites:
                    # Retreive site indices and crop
                    x, y = self.metadata.loc[site_id, feature]
                    batch_data.append(feature_data[:, :, y - lb : y + ub, x - lb : x + ub])  # (C, N, H, W)
                batch[feature] = np.stack(batch_data)
            
            ### aerosol logic 
            elif feature == 'aerosols':
                batch_data = []
                for site_id in sites:
                    x, y = self.metadata.loc[site_id, feature]
                    # for now I'm not doing a full window cause its too much data
                    batch_data.append(feature_data[:, :, :, y, x]) 
                batch[feature] = np.stack(batch_data)
            elif feature == "pv_features":
                key_cols = ["latitude", "longitude", "orientation", "tilt", "kwp"]
                batch_data = []
                df = self.metadata.loc[sites.tolist(), key_cols]
                for _, (lat, lon, orient, tilt, kwp) in df.iterrows():
                    batch_data.append(feature_data[(lat, lon, orient, tilt, kwp)])
                batch[feature] = np.stack(batch_data)
            else:
                raise ValueError(f"Feature '{feature}' not supported")
        return batch
    
    def _get_features(self, t0: datetime) -> Dict[str, np.ndarray]:
        """Loads data as numpy arrays into memory for each feature.
        Use the given t0 to generate the 1 hour history time and 4 hour prediction windows.
        Select the features from their respective data structures into memory.
        Perform shape checking on the time dimension.
        Reshape outputs to shape (C, N, H, W)

        "pv", "hrv", "nonhrv" are taken at 5-minute intervals for the past hour.
        "weather", "aerosols" forecasts are taken at 1-hour intervals from past hour to 4 hours in the future, endpoints included.

        Parameters
        ----------
        t0 : datetime
            Initial time to select data from.

        Returns
        -------
        dict[str, np.ndarray]
            Output data, keys are feature names, values are numpy arrays.
        """
        
        # pv, hrv, nonhrv are provided from first hour
        first_hour = get_hour_slice(t0)
        target_time = get_target_slice(t0)

        # aerosols, nwp in 1-hour intervals
        forecast_times = get_forecast_slice(t0)
        
        data = self.data[(t0.year, t0.month)]
        data_slice = {}
        
        # Always get pv data
        pv = data["pv"].xs(first_hour, drop_level=False)["power"]
        pv = pv.reset_index().pivot(index="timestamp", columns="ss_id", values="power")
        targets: pd.DataFrame = data["pv"].xs(target_time, drop_level=False)["power"]
        targets = targets.reset_index().pivot(index="timestamp", columns="ss_id", values="power")
        
        data_slice["pv"] = pv
        data_slice["targets"] = targets

        if "hrv" in data:
            hrv = data["hrv"]["data"].sel(time=first_hour)  # (N, H, W, 1)
            data_slice["hrv"] = hrv.to_numpy().transpose(3, 0, 1, 2)  # (N, H, W, 1) -> (1, N, H, W)
        if (nonhrv_features := self.features.get("nonhrv", None)):
            dataset = data["nonhrv"]
            if nonhrv_features is True:
                nonhrv_features = dataset.channel.to_numpy().tolist()
            nonhrv_data = []
            for feature in nonhrv_features:
                nonhrv_data.append(dataset["data"].sel(time=first_hour, channel=feature).to_numpy()[..., ::-1, :])  # (C, N, H, W), x-dims are reversed in non-hrv
            nonhrv_data = np.stack(nonhrv_data)
            data_slice["nonhrv"] = nonhrv_data
        if (weather_features := self.features.get("weather", None)):
            dataset = data["weather"].sel(time=forecast_times)
            if weather_features is True:
                weather_features = list(dataset.keys())
            weather_data = []
            for feature in weather_features:
                weather_data.append(dataset[feature].to_numpy())
            weather_data = np.stack(weather_data)
            data_slice["weather"] = weather_data

        ### aerosol logic 
        if (aerosol_features := self.features.get("aerosols", None)):
            dataset = data['aerosols'].sel(time=forecast_times)
            if aerosol_features is True:
                aerosol_features = list(dataset.data_vars)
            aerosol_data = []
            for feature in aerosol_features:
                aerosol_data.append(dataset[feature].to_numpy())
            aerosol_data = np.stack(aerosol_data)
            data_slice['aerosols'] = aerosol_data
        ###

        if (metadata_features := self.features.get("metadata", None)):
            if metadata_features is True:
                metadata_features = ["time", "latitude", "longitude", "orientation", "tilt", "kwp"]
            if "time" in metadata_features:
                data_slice["time"] = np.array(t0, dtype="datetime64[ns]")
                other_metadata = [x for x in metadata_features if x != "time"]
            else:
                other_metadata = metadata_features
            data_slice["metadata"] = self.metadata[other_metadata].astype("float32")
        if self.features.get("pv_features", None):
            data_slice["pv_features"] = self.pv_features[t0.month]
        return data_slice
    
    def __iter__(self):
        """Iterate through the dataset
        On start iteration, shuffle data.
        The iterator lazily retreives batches, and skips failed ones.
        Failed batches are recorded so those timepoints are not used in future iterations.

        Yields
        ------
        dict[str, np.array]
            One batch of data
        """
        if self.shuffle:
            np.random.shuffle(self.initial_times)
        for i in range(len(self)):
            yield self[i]

def get_datasets(
    data_dir: str,
    date_range: Tuple[datetime, datetime],
    test_size: Optional[float] = None,
    seed: Optional[int] = None,
    batch_size: int = 32,
    hrv: Optional[bool] = True,
    nonhrv: Optional[Union[List[str], bool]] = None,
    weather: Optional[Union[List[str], bool]] = None,
    aerosols: Optional[Union[List[Tuple[str, int]], bool]] = None,
    metadata: Optional[Union[List[str], bool]] = None,
    pv_features_file: Optional[str] = None, 
    shuffle: bool = True,
    hrv_crop: int = 128,
    weather_crop: int = 128,
    nonhrv_crop: int = 128,
    zipped=True,
    offset_start_time=False
) -> Tuple[ClimatehackDataset]:
    """Convenience function to make training and test datasets.

    Parameters
    ----------
    data_dir : str
        Absolute path to data files
    date_range : tuple[datetime, datetime]
        (start date, end date) interval for retreiving training data
    test_size : Optional[float], optional
        Float between [0, 1] for fraction of test set, by default None
    seed : Optional[int], optional
        Seed for random splitting and shuffling, by default None
    batch_size : int, optional
        Batch size of generated batches, by default 32
    hrv : Optional[bool], optional
        Whether to include hrv data, by default True
    nonhrv : Optional[Union[list[str], bool]], optional
        Whether to include nonhrv data or list of nonhrv features to include, by default None
    weather : Optional[Union[list[str], bool]], optional
        Whether to include weather data or list of weather features to include, by default None
    aerosols : Optional[Union[list[tuple(str, int)], bool]], optional
        Whether to include aerosols data or list of (feature name, altitude) to include, by default None
    metadata : Optional[Union[list[str], bool]], optional
        Whether to include metadata or list of metadata features to include, by default None
    shuffle : bool, optional
        Whether to shuffle the datasets, by default True
    hrv_crop: int, by default 128
        Height/width of HRV data to crop.
    weather_crop: int, by default 128
        Height/width of weather data to crop.
    Returns
    -------
    tuple[ClimatehackDataset]
        (train dataset, test dataset) tuple of ClimatehackDataset
    """
    features = {}
    if hrv:
        features["hrv"] = hrv
    if nonhrv:
        features["nonhrv"] = nonhrv
    if weather:
        features["weather"] = weather
    if aerosols:
        features["aerosols"] = aerosols
    if metadata:
        features["metadata"] = metadata
    if pv_features_file is not None:
        features["pv_features"] = True
    
    file_times, all_dates = _get_all_dates(*date_range)
    data = _load_data(data_dir, file_times, features, zipped)
    metadata = _load_metadata(data_dir)
    if pv_features_file:
        pv_features = _load_pv_features(data_dir, pv_features_file)
    else:
        pv_features = None
    
    d0 = date_range[0].strftime("%Y-%m-%d")
    d1 = date_range[1].strftime("%Y-%m-%d")
    datatypes = "_".join([k for k in features.keys() if k in ["hrv", "nonhrv", "weather", "aerosols"]])
    checkpoint_name = f"{d0}_to_{d1}_{datatypes}_checkpoint.pkl"
    checkpoint_path = os.path.join(data_dir, "data_checking", checkpoint_name)
    if os.path.isfile(checkpoint_path):
        print("Loading dataset checking checkpoint")
        with open(checkpoint_path, "rb") as f:
            valid_dates, failed_items = pickle.load(f)
    else:
        print("No checkpoint, checking all data")
        valid_dates, failed_items = _check_data(data, all_dates, metadata, features, batch_size)
        with open(checkpoint_path, "wb") as f:
            pickle.dump((valid_dates, failed_items), f)

    train_dates, test_dates = _train_test_split(valid_dates, test_size, seed)

    train_ds = ClimatehackDataset(
        data,
        metadata,
        train_dates,
        features,
        failed_items,
        batch_size=batch_size,
        seed=seed,
        pv_features=pv_features,
        shuffle=shuffle,
        hrv_crop=hrv_crop,
        weather_crop=weather_crop,
        nonhrv_crop=nonhrv_crop,
        offset_start_time=offset_start_time
    )

    if test_dates:
        test_ds = ClimatehackDataset(
            data,
            metadata,
            test_dates,
            features,
            failed_items,
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
            pv_features=pv_features,
            hrv_crop=hrv_crop,
            weather_crop=weather_crop,
            nonhrv_crop=nonhrv_crop,
            offset_start_time=offset_start_time
        )
        return train_ds, test_ds
    else:
        return train_ds


def _get_all_dates(d0: datetime, d1: datetime) -> Tuple[List[Tuple[int, int]], List[datetime]]:
    """Generates a list of (year, month) pairs for filenames and list of datetimes to pass to dataset
    
    Parameters
    ----------
    d0 : datetime
        Start date of interval
    d1 : datetime
        End date of interval

    Returns
    -------
    list[tuple[int, int]], list[datetime]
    """

    assert datetime(2020, 1, 1) <= d0 < d1 <= datetime(2021, 12, 31), \
        "Dates must be between Jan 1, 2020 and Dec 31, 2021"
    
    # Files are split by month
    file_times = []
    if (y0 := d0.year) < d1.year:  # Only 2020 and 2021
        file_times.extend((y0, m) for m in range(d0.month, 13))
        file_times.extend((y0 + 1, m) for m in range(1, d1.month + 1))
    else:
        file_times.extend((y0, m) for m in range(d0.month, d1.month + 1))

    # Get list of all dates between d0 and d1
    all_dates = list(accumulate(repeat(timedelta(days=1), times=(d1 - d0).days), initial=d0))
    return file_times, all_dates


def _load_data(data_dir: str, file_times: List[Tuple[datetime]], features: Dict[str, Any], zipped: bool) -> Dict[str, Any]:
    """Loads relevant files from disk into memory

    Parameters
    ----------
    data_dir : str
        Absolute path to data directory
    file_times : list[tuple[datetime]]
        List of (year, month) dates specifying which files to load
    features : dict[str, Any]
        Only need the keys to specify which files to load. Subset of ["hrv", "nonhrv", "weather", "aerosols"]
    zipped : bool
        Whether the data is zipped or not

    Returns
    -------
    dict[str, Any]
        _description_
    """
    if zipped:
        file_paths = {
            "hrv": "satellite-hrv/<year>/<month>.zarr.zip",
            "nonhrv": "satellite-nonhrv/<year>/<month>.zarr.zip",
            "weather": "weather/<year>/<month>.zarr.zip",
            "aerosols": "aerosols/<year>/<month>.zarr.zip"
        }
    else:
        file_paths = {
            "hrv": "satellite-hrv/<year>/<month>.zarr",
            "nonhrv": "satellite-nonhrv/<year>/<month>.zarr",
            "weather": "weather/<year>/<month>.zarr",
            "aerosols": "aerosols/<year>/<month>.zarr"
        }

    data = {}
    for (year, month) in file_times:
        # Always load PV data
        pv_path = os.path.join(data_dir, f"pv/{year}/{month}.parquet")
        data.setdefault((year, month), {})["pv"] = pd.read_parquet(pv_path).drop("generation_wh", axis=1)
        # Load only specified features
        for feature in features:
            if feature in ["metadata", "pv_features"]:
                continue
            path = file_paths[feature].replace("<year>", str(year)).replace("<month>", str(month))
            path = os.path.join(data_dir, path)
            data.setdefault((year, month), {})[feature] = xr.open_dataset(path, engine="zarr", chunks="auto")
    return data

def _load_metadata(data_dir: str) -> Tuple[pd.DataFrame]:
    site_indices = pd.read_json(os.path.join(data_dir, "indices.json"))
    metadata = pd.read_csv(os.path.join(data_dir, "pv/metadata.csv"))
    inds_to_keep = site_indices.index
    mask = metadata["ss_id"].isin(inds_to_keep)

    metadata = metadata.loc[mask].reset_index(drop=True).drop(["llsoacd", "operational_at"], axis=1)
    metadata = metadata.rename({"latitude_rounded": "latitude", "longitude_rounded": "longitude"}, axis=1)
    metadata = metadata.merge(site_indices, left_on="ss_id", right_index=True)
    metadata.index = metadata["ss_id"].values
    metadata = metadata.drop("ss_id", axis=1)
    return metadata

def _load_pv_features(data_dir: str, file_name: str) -> Dict:
    path = os.path.join(data_dir, file_name)
    with open(path, "rb") as f:
        pv_features = pickle.load(f)
    return pv_features

def _get_times(dates: List[datetime]):
    # Add time and intervals, start at 4am and go to 6pm (6pm + 4h = 10pm)
    start_time = timedelta(hours=4)
    duration = 14
    initial_times = chain.from_iterable(accumulate(repeat(timedelta(hours=1), duration), initial=dt + start_time) for dt in dates)
    initial_times = pd.Series(initial_times)
    return initial_times


def _check_data(data, all_dates, metadata, features, batch_size):
    """Checks data for NaNs by timestep (and site if pv) and lengths of time slices.
    Rejects dates that have no valid times for random shuffling of dates.
    """
    all_sites = set(metadata.index)
    initial_times = _get_times(all_dates)
    failed_items = {
        "dates": {},
        "times": {},
        "sites": {}
    }

    initial_times = pd.Series(initial_times)
    ts_index = pd.DatetimeIndex(initial_times)

    for (y, m), data_dict in data.items():
        filtered_times = initial_times.loc[(ts_index.month == m) & (ts_index.year == y)]
        # Always check pv
        print(f"Checking pv for {y}-{m}")
        failed_items = _check_pv(data_dict["pv"], filtered_times, failed_items, all_sites, batch_size)
        if "hrv" in features:
            print(f"Checking hrv for {y}-{m}")
            failed_items = _check_hrv(data_dict["hrv"], filtered_times, failed_items)
        if "nonhrv" in features:
            print(f"Checking nonhrv for {y}-{m}")
            failed_items = _check_nonhrv(data_dict["nonhrv"], filtered_times, failed_items)
        if "weather" in features:
            print(f"Checking weather for {y}-{m}")
            failed_items = _check_weather(data_dict["weather"], filtered_times, failed_items)
        if "aerosols" in features:
            print(f"Checking aerosols for {y}-{m}")
            failed_items = _check_aerosols(data_dict["aerosols"], filtered_times, failed_items)

    failed_times = sorted(failed_items["times"])
    duration = 14

    for t in failed_times:
        date = datetime(*t.timetuple()[:3])
        failed_items["dates"].setdefault(date, []).append(t)
    remove_dates = []
    for k, v in failed_items["dates"].items():
        if len(v) >= duration + 1:
            remove_dates.append(pd.Timestamp(k))

    valid_dates = [d for d in all_dates if d not in remove_dates]
    return valid_dates, failed_items


def _check_pv(pv: pd.DataFrame, times, failed_items, all_sites, batch_size):
    for t0 in times:
        a = pv.xs(get_pv_slice(t0), drop_level=False)
        a = a.reset_index().pivot(index="timestamp", columns="ss_id", values="power")
        # Ensure all historical and target data is right shape
        if a.shape[0] != 60:
            failed_items["times"].setdefault(t0, set()).update(["pv bad shape"])
        else:
            # Check for NaNs by site or for missing site data
            bad_sites = set(a.columns[np.argwhere(a.isnull().any().values).flatten()])
            missing_sites = all_sites - set(a.columns)
            bad_sites = bad_sites.union(missing_sites)
            # Reject times with insufficient site data
            if len(all_sites) - len(bad_sites) < batch_size:
                failed_items["times"].setdefault(t0, set()).update(["not enough good pv sites"])
            # Record few failed sites
            else:
                failed_items["sites"].setdefault(t0, set()).update(bad_sites)
                failed_items["sites"][t0].update(missing_sites)
    return failed_items


def _check_hrv(hrv: xr.Dataset, times, failed_items):
    hrv_ts = hrv["time"].to_pandas()
    
    # Check valid time shape
    for t0 in times:
        if len(hrv_ts[get_hour_slice(t0)]) != 12:
            failed_items["times"].setdefault(t0, set()).update(["hrv bad shape"])
    
    # Check NaNs
    nans = np.array([hrv.isnull().any(dim=("x_geostationary", "y_geostationary"))["data"]]).flatten()
    bad_inds = np.argwhere(nans).flatten()
    for i in bad_inds:
        t0 = hrv_ts.iloc[i]
        # Need to take bad ind and preceeding 12 timesteps (1 hour) as invalid
        # This is overall rounding up to the next hour and marking that as as invalid
        t0 = t0.ceil("60min")  # Round up to nearest hour
        failed_items["times"].setdefault(t0, set()).update(["hrv nan"])
    return failed_items


def _check_nonhrv(nonhrv: xr.Dataset, times, failed_items):
    nonhrv_ts = nonhrv["time"].to_pandas()
    # Check valid time shape
    for t0 in times:
        if len(nonhrv_ts[get_hour_slice(t0)]) != 12:
            failed_items["times"].setdefault(t0, set()).update(["nonhrv bad shape"])
    # Check NaNs
    nans = np.array([nonhrv.isnull().any(dim=("x_geostationary", "y_geostationary", "channel"))["data"]]).flatten()
    bad_inds = np.argwhere(nans).flatten()
    for i in bad_inds:
        t0 = nonhrv_ts.iloc[i]
        # Need to take bad ind and preceeding 12 timesteps (1 hour) as invalid
        # This is overall rounding up to the next hour and marking that as as invalid
        t0 = t0.ceil("60min")  # Round up to nearest hour
        failed_items["times"].setdefault(t0, set()).update(["nonhrv nan"])
    return failed_items


def _check_weather(weather: xr.Dataset, times, failed_items):
    nwp_ts = weather["time"].to_pandas()
    # Check for valid time shape
    for t0 in times:
        if len(nwp_ts[get_forecast_slice(t0)]) != 6:
            failed_items["times"].setdefault(t0, set()).update(["nwp bad shape"])
    # Check NaNs by timestamp
    keys = list(weather.keys())
    nans = np.array([weather.isnull().any(dim=("latitude", "longitude"))[key] for key in keys])
    bad_inds = np.argwhere(nans)
    for i in np.unique(bad_inds[:, 1]):  # Only take unique time indices
        t0 = nwp_ts.iloc[i]
        for td in [pd.Timedelta(hours=h) for h in range(-4, 2)]:  # From t0+4 and t0-1, endpoint excluded
            failed_items["times"].setdefault(t0 + td, set()).update(["nwp nans"])
    return failed_items


def _check_aerosols(aerosols: xr.Dataset, times, failed_items):
    aerosols_ts = aerosols["time"].to_pandas()
    # Check for valid time shape
    for t0 in times:
        if len(aerosols_ts[get_forecast_slice(t0)]) != 6:
            failed_items["times"].setdefault(t0, set()).update(["aerosol bad shape"])
    # Check NaNs by timestamp
    keys = list(aerosols.keys())
    nans = np.array([aerosols.isnull().any(dim=("latitude", "longitude"))[key] for key in keys])
    bad_inds = np.argwhere(nans)
    for i in np.unique(bad_inds[:, 1]):  # Only take unique time indices
        t0 = aerosols_ts.iloc[i]
        for td in [pd.Timedelta(hours=h) for h in range(-4, 2)]:  # From t0+4 and t0-1, endpoint excluded
            failed_items["times"].setdefault(t0 + td, set()).update(["aerosol nans"])
    return failed_items


def get_forecast_slice(t0):
    t0_hour = t0.floor("60min")  # No-op for times already on the hour
    forecast_times = slice(str(t0_hour - timedelta(hours=1)), str(t0_hour + timedelta(hours=4)))
    return forecast_times


def get_hour_slice(t0):
    first_hour = slice(str(t0 - timedelta(minutes=60)), str(t0 - timedelta(minutes=5)))
    return first_hour


def get_target_slice(t0):
    target_time = slice(str(t0), str(t0 + timedelta(hours=3, minutes=55)))
    return target_time


def get_pv_slice(t0):
    """All PV data for nan checking"""
    return slice(str(t0 - timedelta(minutes=60)), str(t0 + timedelta(hours=3, minutes=55)))


def _train_test_split(all_dates: List[datetime], test_size: float, seed: int) -> Tuple[List[datetime]]:
    """Splits dates into train and test set

    Parameters
    ----------
    all_dates : list[datetime]
        List of all possible dates
    test_size : fload
        Number between 0 and 1 specifying fraction of data for test dataset
    seed : int
        Seed for random number generator

    Returns
    -------
    tuple[list[datetime]]
        List of dates in (train, test) datasets
    """
    rng = np.random.default_rng(seed)
    rng.shuffle(all_dates)
    if test_size is not None:
        assert 0 < test_size < 1, "test_size must be between 0 and 1"
        cutoff = int(len(all_dates) * test_size)
        train_dates = all_dates[cutoff:]
        test_dates = all_dates[:cutoff]
    else:
        train_dates = all_dates
        test_dates = None
    return train_dates, test_dates
