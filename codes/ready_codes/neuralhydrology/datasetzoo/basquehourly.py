from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import xarray

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class BasqueHourly(BaseDataset):
    """Custom dataset class for Basque hourly hydrological data."""

    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(BasqueHourly, self).__init__(cfg=cfg,
                                           is_train=is_train,
                                           period=period,
                                           basin=basin,
                                           additional_features=additional_features,
                                           id_to_int=id_to_int,
                                           scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load hourly time series data for a given basin."""
        csv_path = Path(self.cfg.data_dir) / "hourly" / f"{basin}.txt"
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found for basin: {basin}")

        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index("date")

        # Replace invalid values with NaNs
        for col in ['streamflowmean']: #, 'levelmean', 'streamflowinst', 'levelinst']:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_basque_attributes(self.cfg.data_dir, basins=self.basins)


def load_basque_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    """Load static attributes for Basque hourly catchments."""
    attr_path = Path(data_dir) / 'URA_attributes_v1.0'
    if not attr_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attr_path}")

    txt_files = attr_path.glob('URA_*.txt')

    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    if basins:
        missing = [b for b in basins if b not in df.index]
        if missing:
            raise ValueError(f'Missing attributes for basins: {missing}')
        df = df.loc[basins]

    return df
