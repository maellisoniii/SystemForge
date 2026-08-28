from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd




@dataclass
class EnergyDataProfile:
    """Generic hourly energy profile used by the optimization engines.

    This class is the data adapter layer for the model. A source can be a
    national dataset, building meter export, tariff CSV, synthetic profile,
    or direct arrays.

    Core fields:
    - load_kwh
    - solar_kwh
    - price_per_kwh
    - solar_capacity_factor
    - timestamps

    solar_kwh represents an already-scaled hourly solar generation profile.

    solar_capacity_factor represents hourly solar availability per unit of
    installed solar capacity and is used by capacity co-optimization models.
    """

    name: str
    load_kwh: np.ndarray
    solar_kwh: np.ndarray
    price_per_kwh: np.ndarray
    solar_capacity_factor: Optional[np.ndarray] = None
    timestamps: Optional[pd.Series] = None
    source: str = ""
    analysis_period: str = ""

    @staticmethod
    def _clean_numeric_series(dataframe, column, default_value, multiplier, offset):
        """Return a numeric NumPy array from a source column or constant value."""
        if column is None:
            if default_value is None:
                raise ValueError("required data column is missing from the profile config")
            series = pd.Series(default_value, index=dataframe.index, dtype="float64")
        else:
            series = pd.to_numeric(dataframe[column], errors="coerce")

        series = (
            series
            .interpolate(limit_area="inside")
            .ffill()
            .bfill()
        )

        return series.values * multiplier + offset

    @classmethod
    def from_csv(
            cls,
            name,
            csv_path,
            load_column,
            price_column=None,
            solar_column=None,
            timestamp_column=None,
            analysis_year=None,
            load_multiplier=1.0,
            solar_multiplier=1.0,
            price_multiplier=1.0,
            price_offset=0.0,
            flat_solar_kwh=0.0,
            flat_price_per_kwh=None):
        """Load any CSV profile by mapping source columns into common fields.

        Use multipliers to normalize units. For example:
        - MW for one hour -> kWh: multiply by 1000
        - EUR/MWh -> EUR/kWh: multiply by 1 / 1000
        - A representative system scale can be folded into the load/solar
          multipliers without changing the analysis engine.
        """
        csv_path = Path(csv_path)
        dataframe = pd.read_csv(csv_path)
        timestamps = None
        analysis_period = "full dataset"

        if timestamp_column is not None:
            timestamps = pd.to_datetime(dataframe[timestamp_column], utc=True)
            if analysis_year is not None:
                dataframe = dataframe.loc[timestamps.dt.year == analysis_year].copy()
                timestamps = timestamps.loc[dataframe.index]
                analysis_period = str(analysis_year)

        load_kwh = cls._clean_numeric_series(
            dataframe,
            load_column,
            default_value=None,
            multiplier=load_multiplier,
            offset=0.0,
        )
        solar_kwh = cls._clean_numeric_series(
            dataframe,
            solar_column,
            default_value=flat_solar_kwh,
            multiplier=solar_multiplier,
            offset=0.0,
        )
        price_per_kwh = cls._clean_numeric_series(
            dataframe,
            price_column,
            default_value=flat_price_per_kwh,
            multiplier=price_multiplier,
            offset=price_offset,
        )

        profile = cls(
            name=name,
            load_kwh=load_kwh,
            solar_kwh=solar_kwh,
            price_per_kwh=price_per_kwh,
            timestamps=timestamps,
            source=str(csv_path),
            analysis_period=analysis_period,
        )
        profile.validate()
        return profile

    @classmethod
    def from_arrays(
            cls,
            name,
            load_kwh,
            price_per_kwh,
            solar_kwh=None,
            solar_capacity_factor=None,
            timestamps=None,
            source="manual arrays",
            analysis_period="custom"):
        """Build a profile directly from arrays instead of a CSV file."""
        if solar_kwh is None:
            solar_kwh = np.zeros(len(load_kwh))

        profile = cls(
            name=name,
            load_kwh=np.asarray(load_kwh, dtype=float),
            solar_kwh=np.asarray(solar_kwh, dtype=float),
            price_per_kwh=np.asarray(price_per_kwh, dtype=float),
            solar_capacity_factor=(
                None
                if solar_capacity_factor is None
                else np.asarray(
                    solar_capacity_factor, 
                    dtype=float
                    )
            ),
            timestamps=timestamps,
            source=source,
            analysis_period=analysis_period,
        )
        profile.validate()
        return profile


    def validate(self):
        profile_lengths = {
            len(self.load_kwh),
            len(self.solar_kwh),
            len(self.price_per_kwh),
        }

        if len(profile_lengths) != 1:
            raise ValueError("load, solar, and price profiles must have the same length")
        if len(self.load_kwh) == 0:
            raise ValueError("energy profile cannot be empty")
        if not np.all(np.isfinite(self.load_kwh)):
            raise ValueError("load profile contains missing or non-finite values")
        if not np.all(np.isfinite(self.solar_kwh)):
            raise ValueError("solar profile contains missing or non-finite values")
        if not np.all(np.isfinite(self.price_per_kwh)):
            raise ValueError("price profile contains missing or non-finite values")
        if np.sum(self.load_kwh) <= 0:
            raise ValueError("annual load must be greater than zero")
        if self.solar_capacity_factor is not None:
            if len(self.solar_capacity_factor) != len(
                self.load_kwh
            ):
                raise ValueError(
                    "solar_capacity_factor must have the "
                    "same length as the energy profile"
                )

            if not np.all(
                np.isfinite(self.solar_capacity_factor)
            ):
                raise ValueError(
                    "solar_capacity_factor contains "
                    "missing or non-finite values"
                )

            if np.any(self.solar_capacity_factor < 0):
                raise ValueError(
                    "solar_capacity_factor cannot be negative"
                )

            if np.any(self.solar_capacity_factor > 1):
                raise ValueError(
                    "solar_capacity_factor cannot exceed 1"
                )

    @property
    def annual_load(self):
        return np.sum(self.load_kwh)

    @property
    def annual_solar(self):
        return np.sum(self.solar_kwh)

    @property
    def average_load(self):
        return np.mean(self.load_kwh)

    @property
    def reference_solar_penetration(self):
        if self.annual_solar <= 0:
            raise ValueError(
                "reference solar profile has zero annual generation; provide a "
                "solar profile before sweeping solar penetration"
            )
        return self.annual_solar / self.annual_load
