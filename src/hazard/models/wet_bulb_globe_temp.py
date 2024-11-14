"""Module for calculating indicators of days exceeding wet-bulb globe temperature (WBGT) thresholds."""

import logging
import os
from contextlib import ExitStack
from pathlib import PurePosixPath
from typing_extensions import Iterable, List, Optional, override

import numpy as np
import xarray as xr

from hazard.inventory import Colormap, HazardResource, MapInfo, Scenario
from hazard.models.multi_year_average import (
    BatchItem,
    Indicator,
    ThresholdBasedAverageIndicator,
)
from hazard.protocols import OpenDataset
from hazard.sources.nex_gddp_cmip6 import NexGddpCmip6
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities.tiles import create_tiles_for_resource

logger = logging.getLogger(__name__)


class WetBulbGlobeTemperatureAboveIndicator(ThresholdBasedAverageIndicator):
    """Calculates days exceeding specified WBGT temperature thresholds based on climate data for various scenarios and years, storing results for global heat risk assessment.

    Attributes
        threshold_temps_c : List[float]
            Temperature thresholds in Celsius for calculating exceedance days.
        window_years : int
            Number of years over which to average data.
        gcms : Iterable[str]
            Climate models to use for data analysis.
        scenarios : Iterable[str]
            Emission scenarios to consider for projections.
        central_year_historical : int
            Central year for historical scenario calculations.
        central_years : Iterable[int]
            Central years for projected scenario calculations.

    """

    def __init__(
        self,
        threshold_temps_c: Optional[List[float]] = None,
        window_years: int = 20,
        gcms: Optional[Iterable[str]] = None,
        scenarios: Optional[Iterable[str]] = None,
        central_year_historical: int = 2005,
        central_years: Optional[Iterable[int]] = None,
    ):
        """Initialize the WBGT indicator with specified thresholds, climate models, and scenarios.

        Args:
            threshold_temps_c : List[float]
                Temperature thresholds for exceedance calculations in Celsius.
            window_years : int
                Number of years for averaging data.
            gcms : Iterable[str]
                Global Climate Models to include.
            scenarios : Iterable[str]
                Emission scenarios to evaluate.
            central_year_historical : int
                Central year for historical averages.
            central_years : Iterable[int]
                Target years for future scenario calculations.

        """
        if central_years is None:
            central_years = [2030, 2040, 2050, 2060, 2070, 2080, 2090]
        if scenarios is None:
            scenarios = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
        if gcms is None:
            gcms = [
                "ACCESS-CM2",
                "CMCC-ESM2",
                "CNRM-CM6-1",
                "MPI-ESM1-2-LR",
                "MIROC6",
                "NorESM2-MM",
            ]
        if threshold_temps_c is None:
            threshold_temps_c = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        super().__init__(
            window_years=window_years,
            gcms=gcms,
            scenarios=scenarios,
            central_year_historical=central_year_historical,
            central_years=central_years,
        )
        self.threshold_temps_c = threshold_temps_c

    @override
    def prepare(self, force, download_dir, force_download):
        return super().prepare(force, download_dir, force_download)

    def _calculate_single_year_indicators(
        self, source: OpenDataset, item: BatchItem, year: int
    ) -> List[Indicator]:
        logger.info(f"Starting calculation for year {year}")
        with ExitStack() as stack:
            tas = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "tas", year)
            ).tas
            hurs = stack.enter_context(
                source.open_dataset_year(item.gcm, item.scenario, "hurs", year)
            ).hurs
            results = self._days_wbgt_above_indicators(tas, hurs)
        path = item.resource.path.format(
            gcm=item.gcm, scenario=item.scenario, year=item.central_year
        )
        assert isinstance(item.resource.map, MapInfo)
        result = [
            Indicator(
                results,
                PurePosixPath(path),
                item.resource.map.bounds,
            )
        ]
        logger.info(f"Calculation complete for year {year}")
        return result

    def _days_wbgt_above_indicators(
        self, tas: xr.DataArray, hurs: xr.DataArray
    ) -> xr.DataArray:
        """Create DataArrays containing indicators the thresholds for a single year."""
        tas_c = tas - 273.15  # convert from K to C
        # vpp is water vapour partial pressure in kPa
        vpp = (hurs / 100.0) * 6.105 * np.exp((17.27 * tas_c) / (237.7 + tas_c))
        wbgt = 0.567 * tas_c + 0.393 * vpp + 3.94
        scale = 365.0 / len(wbgt.time)
        if any(coord not in wbgt.coords.keys() for coord in ["lat", "lon", "time"]):
            raise ValueError("expect coordinates: 'lat', 'lon' and 'time'")
        coords = {
            "index": self.threshold_temps_c,
            "lat": wbgt.coords["lat"].values,
            "lon": wbgt.coords["lon"].values,
        }
        output = xr.DataArray(coords=coords, dims=coords.keys())
        for i, threshold_c in enumerate(self.threshold_temps_c):
            output[i, :, :] = xr.where(wbgt > threshold_c, scale, 0.0).sum(dim=["time"])
        return output

    def _onboard_single(self, target, download_dir):
        source = NexGddpCmip6()
        self.run_all(source, target)
        self.create_maps(target, target)

    def create_maps(self, source: OscZarr, target: OscZarr):
        """Create map images."""
        create_tiles_for_resource(source, target, self._resource())

    def _resource(self) -> HazardResource:
        """Create resource."""
        with open(
            os.path.join(os.path.dirname(__file__), "wet_bulb_globe_temp.md"), "r"
        ) as f:
            description = f.read().replace("\u00c2\u00b0", "\u00b0")
        resource = HazardResource(
            hazard_type="ChronicHeat",
            indicator_id="days_wbgt_above",
            indicator_model_id=None,
            indicator_model_gcm="{gcm}",
            params={"gcm": list(self.gcms)},
            path="chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}",
            display_name="Days with wet-bulb globe temperature above threshold in °C/{gcm}",
            description=description,
            display_groups=[
                "Days with wet-bulb globe temperature above threshold in °C"
            ],  # display names of groupings
            group_id="",
            map=MapInfo(  # type: ignore[call-arg] # has a default value for bbox
                colormap=Colormap(
                    name="heating",
                    nodata_index=0,
                    min_index=1,
                    min_value=0.0,
                    max_value=52,
                    max_index=255,
                    units="days/year",
                ),
                bounds=[(-180.0, 85.0), (180.0, 85.0), (180.0, -85.0), (-180.0, -85.0)],
                bbox=[-180.0, -85.0, 180.0, 85.0],
                path="maps/chronic_heat/osc/v2/days_wbgt_above_{gcm}_{scenario}_{year}_map",
                index_values=self.threshold_temps_c,
                source="map_array_pyramid",
            ),
            units="days/year",
            scenarios=[
                Scenario(id="historical", years=[self.central_year_historical]),
                Scenario(id="ssp126", years=list(self.central_years)),
                Scenario(id="ssp245", years=list(self.central_years)),
                Scenario(id="ssp370", years=list(self.central_years)),
                Scenario(id="ssp585", years=list(self.central_years)),
            ],
        )
        return resource
