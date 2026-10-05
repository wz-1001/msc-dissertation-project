"""
trackers.py

Same start()/stop() interface over CodeCarbon and a manual TDP-based
estimate, so benchmark.py doesn't care which one it's using. get_tracker()
falls back from CodeCarbon to the manual estimator automatically if
CodeCarbon can't initialise - happens on shared machines without
permission to read hardware power counters.
"""

from abc import ABC, abstractmethod
import time

ALLOWED_LIBRARIES = ("codecarbon", "manual")


class BaseEmissionsTracker(ABC):
    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def stop(self) -> float:
        ...


class CodeCarbonTracker(BaseEmissionsTracker):
    def __init__(self, output_dir="./output", **kwargs):
        from codecarbon import EmissionsTracker  # imported here, not at module level, so a missing install only breaks this tracker, not the whole file
        self._tracker = EmissionsTracker(
            output_dir=output_dir,
            log_level="error",
            save_to_file=False,
        )

    def start(self):
        self._tracker.start()

    def stop(self) -> float:
        emissions_kg = self._tracker.stop()
        return float(emissions_kg) if emissions_kg else 0.0


class ManualEstimateTracker(BaseEmissionsTracker):
    """
    Single-wattage simplification of Green Algorithms (Lannelongue et al.
    2021) - no per-core/per-GB terms, just one assumed system draw. Trades
    precision for not needing the user to know their own hardware topology.
    """

    DEFAULT_TDP_WATTS = 45.0
    DEFAULT_GRID_INTENSITY_G_PER_KWH = 130.0  # uk grid average - wrong for you if you're not in the uk, override it
    DEFAULT_PUE = 1.0  # 1.0 = no datacenter overhead, correct default for a personal laptop

    def __init__(self, tdp_watts: float = None, grid_intensity_g_per_kwh: float = None, pue: float = None, **kwargs):
        self.tdp_watts = tdp_watts if tdp_watts is not None else self.DEFAULT_TDP_WATTS
        self.grid_intensity_g_per_kwh = (
            grid_intensity_g_per_kwh if grid_intensity_g_per_kwh is not None else self.DEFAULT_GRID_INTENSITY_G_PER_KWH
        )
        self.pue = pue if pue is not None else self.DEFAULT_PUE
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        elapsed_hours = (time.time() - self._t0) / 3600.0
        kwh = (self.tdp_watts / 1000.0) * elapsed_hours * self.pue
        return kwh * self.grid_intensity_g_per_kwh / 1000.0


TRACKER_REGISTRY = {
    "codecarbon": CodeCarbonTracker,
    "manual": ManualEstimateTracker,
}


def get_tracker(library: str = "codecarbon", **kwargs) -> BaseEmissionsTracker:
    library = (library or "codecarbon").lower().strip()
    if library not in TRACKER_REGISTRY:
        raise ValueError(
            f"Unknown emissions library '{library}'. Choose one of: {', '.join(ALLOWED_LIBRARIES)}"
        )

    if library == "codecarbon":
        try:
            return CodeCarbonTracker(**kwargs)
        except Exception as e:
            # don't let a locked-down machine kill the whole run - just measure less precisely instead
            print(f"[trackers.py] CodeCarbon couldn't start ({e}) - falling back to the manual estimator for this run.")
            return ManualEstimateTracker(**kwargs)

    return TRACKER_REGISTRY[library](**kwargs)