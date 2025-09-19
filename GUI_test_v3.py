"""
v3 by Sophia (original Archie, Alika and Sophia): 
This should now call generate_cancer_type_map_v3.py to allow the gui to have the 
option to produce capacity-weighted treated and optimally treated maps

""" 

import sys
import os
import subprocess
from pathlib import Path
import xarray as xr
import pandas as pd 
import numpy as np


from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog, QHBoxLayout, QGroupBox, 
    QSplitter, QCheckBox, QScrollArea, QTextEdit, 
    QListWidget, QListWidgetItem, QTreeWidget, 
    QTreeWidgetItem, QHeaderView, QSpinBox,
    QDoubleSpinBox, QSlider
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, ScalarFormatter
from matplotlib import cm


import io
from PIL import Image
from pycountry import countries
import rasterio

from a_population_density.download_worldpop import download_worldpop
from a_population_density.resample_population import resample_population
from b_cancer_incidence.generate_cancer_type_map_v3 import generate_cancer_type_map
from b_cancer_incidence.generate_cancer_type_map_v3 import generate_population_density_map_only
from b_cancer_incidence.generate_cancer_type_map_v3 import (
    _load_rad_utilisation_csv as _rt_load_csv,
    _norm_key as _rt_norm_key,
    DEFAULT_OPTIMAL_RT_CSV,
    DEFAULT_ACTUAL_RT_DIR,
)

from b_cancer_incidence.generate_cancer_type_map_v3 import get_n_liancs_from_excel 
from c_probability_of_access.visualization.generate_access_map_v2 import generate_accessibility_plot

# ---- DEFAULTS ----

BASE_DIR = Path(__file__).resolve().parents[1]
#DEFAULT_XARRAY_PATH = BASE_DIR / "b_cancer_incidence" / "globocan_xarray.nc"
DEFAULT_XARRAY_PATH = "b_cancer_incidence/globocan_xarray.nc"
DEFAULT_METRIC_NAME = "New_Cases_Number"

# Sentinel (mutually exclusive) cancer buckets
SENTINEL_CANCERS = ("All cancers", "All cancers excl. NMSC")
_SENTINELS_NORM = {s.strip().casefold() for s in SENTINEL_CANCERS}
def _is_sentinel_cancer(name: str) -> bool:
        return name.strip().casefold() in _SENTINELS_NORM

# Map labels (keep in one place to avoid typos)
ACCESS_PROB_DIST = "Probability of Treatment Access (distance)"
ACCESS_PROB_POPW = "Population-Weighted Treatment Access (distance)"
ACCESS_DIST_NEAREST = "Distance to Nearest LINAC (km)"


# ==== All Qthreads below for resampling, downloading and mapping ====

class ResampleThread(QThread):
    """
    QThread worker that resamples a country’s population raster at a target resolution
    without blocking the GUI.

    This thread is a thin wrapper around `resample_population(...)`. When `run()` is
    executed, it calls that function with the constructor arguments and emits a single
    `finished` signal carrying the returned result dictionary.

    Signals:
        finished (dict): Emitted exactly once when processing completes (whether
            success or failure), with the payload being the dictionary returned by
            `resample_population`. The exact keys/structure are defined by that
            function (e.g., may include status, message, and output file path).      <-- CHECK ME

    Args:
        country_name (str): Human-readable country name passed to `resample_population`.
        resolution (float | int | str): Target spatial resolution (km) used for resampling.
        input_dir (str | pathlib.Path): Directory containing the raw WorldPop raster(s).
        output_dir (str | pathlib.Path): Directory where the resampled raster will be written.
        overwrite_resample (bool, optional): If True, allows overwriting an existing
            resampled file. Defaults to False.

    Notes:
        - Do not interact with Qt widgets directly inside this thread; update the UI
          in a slot connected to `finished` (executed on the main thread).
        - If `resample_population` raises an exception, the thread will terminate and
          no `finished` signal will be emitted. Wrap the call in a try/except block
          if you need guaranteed signaling on error.

    Example:
        thread = ResampleThread(country_name, resolution, input_dir, output_dir, overwrite)
        thread.finished.connect(self.on_resample_finished)
        thread.start()
    """ 
    finished = pyqtSignal(dict)  # Emits the full result dictionary

    def __init__(self, country_name, resolution, input_dir, output_dir, overwrite_resample=False):
        super().__init__()
        self.country_name = country_name
        self.resolution = resolution
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.overwrite_resample = overwrite_resample

    def run(self):
        result = resample_population(self.country_name, self.resolution, self.input_dir, self.output_dir, self.overwrite_resample)
        self.finished.emit(result)

class DownloadThread(QThread):
    """
    QThread worker that downloads a country’s WorldPop raster and reports progress.

    This thread wraps `download_worldpop(...)` to keep the GUI responsive. It bridges the
    downloader’s progress callback to a Qt signal and guarantees a terminal `finished`
    signal whether the operation succeeds or fails.

    Signals:
        progress_updated (int): Emitted periodically with the current progress value
            forwarded from `download_worldpop`’s callback. (Scale is defined by the
            downloader)
        finished (bool, str): Emitted once at completion or on error.
            bool: success flag
            str: message (e.g., output path or error text)

    Args:
        country_name (str): Human-readable country name to fetch.
        output_dir (str | pathlib.Path): Directory where the raster will be saved.
        overwrite_download (bool, optional): If True, existing files may be overwritten.
            Defaults to False.

    Threading & UI notes:
        - Do not manipulate Qt widgets inside this thread. Instead, connect the signals
          to slots on the main (GUI) thread, e.g.:
              thread = DownloadThread(country, outdir, overwrite)
              thread.progress_updated.connect(self.update_progress_bar)
              thread.finished.connect(self.on_download_finished)
              thread.start()
        - A `try/except` in `run()` ensures `finished(False, error)` is emitted on exceptions.
        - Treat this as a one-shot worker: create a new instance for each download.
    """
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, country_name, output_dir, overwrite_download=False):
        super().__init__()
        self.country_name = country_name
        self.output_dir = output_dir
        self.overwrite_download = overwrite_download

    def run(self):
        try:
            def progress_callback(progress):
                self.progress_updated.emit(progress)

            success, message = download_worldpop(
                self.country_name,
                self.output_dir,
                progress_callback,
                self.overwrite_download
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))

class PopulationMapThread(QThread):
    """
    QThread worker that renders a population-density map image from a raster and saves outputs.

    This thread wraps `generate_population_density_map_only(...)` so the (potentially
    heavy) raster I/O and rendering happen off the GUI thread. On success it emits the
    rendered image bytes (for immediate preview) and the paths to the written outputs.

    Signals:
        finished (bytes, str, str):
            Emitted once on success with:
                image_data: the in-memory image bytes (e.g., PNG) suitable for display      <-- CHECK ME
              after decoding (e.g., via QPixmap/QImage).
                tif_path: absolute/relative path of the (possibly re/exported) GeoTIFF.     <-- CHECK ME
                png_path: absolute/relative path of the rendered PNG.
        error (str):
            Emitted once if an exception occurs. The string contains a human-readable
            error message. In this case, `finished` is not emitted.

    Args:
        country_code (str): ISO-3 country code used by the generator (e.g., "gbr").
        resolution (float | int | str): Target output resolution in km (passed through).
        population_raster_path (str | pathlib.Path): Path to the population raster to
            visualize (typically the resampled raster).
        output_dir (str | pathlib.Path): Directory where outputs (GeoTIFF/PNG) are written.
        overwrite_existing (bool): Whether to overwrite existing outputs.

    Workflow:
        - Calls `generate_population_density_map_only(..., return_image=True,
          overwrite_existing=...)`.
        - On success, unpacks `(image_data, tif_path, png_path)` and emits `finished`.
        - On exception, emits `error(str(e))`.

    Threading & UI notes:
        - Do not update widgets inside this thread. Connect `finished`/`error` to slots
          on the main thread to update the UI.
        - Treat as a one-shot worker; create a new instance per request.

    Example:
        thread = PopulationMapThread(iso3, 1.0, raster_path, outdir, overwrite=True)
        thread.finished.connect(self.on_population_map_ready)
        thread.error.connect(self.on_population_map_error)
        thread.start()
    """
    finished = pyqtSignal(bytes, str, str)
    error = pyqtSignal(str)

    def __init__(self, country_code, resolution, population_raster_path, output_dir, overwrite_existing):
        super().__init__()
        self.country_code = country_code
        self.resolution = resolution
        self.population_raster_path = population_raster_path
        self.output_dir = output_dir
        self.overwrite_existing = overwrite_existing

    def run(self):
        try:
            image_data, tif_path, png_path = generate_population_density_map_only(
                country_code=self.country_code,
                population_raster_path=self.population_raster_path,
                output_dir=self.output_dir,
                resolution=self.resolution,
                return_image=True,
                overwrite_existing=self.overwrite_existing
            )
            self.finished.emit(image_data, tif_path, png_path)
        except Exception as e:
            self.error.emit(str(e))
                
class MapGenerationThread(QThread):
    """
    QThread worker that generates cancer-type maps from a population raster without blocking the GUI.

    This thread wraps `generate_cancer_type_map(...)`. It forwards parameters such as the
    selected cancer types, resolution, and output options to the generator, then emits:
    - `finished(image_bytes, tif_path, png_path)` on success, or
    - `error(message)` on failure.

    Signals:
        finished (bytes, str, str):
            Emitted once on success with:
                image bytes (for immediate preview via QImage/QPixmap),
                path to the written GeoTIFF,
                path to the rendered PNG.
        error (str):
            Emitted once on exception with a human-readable error message.

    Args:
        country_code (str): ISO-3 country code (typically lowercase, e.g., "gbr").
        cancer_types (list[str]): One or more cancer type labels to include in the map.
        resolution (float | int | str): Target map resolution in kilometers (passed through).
        population_raster_path (str | pathlib.Path): Path to the (resampled) population raster.
        overwrite_cancer_type_map (bool): Whether to overwrite existing outputs.
        include_RT_utilisation (bool): Forwarded flag (e.g., include standard radiotherapy fraction layer).
        include_optimal_RT_utilisation (bool): Forwarded flag (e.g., include optimal radiotherapy fraction layer).

    Workflow:
        - Calls `generate_cancer_type_map(..., return_image=True, ...)`.
        - On success, unpacks `(image_data, tif_path, png_path)` and emits `finished(...)`.
        - On exception, emits `error(str(e))`.
        - Writes brief progress logs to stdout (prefixed with `[THREAD]`).

    Threading & UI notes:
        - Do not manipulate Qt widgets inside this thread; connect `finished`/`error`
          to main-thread slots for UI updates.
        - Treat as a one-shot worker; create a new instance per map request.

    Example:
        thread = MapGenerationThread(
            country_code="gbr",
            cancer_types=["Breast", "Lung"],
            resolution=1.0,
            population_raster_path="a_population_density/resampled/gbr_1.0km.tif",
            overwrite_cancer_type_map=True,
            include_RT_utilisation=True,
            include_optimal_RT_utilisation=False,
        )
        thread.finished.connect(self.on_map_ready)
        thread.error.connect(self.on_map_error)
        thread.start()
    """
    # --- making compatible with return of v3 since it returns a dict. --- 
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, country_code, cancer_types, resolution, population_raster_path, overwrite_cancer_type_map=False, include_RT_utilisation=False, include_optimal_RT_utilisation=False, include_access_map = False, include_capacity_weighted = False, linac_capacity = None, n_linacs = None,):
        super().__init__()
        self.country_code = country_code
        self.cancer_types = cancer_types
        self.resolution = resolution
        self.population_raster_path = population_raster_path
        self.overwrite_cancer_type_map = overwrite_cancer_type_map
        self.include_RT_utilisation = include_RT_utilisation
        self.include_optimal_RT_utilisation = include_optimal_RT_utilisation
        self.include_capacity_weighted = include_capacity_weighted
        self.linac_capacity = linac_capacity
        self.n_linacs = n_linacs

    def run(self):
        try:
            print(f"[THREAD] Starting map generation for {self.country_code}...")
            result= generate_cancer_type_map(
                country_code=self.country_code,
                cancer_types=self.cancer_types,
                resolution=self.resolution,
                population_raster_path=self.population_raster_path,
                return_image=True, # Do we need this                                        <-- CHECK ME
                overwrite_cancer_type_map=self.overwrite_cancer_type_map,
                include_RT_utilisation=self.include_RT_utilisation,
                include_optimal_RT_utilisation= self.include_optimal_RT_utilisation,
                include_capacity_weighted = self.include_capacity_weighted,
                linac_capacity = self.linac_capacity,
                n_linacs = self.n_linacs
            )
            print(f"[THREAD] Finished map generation.")

            self.finished.emit(result)

        except Exception as e:
            print(f"[THREAD] Error during map generation: {e}")
            self.error.emit(str(e))

class AccessMapThread(QThread):
    """
    Generates an Effective Access map (probability or population-weighted)
    on a worker thread and returns:
      - PNG bytes for quick preview,
      - path to the numeric GeoTIFF (so the GUI can load it for live colour scaling),
      - path to the PNG.

    Params:
      country_code: ISO3 lower-case (e.g., "gbr")
      resolution: map pixel size in km (you can keep using this as default lambda)
      population_raster_path: resampled raster path for the country+resolution
      output_dir: where to write outputs
      overwrite_existing: allow overwrite of existing .png/.tif
      lambda_km: optional; if None, falls back to `resolution`
      cutoff_factor: max distance = cutoff_factor * lambda_km
      value_to_plot: "pop_weighted" (default) or "prob"
      mode: "nearest" (fast, default) or "multi" (independence-product)
    """
    finished = pyqtSignal(bytes, str, str)  # (image_bytes, tif_path, png_path)
    error = pyqtSignal(str)

    def __init__(
        self,
        country_code,
        resolution,
        population_raster_path,
        output_dir,
        overwrite_existing=True,
        *,
        lambda_km=None,
        cutoff_factor=5.0,
        value_to_plot="pop_weighted",
        mode="nearest",
    ):
        super().__init__()
        self.country_code = country_code
        self.resolution = float(resolution)
        self.population_raster_path = population_raster_path
        self.output_dir = output_dir
        self.overwrite_existing = bool(overwrite_existing)

        # new, user-tunable knobs (safe defaults)
        self.lambda_km = float(lambda_km) if lambda_km is not None else None
        self.cutoff_factor = float(cutoff_factor)
        self.value_to_plot = str(value_to_plot)
        self.mode = str(mode)

    def run(self):
        try:
            type_tag_map = {
                "pop_weighted": "access_prob_popw",
                "prob":         "access_prob_dist",
                "distance_km":  "distance_to_linac",
            }
            type_tag = type_tag_map.get(self.value_to_plot, "access_prob_dist")

            # Only append mode for probability maps
            if self.value_to_plot in ("pop_weighted", "prob"):
                base = f"{self.country_code}_{self.resolution}km_{type_tag}_{self.mode}"
            else:
                base = f"{self.country_code}_{self.resolution}km_{type_tag}"


            linac_xlsx = os.path.join(
                "c_probability_of_access", "linac", f"{self.country_code}_DIRAC.xlsx"
            )

            # λ defaults to resolution (backward compatible)
            lam = self.lambda_km if self.lambda_km is not None else self.resolution
            cutoff_km = self.cutoff_factor * lam

            print(f"[THREAD] Generating access map: base={base}, λ={lam} km, cutoff={cutoff_km} km, mode={self.mode}")

            # New API returns (array, tif_path, png_path, stats)
            _, tif_path, png_path, stats = generate_accessibility_plot(
                population_raster_path=self.population_raster_path,
                linac_excel_path=linac_xlsx,
                country=self.country_code,
                output_dir=self.output_dir,
                output_name=base,          # basename only; function appends .tif/.png
                lambda_km=lam,
                max_distance_km=cutoff_km,
                dpi=300,
                show_plot=False,
                value_to_plot=self.value_to_plot,  # "pop_weighted" or "prob"
                mode=self.mode,                    # "nearest" or "multi"
                write_tif=True,                    # crucial for GUI interactivity
                overwrite=self.overwrite_existing,
            )

            # Read PNG bytes for the quick preview (GUI will prefer TIFF if present)
            image_data = b""
            if png_path and os.path.exists(png_path):
                with open(png_path, "rb") as f:
                    image_data = f.read()

            # Hand both paths back; your slot will load the TIFF for interactive scaling
            self.finished.emit(image_data, tif_path or "", png_path or "")

        except Exception as e:
            print(f"[THREAD] Error generating access map: {e}")
            self.error.emit(str(e))

# ==== Helper class to sort numbers correctly ====

class _NumericSortItem(QTreeWidgetItem):
    """Enables numeric sorting for column 1 (cases) of the cancer selection lis, with unknown (None) always last in both orders."""
    def __lt__(self, other):
        tw = self.treeWidget()
        if not tw:
            return super().__lt__(other)
        
        col = tw.sortColumn()
        if col != 1:
            # default for non-numeric column
            return super().__lt__(other)

        desc = bool(tw.property("cases_desc")) # set this in apply_cancer_sort()

        def key(it):
            raw = it.data(1, Qt.UserRole)  # float or None
            if raw is None:
                # missing last in both modes
                return (1,0.0)
            num = float(raw)
            if desc:
                # flip only here; we will always call sortByColumn(..., AcendingOrder)
                num = -num
            return (0, num)

        return key(self) < key(other)


# ==== Main GUI Window Class ====

class GeoSpacRadAccess(QMainWindow):
    def __init__(self):
        super().__init__()
        # Adding an instance variable 
        self.recent_countries = []
        self.max_recent = 5  # or however many you want to show
        self.map_thread = None # ensures self.map_thread is always defined
        self.hide_missing_cases = True
        self._suppress_item_changed = False
        self._current_title = ""
        self._is_probability_map = False
        self._is_distance_map = False
        self.setup_ui()
          

    
    # ---- Initial UI setup ----

    def setup_ui(self):
        """
        UPDATE ME
        This builds and wires up the main window UI.

        This method constructs the entire interface and connects signals to slots. It creates a
        two-panel layout separated by a horizontal splitter:

        Left panel (controls and user interaction):
        1) Download Raw Data
            - Country selector (QComboBox) populated via `update_country_dropdown()`
            - Download button (QPushButton) via `initiate_download()`
            - Progress bar for download (QProgressBar), hidden until used

        2) Resample Data
            - Resolution selector in km (QComboBox: 0.5–50km)
            - Resample button (disabled by default) via `initiate_resample()`
            and enabled/disabled by `check_resample_availability()`

        3) Generate Cancer Type Map
            - Multi-select cancer type list (QListWidget) populated from `load_cancer_types()`
            with user-checkable items
            - “Select All Cancer Types” checkbox via `toggle_select_all_cancers()`
            - Map type selector (QComboBox: Incidence / Treated / Optimally Treated / Population)
            - Generate Map button (disabled by default) via `initiate_cancer_type_map_generate()`
            and enabled/disabled by `check_cancer_map_availability()`

        Right panel (output):
        - Image area (QLabel) where generated maps are displayed
        - Read-only status log (QTextEdit)

        Signals connected:                     VIA
        - `download_btn.clicked`                ->  `initiate_download`
        - `resample_btn.clicked`                ->  `initiate_resample`
        - `generate_map_btn.clicked`            ->  `initiate_cancer_type_map_generate`
        - `country_combo.currentTextChanged`    ->  `update_country_dropdown(country)`
        - `country_combo.currentTextChanged`    ->  `check_resample_availability`
        - `country_combo.currentTextChanged`    ->  `check_cancer_map_availability`
        - `resolution_combo.currentTextChanged` ->  `check_cancer_map_availability`
        - `map_type_combo.currentTextChanged`   ->  `check_cancer_map_availability`
        - `select_all_checkbox.stateChanged`    ->  `toggle_select_all_cancers`

        Side effects:
        - Sets window title and fixed size.
        - Creates and assigns numerous instance attributes (e.g., `country_combo`, `progress`,
            `resample_btn`, `cancer_list`, `image_label`, `status_text`, etc.).
        - Installs a central widget with a splitter containing left/right panels.
        - Applies initial UI state: hides progress bar; disables resample/map buttons until
            availability checks pass; sets reasonable min/max sizes.

        Requirements/assumptions:
        - This class is a QMainWindow (uses `setCentralWidget`).
        - Needs helper methods `update_country_dropdown`, `check_resample_availability`,
            `check_cancer_map_availability`, `load_cancer_types`, `toggle_select_all_cancers`,
            `initiate_download`, `initiate_resample`, and `initiate_cancer_type_map_generate`
        - Long-running tasks triggered by the buttons should be (and are so far) executed off the GUI thread
            (e.g., via workers/separate threads) to keep the interface responsive.

        Returns:
        None
        """

        self.setWindowTitle("Geospatial Modelling of Radiotherapy Access")
        self.setFixedSize(1300, 1000)

        # Main splitter (used at the end)
        splitter = QSplitter(Qt.Horizontal)

        # ==== UI left panel  ====

        left_panel = QWidget()
        left_layout = QVBoxLayout()


        # ---- Download Group ----
        download_group = QGroupBox("Download Raw Data")
        download_layout = QVBoxLayout()
        
        self.country_label = QLabel("Select a country:")
        self.country_combo = QComboBox()
        self.update_country_dropdown() #Fill the dropdown with recent + all, see helper below
        
        self.download_btn = QPushButton("Download")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        download_layout.addWidget(self.country_label)
        download_layout.addWidget(self.country_combo)
        download_layout.addWidget(self.download_btn)
        download_layout.addWidget(self.progress)
        download_group.setLayout(download_layout)


        # ---- Resample Group ----
        resample_group = QGroupBox("Resample Data")
        resample_layout = QVBoxLayout()
        
        self.resolution_label = QLabel("Select resolution (km):")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["0.5", "1", "2", "5", "10", "50"])
        
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.setEnabled(False) # initially disabled
        self.check_resample_availability() # check if resampling is available, if so enable the button
        
        resample_layout.addWidget(self.resolution_label)
        resample_layout.addWidget(self.resolution_combo)
        resample_layout.addWidget(self.resample_btn)
        resample_group.setLayout(resample_layout)


        # ---- Cancer Type & Map Generation Group ----
        map_group = QGroupBox("Generate Cancer Type Map")
        map_layout = QVBoxLayout()

        # Cancer type list
        self.cancer_label = QLabel("Select a cancer type:")

        # New 2-column tree with checkboxes
        self.cancer_table = QTreeWidget()
        self.cancer_table.setProperty("cases_desc", False)
        self.cancer_table.setColumnCount(2)
        self.cancer_table.setHeaderLabels(["Cancer type", "Cases"])
        self.cancer_table.setRootIsDecorated(False)
        self.cancer_table.setAlternatingRowColors(True)
        self.cancer_table.setSortingEnabled(True)
        self.cancer_table.sortByColumn(0, Qt.AscendingOrder)

        # Nice sizing
        hdr = self.cancer_table.header()
        hdr.setSortIndicatorShown(True)
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)

        # Populate for current country
        self.refresh_cancer_table()

    
        # "Select All Cancers" Checkbox
        self.select_all_checkbox = QCheckBox("Select All Cancer Types") 
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all_cancers)

        # Why is this commented (Archie)
        #self.include_fraction_checkbox = QCheckBox("Include radiotherapy fraction") 
        #self.include_fraction_checkbox.setChecked(False) 

        # Map type box
        self.map_type_label = QLabel("Select map to generate:")
        self.map_type_combo = QComboBox() 
        self.map_type_combo.addItems([
            "Cancer Incidence", "Treated by Radiotherapy", 
            "Optimally Treated by Radiotherapy", "Population Density", 
            ACCESS_PROB_DIST,
            ACCESS_PROB_POPW,
            ACCESS_DIST_NEAREST,

        ])
        self.generate_map_btn = QPushButton("Generate Map")
        self.generate_map_btn.setEnabled(False)  
        self.check_cancer_map_availability() # check if cancer map generation is available, if so enable the button

        # --- adding new widgets to generate capacity-weighted maps ---
        self.capacity_weighted_checkbox = QCheckBox("Enable capacity-weighted maps") 
        self.capacity_weighted_checkbox.setChecked(False)
        self.capacity_weighted_checkbox.setVisible(False) 

        self.linac_capacity_label = QLabel("Linac capacity (patients/year):") 
        self.linac_capacity_label.setVisible(False) 

        # Horizontal layout to hold slider + spinbox 
        self.linac_capacity_layout = QHBoxLayout() 

        # Slider
        self.linac_capacity_slider = QSlider(Qt.Horizontal)
        self.linac_capacity_slider.setRange(100, 700)
        self.linac_capacity_slider.setTickInterval(50)
        self.linac_capacity_slider.setTickPosition(QSlider.TicksBelow)
        self.linac_capacity_slider.setVisible(False)

        # Spinbox for precise adjustment
        self.linac_capacity_spin = QSpinBox()
        self.linac_capacity_spin.setRange(100, 700)
        self.linac_capacity_spin.setSingleStep(10)
        self.linac_capacity_spin.setVisible(False)

        # Keep them synced
        self.linac_capacity_slider.valueChanged.connect(self.linac_capacity_spin.setValue)
        self.linac_capacity_spin.valueChanged.connect(self.linac_capacity_slider.setValue)

        # Add to layout
        self.linac_capacity_layout.addWidget(self.linac_capacity_slider)
        self.linac_capacity_layout.addWidget(self.linac_capacity_spin)

        map_layout.addWidget(self.cancer_label)

        map_layout.addWidget(self.cancer_table)

        map_layout.addWidget(self.select_all_checkbox) 
        map_layout.addWidget(self.map_type_label)
        map_layout.addWidget(self.map_type_combo)
        map_layout.addWidget(self.capacity_weighted_checkbox)
        map_layout.addWidget(self.linac_capacity_label) 
        map_layout.addLayout(self.linac_capacity_layout)
        map_layout.addWidget(self.generate_map_btn)


        map_group.setLayout(map_layout)


        # ---- Arranging ----
        # Add groups to left layout
        left_layout.addWidget(download_group)
        left_layout.addWidget(resample_group)
        left_layout.addWidget(map_group)

        # Setting size of left panel
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(500)
        left_panel.setMinimumWidth(350)


        # ==== Right panel for image display ====

        right_panel  = QWidget()
        self._extent = None
        right_layout = QVBoxLayout()
        
        """
        self.image_label = QLabel("Generated map will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        """

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        
        """
        right_layout.addWidget(QLabel("Generated Map:"))
        right_layout.addWidget(self.image_label)
        """

        right_layout.addWidget(QLabel("Generated Map:"))

        # ---- Log10 upper-limit control (10^k) ----
        row = QHBoxLayout()
        row.addWidget(QLabel("Log scale"))
        self.exp_spin = QSpinBox()
        self.exp_spin.setRange(1, 10)   # k in 10^k
        self.exp_spin.setValue(6)         # default; will auto-set on first image
        row.addWidget(QLabel("Upper 10^k (k):"))
        row.addWidget(self.exp_spin)
        self.auto_btn = QPushButton("Auto from data")
        row.addWidget(self.auto_btn)

        # bucket < 1 option
        self.under1_checkbox = QCheckBox("Colour all values < 1 the same")
        self.under1_checkbox.setChecked(False)
        row.addWidget(self.under1_checkbox)

        # ---- λ (km) control for access maps ----
        self.lambda_label = QLabel("    λ (km):")
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setDecimals(1)
        self.lambda_spin.setRange(0.1, 5000.0)
        self.lambda_spin.setSingleStep(0.5)
        self.lambda_spin.setSuffix(" km")
        # default λ = current resolution; also starts disabled until an access map is chosen
        self.lambda_spin.setValue(float(self.resolution_combo.currentText()))
        self.lambda_label.setVisible(False)
        self.lambda_spin.setVisible(False)

        row.addWidget(self.lambda_label)
        row.addWidget(self.lambda_spin)

        # ---- NEW: Access mode selector ----
        self.mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Nearest LINAC (fast)",
            "Multiple LINACs (independence)"
        ])
        self.mode_combo.setCurrentIndex(0)
        self.mode_label.setVisible(False)
        self.mode_combo.setVisible(False)
        row.addWidget(self.mode_label)
        row.addWidget(self.mode_combo)

        right_layout.addLayout(row)

        # --- Matplotlib canvas + toolbar (interactive) ---
        self.fig = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()  # start without axes
        self.toolbar = NavigationToolbar(self.canvas, self)

        # A gentle frame like your QLabel had
        self.canvas.setMinimumSize(650, 650)
        self.canvas.setStyleSheet("border: 1px solid gray; background-color: #f8f8f8;")

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        # --- end Matplotlib block ---

        # ---- state + signals ----
        self._data = None     # 2D masked array (positive numeric values only)
        self._im = None       # AxesImage
        self._cbar = None
        self.exp_spin.valueChanged.connect(self._apply_exp_upper)
        self.auto_btn.clicked.connect(self._auto_from_data_set_upper)
        self.under1_checkbox.stateChanged.connect(lambda _:
                                                  self._redraw_with_current_settings())
        
        # update λ visibility & behavior when map-type or resolution changes
        self.map_type_combo.currentTextChanged.connect(self._on_map_type_changed)
        #self.resolution_combo.currentTextChanged.connect(self._on_resolution_changed)
        
        self.map_type_combo.currentTextChanged.connect(self._on_map_type_changed_capacity_controls)

        # Debounce timer for access-map refreshes
        self._access_regen_timer = QTimer(self)
        self._access_regen_timer.setSingleShot(True)
        self._access_regen_timer.setInterval(300)  # 300 ms debounce
        self._access_regen_timer.timeout.connect(self._refresh_access_map)

        self.lambda_spin.valueChanged.connect(self._schedule_access_map_refresh)
        self.mode_combo.currentIndexChanged.connect(self._schedule_access_map_refresh)

        

        right_layout.addWidget(QLabel("Status:"))
        right_layout.addWidget(self.status_text)

        right_panel.setLayout(right_layout)
        

        # ==== Constructing ====
        
        # Splitter add panels
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)


        # === Register Signals ====

        # Buttons clicked signals
        self.download_btn.clicked.connect(self.initiate_download)
        self.resample_btn.clicked.connect(self.initiate_resample)
        self.generate_map_btn.clicked.connect(self.initiate_cancer_type_map_generate)

        # Country selected signal
        self.country_combo.currentTextChanged.connect(lambda country: self.update_country_dropdown(country))
        
        # When country/resolultion/map-type changes, update and check for either raw data for resampling/resampled file for map
        self.country_combo.currentTextChanged.connect(self.check_resample_availability) 
        self.country_combo.currentTextChanged.connect(lambda _: self.refresh_cancer_table())
        self.country_combo.currentTextChanged.connect(self.check_cancer_map_availability) 

        self.resolution_combo.currentTextChanged.connect(self.check_cancer_map_availability) 

        self.cancer_table.itemChanged.connect(self._on_cancer_item_changed)

        self.map_type_combo.currentTextChanged.connect(self.check_cancer_map_availability)

    # ---- New unsorted Helpers for the cancer table ----

    def _get_current_iso3(self) -> str | None:
        try:
            return countries.lookup(self.country_combo.currentText()).alpha_3.upper()
        except Exception:
            return None
        
    def _load_cancer_case_counts(self, iso3: str, 
                                 metric: str = DEFAULT_METRIC_NAME,
                                 xarray_path: str | Path = None) -> dict[str, float]:
        """
        Returns {Cancer -> cases} for the given ISO3 and metric.
        Falls back to {} on error (UI still shows cancers, cases blank)
        """
        try:
            p = Path(xarray_path) if xarray_path else DEFAULT_XARRAY_PATH
            da = xr.load_dataarray(p)
            # Select down to Cancer dimension
            sel = da.sel(Metric=metric, ISO3=iso3)
            # Convert to a Series keyed by Cancer names
            series = sel.to_series()
            # Make plain dict with floats
            return {str(idx): float(val) for idx, val in series.items() if pd.notna(val)}
        except Exception as e:
            self.update_status(f"Warning: couldn't load case counts: {e}")
            return {}
        
    def refresh_cancer_table(self):
        """
        Rebuild the 2-column cancer table for the selected country, 
        preserving existing checkmarks when possible
        """
        # Preserve existing checks
        previously_checked = set(self.get_selected_cancer_types())

        cancers = self.load_cancer_types()
        iso3 = self._get_current_iso3()
        counts = self._load_cancer_case_counts(iso3) if iso3 else {}

        self.cancer_table.clear()
        self.cancer_table.setSortingEnabled(False)

        self._suppress_item_changed = True
        try:
            for ctype in cancers:
                val = counts.get(ctype)
                has_num = (val is not None) and np.isfinite(val)

                if self.hide_missing_cases and not has_num:
                    continue # Skip missing cases entirely

                display = f"{val:,.0f}" if has_num else "-"
                item = _NumericSortItem([ctype, display])

                # Store raw number for numeric sort
                item.setData(1, Qt.UserRole, float(val) if has_num else None)

                # make checkable 
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Checked if ctype in previously_checked else Qt.Unchecked)
                # align cases right
                item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                self.cancer_table.addTopLevelItem(item)
        finally:
            self._suppress_item_changed = False

        # apply current sort choice
        self.cancer_table.setSortingEnabled(True)

    def _on_cancer_item_changed(self, item: QTreeWidgetItem, column: int):
        # Only react to user check/uncheck on the checkbox column
        if self._suppress_item_changed or column != 0:
            return

        name = item.text(0)
        checked = (item.checkState(0) == Qt.Checked)

        # If a sentinel gets checked, uncheck every other item
        if checked and _is_sentinel_cancer(name):
            self._suppress_item_changed = True
            try:
                for i in range(self.cancer_table.topLevelItemCount()):
                    it = self.cancer_table.topLevelItem(i)
                    if it is not item:
                        it.setCheckState(0, Qt.Unchecked)
            finally:
                self._suppress_item_changed = False
            return

        # If a non-sentinel gets checked, uncheck all sentinel items
        if checked and not _is_sentinel_cancer(name):
            self._suppress_item_changed = True
            try:
                for i in range(self.cancer_table.topLevelItemCount()):
                    it = self.cancer_table.topLevelItem(i)
                    if _is_sentinel_cancer(it.text(0)):
                        it.setCheckState(0, Qt.Unchecked)
            finally:
                self._suppress_item_changed = False

    # ---- Colourbar Graph Update ----

    def _read_raster(self, tif_path: str, mask_zero_for_log: bool = True) -> np.ma.MaskedArray:
        """Read first band as float64, mask non-finite and <= 0 (LogNorm-safe)."""
        with rasterio.open(tif_path) as ds:
            data = ds.read(1).astype(np.float64)
            b = ds.bounds
            self._extent = (b.left, b.right, b.bottom, b.top)

        data = np.where(np.isfinite(data), data, np.nan)
        m = np.ma.masked_invalid(data)
        if mask_zero_for_log:
            # For log-scaled maps: exclude 0 to keep LogNorm happy
            m = np.ma.masked_less_equal(m, 0.0)
        else:
            # For probability maps (linear): allow 0, only mask negatives
            m = np.ma.masked_less(m, 0.0)
        return m

    def display_raster_from_tif(self, tif_path: str):
        try:
            # mask zeros ONLY for log-style maps (neither probability nor distance)
            mask_zeros = (not getattr(self, "_is_probability_map", False)) and (not getattr(self, "_is_distance_map", False))
            self._data = self._read_raster(tif_path, mask_zero_for_log=mask_zeros)
            
            if self._data is None or self._data.size == 0 or self._data.mask.all():
                self.update_status("No positive data to display.")
                return

            self._plot_log(self._data)  # (name kept, behaviour branches inside)

            # Toggle controls: log-scale widgets off for probability maps
            is_prob = getattr(self, "_is_probability_map", False)
            is_dist = getattr(self, "_is_distance_map", False)
            disable_logs = is_prob or is_dist

            self.exp_spin.setEnabled(not disable_logs)
            self.auto_btn.setEnabled(not disable_logs)
            if hasattr(self, "under1_checkbox"):
                self.under1_checkbox.setEnabled(not disable_logs)
                if disable_logs and self.under1_checkbox.isChecked():
                    self.under1_checkbox.setChecked(False)

            self.update_status(f"Rendered from: {tif_path}")
        except Exception as e:
            self.update_status(f"Error reading raster: {e}")

    def _plot_log(self, data: np.ma.MaskedArray, vmin=None, vmax=None):
        vals = data.compressed()
        if vals.size == 0:
            self.update_status("No data available for plotting.")
            return

        is_prob = getattr(self, "_is_probability_map", False)
        is_dist = getattr(self, "_is_distance_map", False)

        if is_prob:
            # ---- Linear scale for probability 0..1 ----
            # Default limits
            if vmin is None:
                vmin = 0.0
            if vmax is None:
                # cap at 1.0 but allow lower if all data < 1
                vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
                vmax = min(max(vmax, 0.0), 1.0)
                if vmax <= vmin:
                    vmax = vmin + 1e-6

            # Safety (prevents the “Invalid vmin or vmax” crash)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = 0.0, 1.0

            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.get_cmap("viridis").copy()
            #cmap.set_bad("#f0f0f0")   # <— sea/nodata light grey
            extend_mode = "neither"

        elif is_dist:
            # ---- Linear scale for distance (km), include zeros ----
            if vmin is None: vmin = 0.0
            if vmax is None:
                # robust upper for pretty map; adjust if you prefer max()
                vmax = float(np.nanpercentile(vals, 98)) if vals.size else 1.0
                if vmax <= vmin: vmax = vmin + 1e-6
            norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.get_cmap("viridis").copy()
            extend_mode = "neither"

        else:
            # ---- Log scale for non-probability maps ----
            if vmin is None:
                vmin = float(np.percentile(vals, 2))
                vmin = max(vmin, 1e-12)
            if vmax is None:
                vmax = float(np.percentile(vals, 98))
                if vmax <= vmin:
                    vmax = vmin * 10.0

            bucket_under_one = hasattr(self, "under1_checkbox") and self.under1_checkbox.isChecked()
            if bucket_under_one:
                vmin = max(vmin, 1.0)
            norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
            cmap = cm.get_cmap("viridis").copy()
            extend_mode = "min" if bucket_under_one else "neither"
            if bucket_under_one:
                cmap.set_under("#d9d9d9")

        # --- draw/update ---
        if self._im is None:
            self.ax.clear()
            self.ax.set_axis_on()
            self._im = self.ax.imshow(
                data, origin="upper", cmap=cmap, norm=norm,
                interpolation="nearest", extent=getattr(self, "_extent", None)
            )
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_xlabel("Longitude")
            self.ax.set_ylabel("Latitude")
            if self._cbar:
                self._cbar.remove()
            self._cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04, extend=extend_mode)
        else:
            self._im.set_data(data)
            self._im.set_norm(norm)
            self._im.set_cmap(cmap)
            if self._extent is not None:
                self._im.set_extent(self._extent)
            self.ax.set_aspect("equal", adjustable="box")
            need_recreate = (self._cbar is None) or (getattr(self._cbar, "extend", None) != extend_mode)
            if need_recreate:
                if self._cbar:
                    self._cbar.remove()
                self._cbar = self.fig.colorbar(self._im, ax=self.ax, fraction=0.046, pad=0.04, extend=extend_mode)
            else:
                self._cbar.update_normal(self._im)

        # ---- tick formatting ----
        if is_prob:
            # linear ticks 0..1
            ticks = np.linspace(0.0, 1.0, 6)
            try:
                self._cbar.set_ticks(ticks)
                self._cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
            except Exception:
                pass
            self._cbar.set_label("Probability", rotation=90)
        
        elif is_dist:
            # linear ticks in km
            try:
                lo, hi = norm.vmin, norm.vmax
                ticks = np.linspace(lo, hi, 6)
                self._cbar.set_ticks(ticks)
                self._cbar.set_ticklabels([f"{t:.0f}" for t in ticks])
            except Exception:
                pass
            self._cbar.set_label("Distance (km)", rotation=90)
        
        else:
            self._cbar.locator = LogLocator(base=10)
            self._cbar.formatter = ScalarFormatter()
            self._cbar.update_ticks()

        # ---- keep title + update exp_spin state for log-only ----
        if is_prob or is_dist:
            # Sync the spin to something sensible but keep it disabled by caller
            self.exp_spin.blockSignals(True)
            self.exp_spin.setValue(0)
            self.exp_spin.blockSignals(False)
        else:
            import math
            k = int(math.ceil(math.log10(max(vmax, 1e-12))))
            self.exp_spin.blockSignals(True)
            self.exp_spin.setValue(k)
            self.exp_spin.blockSignals(False)

        self.ax.set_title(getattr(self, "_current_title", ""), fontsize=11, pad=6)
        self.canvas.draw_idle()

    def _redraw_with_current_settings(self):
        if self._data is None or self._im is None:
            return
        current_norm = getattr(self._im, "norm", None)
        vmin = getattr(current_norm, "vmin", None)
        vmax = getattr(current_norm, "vmax", None)
        self._plot_log(self._data, vmin=vmin, vmax=vmax)

    def _apply_exp_upper(self):
        """When k changes: set vmax = 10^k, keep vmin sane, redraw."""
        if getattr(self, "_is_probability_map", False):
            return
        if self._data is None or self._im is None:
            return
        import math
        k = self.exp_spin.value()
        vmax = 10.0 ** k

        # keep existing vmin if possible
        current_norm = self._im.norm
        vmin = getattr(current_norm, 'vmin', None)

        # Respect the bucket-under-1 setting
        bucket_under_one = hasattr(self, "under1_checkbox") and self.under1_checkbox.isChecked()
        floor = 1.0 if bucket_under_one else 1e-12

        if vmin is None or vmin >= vmax:
            vmin = max(vmax / 1000.0, 1e-12)  # default: 3 decades below vmax
        else:
            vmin = max(vmin, floor)
        self._plot_log(self._data, vmin=vmin, vmax=vmax)

    def _auto_from_data_set_upper(self):
        """Pick k from the data (≈ceil(log10(98th percentile)))."""
        if getattr(self, "_is_probability_map", False):
            return
        
        if self._data is None:
            return
        vals = self._data.compressed()
        if vals.size == 0:
            return
        import math
        vmax = float(np.percentile(vals, 98))
        vmax = max(vmax, 1e-12)
        k = int(math.ceil(math.log10(vmax)))
        # set spin (which will call _apply_exp_upper via signal)
        self.exp_spin.setValue(k)

    def _rt_title_suffix(self, iso3: str, cancers: list[str],
                     include_actual: bool, include_optimal: bool) -> str:
        """
        Build a short note for the title, mirroring the generator's logic:
        - If 'actual' selected but actual CSV missing → fallback note.
        - If 'actual' selected but some cancers missing in actual CSV → list them.
        - If 'optimal' selected or incidence → no suffix.
        """
        try:
            # Only relevant when user asked for "Treated by Radiotherapy"
            if not include_actual or include_optimal:
                return ""

            # Load optimal map (for parity; we only need it to mirror conditions)
            opt_map = _rt_load_csv(DEFAULT_OPTIMAL_RT_CSV)

            # Load per-country actual (may be absent)
            actual_path = os.path.join(DEFAULT_ACTUAL_RT_DIR, f"{iso3.upper()}.csv")
            act_map = _rt_load_csv(actual_path) if os.path.exists(actual_path) else None

            used_optimal = False
            missing_actual_for: list[str] = []

            if act_map is None:
                # No actual file at all → full fallback
                used_optimal = True
                missing_actual_for = cancers[:]
            else:
                # Actual exists; check per-cancer coverage
                for ct in cancers:
                    if _rt_norm_key(ct) not in act_map:
                        used_optimal = True
                        missing_actual_for.append(ct)

            if used_optimal:
                # Match the generator’s wording
                if missing_actual_for:
                    return f"\n(fallback to optimal for some cancers) [no actual for: " + ", ".join(missing_actual_for) + "]"
                else:
                    return " (fallback to optimal for some cancers)"
            return ""
        except Exception as e:
            # Non-fatal: if CSVs misbehave, just show no suffix and log a hint
            self.update_status(f"Note: could not compute RT title suffix ({e}).")
            return ""

    def _is_access_map_text(self, text: str) -> bool:
        # supports both the new labels and the legacy one
        return text in (
            ACCESS_PROB_DIST,
            ACCESS_PROB_POPW,
        )
    
    def _is_distance_map_text(self, text: str) -> bool:
        return text == ACCESS_DIST_NEAREST

    def _current_access_mode(self) -> str:
        """
        Return "nearest" or "multi" based on the access-mode combo.
        """
        if not hasattr(self, "mode_combo"):
            return "nearest"
        t = self.mode_combo.currentText().lower()
        return "multi" if "multi" in t else "nearest"
    
    # ---- Lambda button functions ----

    def _on_map_type_changed(self, _text: str):
        """Show/hide λ control depending on whether an access map is selected."""
        is_access = self._is_access_map_text(self.map_type_combo.currentText())
        is_dist   = self._is_distance_map_text(self.map_type_combo.currentText())

        self.lambda_label.setVisible(is_access)
        self.lambda_spin.setVisible(is_access)
        self.mode_label.setVisible(is_access)
        self.mode_combo.setVisible(is_access)
        if is_access:
            # default λ to 20 when switching into access mode
            try:
                self.lambda_spin.blockSignals(True)
                self.lambda_spin.setValue(float(20))
            finally:
                self.lambda_spin.blockSignals(False)

    """def _on_resolution_changed(self, _text: str):
        """"""Keep λ in sync with resolution by default when in access mode.""""""
        if self._is_access_map_text(self.map_type_combo.currentText()):
            try:
                self.lambda_spin.blockSignals(True)
                self.lambda_spin.setValue(float(self.resolution_combo.currentText()))
            finally:
                self.lambda_spin.blockSignals(False)
"""
    def _schedule_access_map_refresh(self, *_):
        """Start/restart the debounce timer after λ or Mode changes."""
        if not self._is_access_map_text(self.map_type_combo.currentText()):
            return
        # Restart the single-shot timer (debounce)
        self._access_regen_timer.stop()
        self._access_regen_timer.start()

    def _refresh_access_map(self):
        """
        Debounced refresh: recompute access map with current λ and Mode.
        Uses silent overwrite to avoid prompts during interactive tweaking.
        """
        if not self._is_access_map_text(self.map_type_combo.currentText()):
            return

        country = self.country_combo.currentText()
        if not country:
            return

        try:
            iso3_lower = countries.lookup(country).alpha_3.lower()
            resolution = float(self.resolution_combo.currentText())
            population_raster_path = f"a_population_density/resampled/{iso3_lower}_{resolution}km.tif"
            if not os.path.exists(population_raster_path):
                self.update_status("No resampled raster found; cannot refresh access map.")
                return
        except Exception as e:
            self.update_status(f"λ/Mode change ignored (bad state): {e}")
            return

        # Access subtype and value_to_plot
        mt = self.map_type_combo.currentText()
        if self._is_access_map_text.__defaults__ is None:
            # Nothing special, but make linters happy
            pass
        vtp = ("prob" if ("distance" in mt.lower()) else "pop_weighted")

        lam = float(self.lambda_spin.value())
        mode = self._current_access_mode()

        output_dir = "c_probability_of_access/access_probability_maps"
        os.makedirs(output_dir, exist_ok=True)

        # Cancel any running worker
        if self.map_thread is not None and self.map_thread.isRunning():
            self.map_thread.quit()
            self.map_thread.wait()

        # Title (reflect λ)
        if vtp == "prob":
            self._current_title = f"{country} — Probability of treatment access (distance) (λ={lam:g} km)"
        else:
            self._current_title = f"{country} — Probability of treatment access (population-weighted) (λ={lam:g} km)"

        # Silent refresh (overwrite) for interactive tweaks
        self.map_thread = AccessMapThread(
            country_code=iso3_lower,
            resolution=resolution,
            population_raster_path=population_raster_path,
            output_dir=output_dir,
            overwrite_existing=True,
            lambda_km=lam,
            cutoff_factor=5.0,
            value_to_plot=vtp,
            mode=mode,                      # <-- wire the chosen mode
        )
        self.map_thread.finished.connect(self.cancer_type_map_completed)
        self.map_thread.error.connect(self.on_map_generation_error)
        self.map_thread.start()


    # ---- HELPER METHODS for setup_ui(): re-ordered by appearance ----

    def update_country_dropdown(self, selected_country=None):
        """
        Refreshes the country QComboBox with “Recent Selections” and the full country list.

        This helper repopulates `self.country_combo` by:
            - Preserving the current selection (unless an explicit `selected_country` is supplied).  
            - Maintaining a MRU-style list of recently chosen countries in `self.recent_countries`
            (bounded by `self.max_recent`).  
            - Rendering two non-selectable headers, “Recent Selections” (if any) and “All Countries”,
            separated by a visual separator.  
            - Listing all countries (alphabetical) from the module-level `countries` collection
            (from `pycountry.countries`), excluding those already shown under “Recent Selections”.  
            - Reselecting the previously active country if it still exists in the menu.

        Signals on the combo box are temporarily blocked to avoid spurious `currentTextChanged`
        emissions while the model is being rebuilt.

        Args:
            selected_country (str | None): Country name to keep selected after the refresh.
                If None, the method uses the combo’s current text.

        Side Effects:
            - Mutates `self.recent_countries` (adds the current selection to the front, trims to
            `self.max_recent` defined above).
            - Clears and repopulates `self.country_combo`, sets header rows disabled, inserts a separator.
            - May change the current index of `self.country_combo`.

        Requirements/Assumptions:
            - `self.country_combo` is a QComboBox.
            - `self.recent_countries` is a list-like container of country names (strings).
            - `self.max_recent` is an integer ≥ 0.
            - A module-level iterable `countries` is available, yielding objects with a `.name` attribute
            (i.e., `pycountry.countries`).

        Returns:
            None
        """
        # Preserve current selection
        if selected_country is None:
            selected_country = self.country_combo.currentText()

        all_countries = sorted([country.name for country in countries]) # countries is imported from pycountry

        # Move selected country to recent list
        if selected_country and selected_country not in self.recent_countries:
            self.recent_countries.insert(0, selected_country)
            if len(self.recent_countries) > self.max_recent:
                self.recent_countries.pop()

        self.country_combo.blockSignals(True)  # prevent triggering signals while updating
        self.country_combo.clear()

        # Add recent countries
        if self.recent_countries:
            self.country_combo.addItem("Recent Selections")
            self.country_combo.model().item(self.country_combo.count() - 1).setEnabled(False)
            for country in self.recent_countries:
                self.country_combo.addItem(country)
            self.country_combo.insertSeparator(self.country_combo.count())

        # Add all countries
        self.country_combo.addItem("All Countries")
        self.country_combo.model().item(self.country_combo.count() - 1).setEnabled(False)
        for country in all_countries:
            if country not in self.recent_countries:
                self.country_combo.addItem(country)

        # Reselect the previously selected country
        index = self.country_combo.findText(selected_country)
        if index != -1:
            self.country_combo.setCurrentIndex(index)

        self.country_combo.blockSignals(False)

    def check_resample_availability(self):
        """
        Enables/disables the “Resample” button based on the presence of a raw WorldPop raster
        for the currently selected country.

        Behavior:
        - Reads the current country name from `self.country_combo`.
        - Uses `pycountry.countries.lookup(country)` to resolve the country object and its
            3-letter ISO code (`alpha_3`), lower-cases it, and constructs the expected input
            filepath: `a_population_density/raw_from_worldpop/{alpha3}_raw.tif`.
        - Sets `self.resample_btn` enabled if that file exists; otherwise disables it.
        - On any lookup or filesystem error, safely disables the button.

        Notes:
        - If no country is selected (empty string), the method returns without changing state.
        - Assumes the project’s raw WorldPop rasters follow the naming convention
            `{iso3_lower}_raw.tif` and live under `a_population_density/raw_from_worldpop/`.
        - Should probably change the file selection to use Path so that it works on all 
        operating systems? I think it'll probs work but maybe just to be safe.

        Args:
            None

        Side Effects:
            - Mutates the enabled state of `self.resample_btn`.

        Returns:
            None
        """
        country = self.country_combo.currentText()
        if not country:
            return
            
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            input_file = os.path.join("a_population_density/raw_from_worldpop", f"{country_code}_raw.tif")
            self.resample_btn.setEnabled(os.path.exists(input_file))
        except:
            self.resample_btn.setEnabled(False)

    def load_cancer_types(self, xarray_path: str | Path | None = None) -> list[str]:
        """
        Load and return a sorted list of unique cancer types from the on-disk xarray tensor.

        Reads the DataArray at `xarray_path` (defaults to b_cancer_incidence/globocan_xarray.nc),
        validates required coords/dims, extracts `Cancer` coordinate values, de-duplicates,
        sorts, and returns them. On error, shows a dialog and returns [].

        """
        try:
            p = Path(xarray_path) if xarray_path else DEFAULT_XARRAY_PATH
            #if not p.exists():
            #    raise FileNotFoundError(f"Tensor not found: {p}")

            da = xr.load_dataarray(p)
            required = {"Cancer", "Metric", "ISO3"}
            if not required.issubset(set(da.dims)) and not required.issubset(set(da.coords)):
                raise ValueError(f"Tensor missing required dims/coords {required}; found dims={list(da.dims)} coords={list(da.coords)}")

            cancers = [str(c) for c in da.coords["Cancer"].values]
            cancers = sorted({c for c in cancers if c and str(c).strip().lower() not in ("nan", "none")})
            return cancers
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cancer types from xarray: {e}")
            return []

    def check_cancer_map_availability(self): 
        """
        Enables/disables the “Generate Map” button depending on whether a resampled
        population raster exists for the currently selected country and resolution.

        Behavior:
        - Reads selections from:
            • `self.country_combo` (country name)
            • `self.resolution_combo` (resolution in km; parsed to float)
            • `self.map_type_combo` (map type; currently not used in the file check)       <-- CHECK ME
        - Resolves the country’s ISO3 code via `pycountry.countries.lookup(country)`,
            lower-cases it, and constructs the expected filepath:
            `a_population_density/resampled/{iso3_lower}_{resolution}km.tif`.
        - If the file exists, enables `self.generate_map_btn`; otherwise disables it.
        - If no country/resolution is set or an error occurs (e.g., lookup fails),
            the button is disabled.

        Notes:
        - The filename uses the string form of the parsed float for `resolution`
            (e.g., "1" → "1.0km"); check saved rasters follow the same naming
            convention to avoid mismatches.
        - `map_type` is read for potential future logic but does not affect the
            availability check at present.
        - Unless I'm mistaken, doens't check whether cancer data actually exists, will add maybe?   <--- CHECK ME

        Args:
            None

        Side Effects:
            - Mutates the enabled state of `self.generate_map_btn`.

        Requirements/Assumptions:
            - `self.country_combo`, `self.resolution_combo`, `self.map_type_combo`
            are valid Qt widgets with current selections (they are).
            - `self.generate_map_btn` is a QPushButton (it is).
            - Module-level `countries` (e.g., `pycountry.countries`) is available (it is).
            - The resampled rasters are stored under
            `a_population_density/resampled/` with names
            `{iso3_lower}_{resolution}km.tif` (they are).

        Returns:
            None
        """
        country = self.country_combo.currentText()
        resolution = float(self.resolution_combo.currentText())
        map_type = self.map_type_combo.currentText()
        if not country or not resolution:
            self.generate_map_btn.setEnabled(False)
            return
            
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            input_file = os.path.join("a_population_density/resampled", f"{country_code}_{resolution}km.tif")
            exists = os.path.exists(input_file)
            self.generate_map_btn.setEnabled(os.path.exists(input_file))
            self.generate_map_btn.setEnabled(exists)
        except:
            self.generate_map_btn.setEnabled(False)

    def toggle_select_all_cancers(self, state):
        """
        UPDATE ME
        Check or uncheck every cancer-type item to mirror the “Select All Cancer Types” checkbox.

        Behavior:
        - If `state` is `Qt.Checked`, sets all items in `self.cancer_list` to `Qt.Checked`.
        - Otherwise (including `Qt.Unchecked` and `Qt.PartiallyChecked`), sets all items to `Qt.Unchecked`.

        Args:
            state (int | Qt.CheckState): The state passed from the checkbox’s
                `stateChanged` signal.

        Side Effects:
            - Mutates the check state of each `QListWidgetItem` in `self.cancer_list`.
            - May emit per-item change signals (e.g., `itemChanged`) for each update.

        Notes:
            - I may need to edit this (SORRY SOPHIA) to work with the GLOBOCAN data, as this already includes       <-- CHECK ME
            an all cancers option? Will need to edit the load cancer types also probs sorry!

        Requirements/Assumptions:
            - `self.cancer_list` is a `QListWidget` whose items are checkable
            (`Qt.ItemIsUserCheckable` flag set).

        Returns:
            None
        """
        
        check_state = Qt.Checked if state == Qt.Checked else Qt.Unchecked
        self._suppress_item_changed = True
        try:
            for i in range(self.cancer_table.topLevelItemCount()):
                item = self.cancer_table.topLevelItem(i)
                name = item.text(0)

                if check_state == Qt.Checked:
                    # Select-all: check everything EXCEPT the two sentinel buckets
                    if _is_sentinel_cancer(name):
                        item.setCheckState(0, Qt.Unchecked)
                    else:
                        item.setCheckState(0, Qt.Checked)
                else:
                    # Unselect-all: uncheck everything, including sentinels
                    item.setCheckState(0, Qt.Unchecked)
        finally:
            self._suppress_item_changed = False

    def initiate_download(self):
        """
        Validates the selected country, handles overwrite confirmation, and starts a
        background download of the country’s raw WorldPop raster.

        Steps:
        1) Read the selected country from `self.country_combo`. If none is selected,
            show a critical error dialog and return.
        2) Resolve the ISO-3 code via `pycountry.countries.lookup(country)` and build the
            expected target path:
                a_population_density/raw_from_worldpop/{iso3_lower}_raw.tif
        3) If the target file already exists, prompt the user to overwrite. If declined,
            return; otherwise set `overwrite_download = True`.
        4) Initialise UI state for a long-running task:
            - Reset and show the progress bar.
            - Disable the Download button to prevent concurrent launches.
        5) Create and start `DownloadThread(country, output_dir, overwrite_download)`,
            wiring signals:
            - `progress_updated(int)` connect to `self.update_progress_bar`
            - `finished(bool, str)`   connect to `self.download_complete`

        Notes:
        - Any lookup/IO error during the existence check is surfaced via a critical
            message box and the method returns early.
        - This method is intended to be connected to the Download button created in
            `setup_ui()`; all UI updates occur on the main thread via connected slots.

        Side Effects:
        - Displays modal dialogs (errors, overwrite confirmation).
        - Mutates the progress bar visibility/value and the Download button enabled state.
        - Spawns a worker thread stored on `self.download_thread`.

        Returns:
        None
        """
        country = self.country_combo.currentText()
        if not country:
            QMessageBox.critical(self, "Error", "Please select a country.")
            return

        output_dir = "a_population_density/raw_from_worldpop"
        if not output_dir:
            return

        # Check if file exists
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            target_file = os.path.join(output_dir, f"{country_code}_raw.tif")
            
            if os.path.exists(target_file):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"File already exists at:\n{target_file}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                overwrite_download = True
            else:
                overwrite_download = False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't check file: {str(e)}")
            return

        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.download_btn.setEnabled(False)

        self.download_thread = DownloadThread(country, output_dir, overwrite_download)
        self.download_thread.progress_updated.connect(self.update_progress_bar)
        self.download_thread.finished.connect(self.download_complete)
        self.download_thread.start()

    def initiate_resample(self):
        """
        Validate inputs, handle overwrite confirmation, and launch a background
        resampling job for the selected country at the chosen resolution.

        Steps:
        1) Read selections:
            - Country from `self.country_combo`
            - Resolution (km) from `self.resolution_combo` (parsed as float)
        2) Resolve ISO-3 code via `pycountry.countries.lookup(country)` and build the
            expected output filename:
                a_population_density/resampled/{iso3_lower}_{resolution}km.tif
            (e.g., resolution 1 to "1.0km"; check file naming matches this format.).                    <-- CHECK ME
        3) If the target file already exists, prompt for overwrite. If declined, return.
        4) Create and start `ResampleThread(country, resolution, input_dir, output_dir, overwrite)`.
            - Connect `finished(dict)` to `self.resample_complete`
        5) Disable the Resample button and show a non-modal “Processing” QMessageBox
            with a Cancel button to indicate work is in progress.

        Notes:
        - Any exception during lookup/path checks is surfaced via a critical QMessageBox
            and the method returns early.
        - The Cancel button on the progress dialog is not wired to cancel the thread;
            add a handler (e.g., call `requestInterruption()` and check in worker) if we
            want true cancellation semantics.                                                          <-- CHECK ME

        Side Effects:
        - Displays modal dialogs (error/overwrite prompt) and a non-modal progress dialog.
        - Disables `self.resample_btn` until completion.
        - Spawns a `ResampleThread` and stores it on `self.resample_thread`.

        Returns:
        None
        """
        country = self.country_combo.currentText()
        resolution = float(self.resolution_combo.currentText())
    
        output_dir = "a_population_density/resampled"
        if not output_dir:
            return
        
        input_dir = "a_population_density/raw_from_worldpop"
        if not input_dir:
            return        

        # Check if file exists already
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            target_file = os.path.join(output_dir, f"{country_code}_{resolution}km.tif")
            
            if os.path.exists(target_file):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"File already exists at:\n{target_file}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                overwrite_resample = True
            else:
                overwrite_resample = False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't check file: {str(e)}")
            return

        self.resample_thread = ResampleThread(country, resolution, input_dir, output_dir, overwrite_resample)
        self.resample_thread.finished.connect(self.resample_complete)
        self.resample_thread.start()
        
        self.resample_btn.setEnabled(False)
        self.processing_msgbox = QMessageBox(self)
        self.processing_msgbox.setWindowTitle("Processing")
        self.processing_msgbox.setText("Resampling in progress...")
        self.processing_msgbox.setStandardButtons(QMessageBox.Cancel) 
        self.processing_msgbox.setModal(False) # no buttons, so user can’t close it manually
        self.processing_msgbox.show()

    def _on_map_type_changed_capacity_controls(self,text): 
        if text in ["Treated by Radiotherapy", "Optimally Treated by Radiotherapy"]: 
            self.capacity_weighted_checkbox.setVisible(True) 
            self.linac_capacity_label.setVisible(True)
            self.linac_capacity_slider.setVisible(True)
            self.linac_capacity_spin.setVisible(True) 
        else:
            self.capacity_weighted_checkbox.setVisible(False) 
            self.linac_capacity_label.setVisible(False)
            self.linac_capacity_slider.setVisible(False)
            self.linac_capacity_spin.setVisible(False)  

    def _expected_access_png(self, iso3_lower: str, resolution: float, value_to_plot: str) -> str:
        tag = "access_prob_popw" if value_to_plot == "pop_weighted" else "access_prob_dist"
        base = f"{iso3_lower}_{resolution}km_{tag}.png"
        return os.path.join("c_probability_of_access", "access_probability_maps", base)

    def _expected_distance_png(self, iso3_lower: str, resolution: float) -> str:
        base = f"{iso3_lower}_{resolution}km_distance_to_linac.png"
        return os.path.join("c_probability_of_access", "access_probability_maps", base)


    def initiate_cancer_type_map_generate(self):
        """
        UPDATE ME
        Orchestrates generation of either a Population Density map or a Cancer-Type map
        for the selected country at the chosen resolution, without blocking the GUI.

        Workflow:
        1) Read UI selections:
            - Country (`self.country_combo`)
            - Map type (`self.map_type_combo`): one of
                  "Population Density"
                  "Treated by Radiotherapy"
                  "Optimally Treated by Radiotherapy"
                  "Cancer Incidence"
            - Resolution in km (`self.resolution_combo`, parsed to float)
            - Cancer types (checked items in `self.cancer_list`), unless the map type
            is "Population Density" (no cancer types required).
            If a cancer map is requested and no types are selected, show a critical
            message box and return.
        2) Determine output intent and flags:
            - `include_RT_utilisation`  ← True iff "Treated by Radiotherapy"
            - `include_optimal_RT_utilisation` ← True iff "Optimally Treated by Radiotherapy"
            - Choose `filename_prefix` and `output_subfolder` accordingly, then
            `os.makedirs(output_subfolder, exist_ok=True)`.                                     <-- CHECK ME (will need to see if applies beyond CRUK Data)
        3) Resolve the ISO-3 country code via `pycountry.countries.lookup(country)` and
            build the input raster path:
                a_population_density/resampled/{iso3_lower}_{resolution}km.tif
            Construct a safe label from selected cancer types and the target PNG path:
                {output_subfolder}/{iso3_lower}_{safe_label}_{resolution}km_{prefix}_density.png
        4) If the target file already exists, prompt the user to overwrite; set an
            `overwrite` flag accordingly. On exceptions, show an error dialog and return.
        5) Update UI state and threading:
            - Call `self.update_status(...)` to report progress.
            - Disable `self.generate_map_btn`.
            - If a prior `self.map_thread` is running, request it to stop (`quit()`/`wait()`).
            - Start the appropriate worker:
                  Population maps → `PopulationMapThread(...)`
                  Cancer maps     → `MapGenerationThread(...)`                                 <-- CHECK ME (will need to add in probability maps + major cities overlay)
            Connect:
                  `finished(...)` → `self.cancer_type_map_completed`
                  `error(str)`    → `self.on_map_generation_error`
            - The worker emits image bytes plus output paths on success.

        Side Effects:
        - Displays modal dialogs for validation/overwrite and errors.
        - Creates output directories if needed.
        - Disables the Generate button during processing.
        - Spawns a QThread and assigns it to `self.map_thread`.
        - Writes status text via `self.update_status`.

        Requirements/Assumptions:
        - A resampled population raster exists at
            `a_population_density/resampled/{iso3_lower}_{resolution}km.tif`.
        - `PopulationMapThread` and `MapGenerationThread` classes are available and emit
            the documented `finished` and `error` signals.
        - Slots `self.cancer_type_map_completed(bytes, str, str)` and
            `self.on_map_generation_error(str)` exist.
        - `countries` refers to `pycountry.countries`.

        Returns:
        None
        """
        country = self.country_combo.currentText()
        map_type_text = self.map_type_combo.currentText()
        
        is_access_map    = self._is_access_map_text(map_type_text)
        is_distance_map  = self._is_distance_map_text(map_type_text)

        self._is_probability_map = bool(is_access_map)
        self._is_distance_map    = bool(is_distance_map)




        if map_type_text == "Population Density" or is_access_map:
            selected_cancer_types = []
        else:
            selected_cancer_types = self.get_selected_cancer_types()
            if not selected_cancer_types:
                QMessageBox.critical(self, "error", "Please select at least one cancer type.")
                return

        resolution = float(self.resolution_combo.currentText())
        map_type_text = self.map_type_combo.currentText()

        # Construct target file name
        safe_label = "_".join(ct.replace(" ", "_") for ct in selected_cancer_types) if selected_cancer_types else "All cancers"
        cancers = ", ".join(selected_cancer_types) if selected_cancer_types else "All cancers"


        # Set flags
        include_RT_utilisation = map_type_text == "Treated by Radiotherapy"
        include_optimal_RT_utilisation = map_type_text == "Optimally Treated by Radiotherapy"

        # --- setting a flag for capacity --- 
        include_capacity_weighted = self.capacity_weighted_checkbox.isChecked() 

        #Getting linac capacity and number of linacs from user inputs: 
        linac_capacity = float(self.linac_capacity_spin.value()) if self.capacity_weighted_checkbox.isChecked() else None 
        country_code = countries.lookup(country).alpha_3.lower()
        n_linacs = get_n_liancs_from_excel(country_code) 

        if map_type_text == "Population Density":
            filename_prefix = "population"
            output_subfolder = "a_population_density/population_density_maps"
            title = f"{country} — Population Density ({resolution} km)"

        elif map_type_text == ACCESS_PROB_DIST:
            filename_prefix = "access_prob_distance"
            output_subfolder = "c_probability_of_access/access_probability_maps"
            title = f"{country} — Probability of treatment access (distance) (λ={resolution} km)"    #<- update me 

        elif map_type_text == ACCESS_PROB_POPW:
            filename_prefix = "access_prob_popweighted"
            output_subfolder = "c_probability_of_access/access_probability_maps"
            title = f"{country} — Probability of treatment access (population-weighted) (λ={resolution} km)"
        
        elif include_optimal_RT_utilisation:
            filename_prefix = "optimally_treated"
            output_subfolder = "b_cancer_incidence/cancer_type_maps/optimally_treated"
            title = f"{country} — {map_type_text}: {cancers} ({resolution} km)"
        
        elif include_RT_utilisation:
            filename_prefix = "treated"
            output_subfolder = f"b_cancer_incidence/cancer_type_maps/{filename_prefix}_maps"
            title = f"{country} — {map_type_text}: {cancers} ({resolution} km)"

        elif map_type_text == ACCESS_DIST_NEAREST:
            filename_prefix = "distance_to_linac"
            output_subfolder = "c_probability_of_access/access_probability_maps"
            title = f"{country} — Distance to nearest LINAC (km)"
        
        else:
            filename_prefix = "incidence"
            output_subfolder = f"b_cancer_incidence/cancer_type_maps/{filename_prefix}_maps"
            title = f"{country} — {map_type_text}: {cancers} ({resolution} km)"

        # We need ISO3 for the per-country CSV name:
        iso3 = countries.lookup(country).alpha_3

        # Build a suffix that mirrors the generator's "fallback" logic
        suffix = self._rt_title_suffix(
            iso3=iso3,
            cancers=selected_cancer_types,
            include_actual=include_RT_utilisation,
            include_optimal=include_optimal_RT_utilisation
        )

        # Append and expose to the plotting code
        title = title + suffix
        self._current_title = title  


        os.makedirs(output_subfolder, exist_ok=True)

        try:
            country_code = countries.lookup(country).alpha_3.lower()
            population_raster_path = f"a_population_density/resampled/{country_code}_{resolution}km.tif"

            if is_access_map:
                vtp = "pop_weighted" if map_type_text == ACCESS_PROB_POPW else "prob"
                target_file = self._expected_access_png(country_code, resolution, vtp)
            elif is_distance_map:
                target_file = self._expected_distance_png(country_code, resolution)    
            else:
                # existing naming for population/cancer maps:
                safe_label = "_".join(ct.replace(" ", "_") for ct in selected_cancer_types) if selected_cancer_types else "All cancers"
                target_file = f"{output_subfolder}/{country_code}_{safe_label}_{resolution}km_{filename_prefix}_density.png"

            """# THEN check for overwrite
            if os.path.exists(target_file):
                reply = QMessageBox.question(
                    self, "Overwrite?", f"{target_file} exists. Overwrite?", QMessageBox.Yes | QMessageBox.No
                )
                overwrite = reply == QMessageBox.Yes
            else:
                overwrite = False"""
            overwrite = True


        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        # Update status
        self.update_status(f"Generating map for {', '.join(selected_cancer_types)}...")
        self.generate_map_btn.setEnabled(False)

        if self.map_thread is not None and self.map_thread.isRunning():
            self.map_thread.quit()
            self.map_thread.wait()

        # Create and start the thread
        if filename_prefix == "population":
            self.map_thread = PopulationMapThread(
                country_code=country_code,
                resolution=resolution,
                population_raster_path=population_raster_path,
                output_dir=output_subfolder,
                overwrite_existing=overwrite
            )

        elif is_distance_map:
            # Distance doesn’t use λ or multi-LINAC “mode” — always nearest
            self._current_title = title
            self.map_thread = AccessMapThread(
                country_code=country_code,
                resolution=resolution,
                population_raster_path=population_raster_path,
                output_dir=output_subfolder,
                overwrite_existing=True,
                lambda_km=None,          # ignored by generator for distance
                cutoff_factor=5.0,       # ignored for distance
                value_to_plot="distance_km",  # <— KEY
                mode="nearest",          # <— force nearest
            )
        
        elif is_access_map: 
            """self.map_thread = AccessMapThread(country_code=country_code,
            resolution=resolution,
            population_raster_path=population_raster_path,
            output_dir=output_subfolder,
            overwrite_existing=overwrite
        )"""
            vtp = "pop_weighted" if map_type_text == ACCESS_PROB_POPW else "prob"
            mode = self._current_access_mode()

            lam = float(self.lambda_spin.value()) if self.lambda_spin.isVisible() else float(self.resolution_combo.currentText())
            # Update title with λ
            if vtp == "prob":
                title = f"{country} — {ACCESS_PROB_DIST} (λ={lam:g} km)"
            else:
                title = f"{country} — {ACCESS_PROB_POPW} (λ={lam:g} km)"
            self._current_title = title

            # keep your existing arguments; add optional lambda if you expose a control for it
            self.map_thread = AccessMapThread(
                country_code=country_code,                 # e.g. "gbr"
                resolution=resolution,                     # km (also default λ if you like)
                population_raster_path=population_raster_path,
                output_dir=output_subfolder,
                overwrite_existing=True,
                lambda_km=lam,
                cutoff_factor=5.0,                         # same behaviour as before
                value_to_plot=vtp,                    
                mode=mode,                            # or "multi" for independence-product
            )

        else:
            # Call thread for cancer maps
            self.map_thread = MapGenerationThread(
                country_code = country_code,
                cancer_types = selected_cancer_types,
                resolution = resolution,
                population_raster_path = population_raster_path,
                overwrite_cancer_type_map = overwrite,
                include_RT_utilisation = include_RT_utilisation,
                include_optimal_RT_utilisation = include_optimal_RT_utilisation, 
                include_capacity_weighted = include_capacity_weighted,
                linac_capacity = linac_capacity,
                n_linacs = n_linacs,
            )

        self.map_thread.finished.connect(self.cancer_type_map_completed)
        self.map_thread.error.connect(self.on_map_generation_error)
        self.map_thread.start()

    # End of main helper methods for setup_ui():

    # ---- Small Sub-Helpers ----

    def update_progress_bar(self, value):
        """
        Update the download progress indicator.

        Intended as the slot connected to `DownloadThread.progress_updated(int)`.
        Sets the value of `self.progress` (a QProgressBar). 

        Args:
            value (int): New progress value for the bar.

        Returns:
            None
        """
        self.progress.setValue(value)

    def download_complete(self, success, message):
        """
        Finalises the download workflow: restore UI state, notify the user, and
        refresh resample availability.

        Intended as the slot connected to `DownloadThread.finished(bool, str)`.
        Hides the progress bar, re-enables the Download button, and displays an
        information dialog on success or an error dialog on failure. On success,
        it also calls `check_resample_availability()` so the “Resample” button
        becomes enabled if the new raster is present.

        Args:
            success (bool): True if the download completed successfully.
            message (str): User-facing status text (e.g., output path or error).

        Side Effects:
            - Updates progress bar visibility and Download button enabled state.
            - Shows a QMessageBox (information or critical).
            - Enables the Resample button via `check_resample_availability()`.

        Returns:
            None
        """
        self.progress.setVisible(False)
        self.download_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.check_resample_availability()
        else:
            QMessageBox.critical(self, "Error", message)

    def resample_complete(self, result):
        """
        Handles completion of the background resampling job: restore UI, show outcome,
        and enable subsequent map generation.

        Intended as the slot connected to `ResampleThread.finished(dict)`. It:
            Re-enables the Resample button.  
            Closes and disposes of the non-modal “Processing” QMessageBox if it exists.  
            On success (`result['success'] == True`):
                - Builds a summary message including
                `original_population`, `resampled_population`, and `output_path`.  
                - Shows an information dialog to the user.  
                - Enables the Generate Map button to allow the next step.  
            On failure:
                - Shows a critical error dialog with `result['message']`.

        Args:
            result (dict): Result payload emitted by `ResampleThread`. Expected keys:
                - 'success' (bool): Overall outcome.
                - 'original_population' (int | float): Aggregate population from the input raster.
                - 'resampled_population' (int | float): Aggregate population from the resampled raster.
                - 'output_path' (str): Path to the written resampled raster.
                - 'message' (str): Human-readable error text (present/used on failure).

        Side Effects:
            - Mutates UI control states (buttons, message boxes).
            - Displays QMessageBox dialogs (information or critical).
            - Enables map generation on success.

        Returns:
            None
        """
        self.resample_btn.setEnabled(True)

        if hasattr(self, 'processing_msgbox'):
            self.processing_msgbox.close()
            del self.processing_msgbox
        
        if result['success']:
            msg = (
                f"Resampling successful!\n\n"
                f"Original population: {result['original_population']:,.0f}\n"
                f"Resampled population: {result['resampled_population']:,.0f}\n"
                f"Saved to: {result['output_path']}"
            )
            QMessageBox.information(self, "Success", msg)
            self.generate_map_btn.setEnabled(True)  # enable map generation
        else:
            QMessageBox.critical(self, "Error", result['message'])

    def update_status(self, message):
        """
        Append a line of status text to the UI’s log area.

        Args:
            message (str): The message to display. Should be short and human-readable.

        Notes:
            - Call this on the GUI thread. If emitting from a worker thread, connect a
            signal to this slot so Qt marshals the call safely.
            - For very frequent updates, consider throttling to avoid UI jank.
        """
        self.status_text.append(message)
    
    def display_image(self, image_data):
        """
        Preview an already-coloured PNG/JPEG (no live colour scaling).
        """
        if not image_data:
            self.ax.clear(); self.ax.set_axis_off()
            if self._cbar:
                self._cbar.remove(); self._cbar = None
                self.ax.set_title(getattr(self, "_current_title", ""), fontsize=11, pad=6)
            self.canvas.draw_idle()
            self.update_status("No image to display.")
            return

        try:
            img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            arr = np.array(img)
            self._last_image_arr = arr
        except Exception as e:
            self.ax.clear(); self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, f"Image decode error:\n{e}",
                        ha="center", va="center", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            self.update_status(f"Image decode error: {e}")
            return

        self.ax.clear(); self.ax.set_axis_off()
        # Show as-is (no norm/cmap on RGBA)
        self.ax.imshow(arr, origin="upper", interpolation="nearest")
        self.ax.set_aspect("equal", adjustable="box")
        # Remove any old colourbar to avoid confusion
        if self._cbar:
            self._cbar.remove(); self._cbar = None
        self.canvas.draw_idle()
        self.update_status("PNG preview displayed.")

    def cancer_type_map_completed(self, result):
        """
        Handle completion of map generation: re-enable UI, render preview, and log outputs.

        Intended as the slot connected to:
        - `PopulationMapThread.finished(bytes, str, str)`
        - `MapGenerationThread.finished(bytes, str, str)`

        Behavior:
        - Re-enables the Generate Map button.
        - If `image_data` is provided, displays the image via `display_image(image_data)`
            and appends a status message including the TIFF and PNG output paths.
        - If `image_data` is missing/empty, logs that no image data was returned.

        Args:
            image_data (bytes | None): Encoded image data suitable for QPixmap/QImage.
            tif_path (str): Filesystem path to the generated GeoTIFF.
            png_path (str): Filesystem path to the rendered PNG.

        Side Effects:
            - Mutates the enabled state of `self.generate_map_btn`.
            - Updates the preview area via `display_image`.
            - Appends messages to the status log via `update_status`.

        Threading:
            - Should be invoked on the GUI thread (Qt will marshal the signal-slot call).

        Returns:
            None
        """
        self.generate_map_btn.setEnabled(True)

        image_data = result["image_bytes"]
        tif_path = result["tif_path"]
        png_path = result["png_path"] 
        mode = result["mode"] 
        
        if tif_path:  # best path: draw numeric data with live scale
            self.display_raster_from_tif(tif_path)
            self.update_status(f"Map generated successfully!\nTIFF: {tif_path}\nPNG: {png_path}")
            return
        
        if image_data:
            # Disable scale controls when showing RGBA
            self.exp_spin.setEnabled(False)
            self.auto_btn.setEnabled(False)
            self.display_image(image_data)
            self.update_status(f"(Previewed PNG only)\nPNG: {png_path or '(in-memory)'}")
        else:
            self.update_status("Map generated but no image data returned.")

    def on_map_generation_error(self, error_msg):
        """
        Responds to a map-generation failure: restore UI, notify the user, and log details.

        Intended as the slot connected to:
        - `PopulationMapThread.error(str)`
        - `MapGenerationThread.error(str)`

        Behavior:
        - Re-enables the Generate Map button so the user can try again.
        - Displays a critical QMessageBox with the error details.
        - Appends a concise error line to the status log via `update_status`.

        Args:
            error_msg (str): Human-readable description of the error that occurred.

        Side Effects:
            - Mutates the enabled state of `self.generate_map_btn`.
            - Shows a modal error dialog.
            - Writes to the status text area.
        """
        self.generate_map_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Map generation failed:\n{error_msg}")
        self.update_status(f"Error: {error_msg}")

    def get_selected_cancer_types(self) -> list[str]:
        out = []
        for i in range(self.cancer_table.topLevelItemCount()):
            item = self.cancer_table.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                out.append(item.text(0))
        return out
    
    # End of small sub-helpers
    
# ==== Run Below ====
if __name__ == "__main__":
    #ensuring proper shutdown
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)

    window = GeoSpacRadAccess()
    window.show()
    ret = app.exec_() 

    #Wait for background to finish (if still running) 
    if window.map_thread and window.map_thread.isRunning():
        window.map_thread.quit()
        window.map_thread.wait()

    sys.exit(ret)