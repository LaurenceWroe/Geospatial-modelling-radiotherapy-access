import sys
import os
import subprocess
import pandas as pd 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog, QHBoxLayout, QGroupBox, QSplitter
)
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QScrollArea, QTextEdit, QListWidget, QListWidgetItem

import io
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from pycountry import countries
from a_population_density.download_worldpop import download_worldpop
from a_population_density.resample_population import resample_population
from b_cancer_incidence.generate_cancer_type_map import generate_cancer_type_map
from b_cancer_incidence.generate_cancer_type_map import generate_population_density_map_only
from c_probability_of_access.visualization.plot_accessibility_probability import calculate_accessibility_probability
from c_probability_of_access.visualization.plot_accessibility_probability import plot_accessibility_probability
#from c_probability_of_access.plot_probability_cutoff import plot_accessibility_probability
#from c_probability_of_access.plot_probability import plot_accessibility_probability

# All Qthreads below for resampling, downloading and mapping:

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
        include_fraction (bool): Forwarded flag (e.g., include standard radiotherapy fraction layer).
        include_optimal_fraction (bool): Forwarded flag (e.g., include optimal radiotherapy fraction layer).

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
            include_fraction=True,
            include_optimal_fraction=False,
        )
        thread.finished.connect(self.on_map_ready)
        thread.error.connect(self.on_map_error)
        thread.start()
    """
    finished = pyqtSignal(bytes, str, str)
    error = pyqtSignal(str)

    def __init__(self, country_code, cancer_types, resolution, population_raster_path, overwrite_cancer_type_map=False, include_fraction=False, include_optimal_fraction=False, include_access_map = False):
        super().__init__()
        self.country_code = country_code
        self.cancer_types = cancer_types
        self.resolution = resolution
        self.population_raster_path = population_raster_path
        self.overwrite_cancer_type_map = overwrite_cancer_type_map
        self.include_fraction = include_fraction
        self.include_optimal_fraction = include_optimal_fraction

    def run(self):
        try:
            print(f"[THREAD] Starting map generation for {self.country_code}...")
            image_data, tif_path, png_path = generate_cancer_type_map(
                country_code=self.country_code,
                cancer_types=self.cancer_types,
                resolution=self.resolution,
                population_raster_path=self.population_raster_path,
                return_image=True,
                overwrite_cancer_type_map=self.overwrite_cancer_type_map,
                include_fraction=self.include_fraction,
                include_optimal_fraction = self.include_optimal_fraction
            )
            print(f"[THREAD] Finished map generation.")

            self.finished.emit(image_data, tif_path, png_path)

        except Exception as e:
            print(f"[THREAD] Error during map generation: {e}")
            self.error.emit(str(e))

class AccessMapThread(QThread): 
    finished = pyqtSignal(bytes, str, str) 
    error = pyqtSignal(str) 

    def __init__(self, country_code, resolution, population_raster_path, output_dir, overwrite_existing=False):
        super().__init__()
        self.country_code = country_code 
        self.resolution = resolution 
        self.population_raster_path = population_raster_path 
        self.output_dir = output_dir 
        self.overwrite_existing = overwrite_existing 
    
        def run(self):
            try:
                from c_probability_of_access.visualization.plot_accessibility_probability import plot_accessibility_probability

                output_path = os.path.join(self.output_dir, f"{self.country_code}_{self.resolution}km_access_probability.png")
                print(f"[THREAD] Generating access map at {output_path}...")

                plot_accessibility_probability(
                    population_raster_path=self.population_raster_path,
                    linac_excel_path=f"c_probability_of_access/linac/{self.country_code}_DIRAC.xlsx",  # Adjust if needed
                    output_path=output_path
                )

                with open(output_path, "rb") as f:
                    image_data = f.read()

                self.finished.emit(image_data, "", output_path)

            except Exception as e:
                print(f"[THREAD] Error generating access map: {e}")
                self.error.emit(str(e))
# Main window

class GeoSpacRadAccess(QMainWindow):
    def __init__(self):
        super().__init__()
        # Adding an instance variable 
        self.recent_countries = []
        self.max_recent = 5  # or however many you want to show
        self.map_thread = None # ensures self.map_thread is always defined
        self.setup_ui()
    
    # Initial UI setup

    def setup_ui(self):
        """
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
        self.setFixedSize(1200, 800)

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
        cancer_types = self.load_cancer_types()

        self.cancer_list = QListWidget()
        self.cancer_list.setSelectionMode(QListWidget.MultiSelection)

        for ctype in cancer_types:
            item = QListWidgetItem(ctype)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.cancer_list.addItem(item)

        # "Select All Cancers" Checkbox
        self.select_all_checkbox = QCheckBox("Select All Cancer Types") 
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all_cancers)

        # Why is this commented (Archie)
        #self.include_fraction_checkbox = QCheckBox("Include radiotherapy fraction") 
        #self.include_fraction_checkbox.setChecked(False) 

        # Map type box
        self.map_type_label = QLabel("Select map to generate:")
        self.map_type_combo = QComboBox() 
        self.map_type_combo.addItems(["Cancer Incidence", "Treated by Radiotherapy", "Optimally Treated by Radiotherapy", "Population Density", "Effective Access (Population-Weighted)"])
        self.generate_map_btn = QPushButton("Generate Map")
        self.generate_map_btn.setEnabled(False)  
        self.check_cancer_map_availability() # check if cancer map generation is available, if so enable the button

        map_layout.addWidget(self.cancer_label)
        map_layout.addWidget(self.cancer_list)
        map_layout.addWidget(self.select_all_checkbox) 
        map_layout.addWidget(self.map_type_label)
        map_layout.addWidget(self.map_type_combo)
        map_layout.addWidget(self.generate_map_btn)
        map_group.setLayout(map_layout)


        # ---- Arranging ----
        # Add groups to left layout
        left_layout.addWidget(download_group)
        left_layout.addWidget(resample_group)
        left_layout.addWidget(map_group)

        # Setting size of left panel
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(450)
        left_panel.setMinimumWidth(350)


        # ==== Right panel for image display ====

        right_panel  = QWidget()
        right_layout = QVBoxLayout()
        
        self.image_label = QLabel("Generated map will appear here")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(80)
        self.status_text.setReadOnly(True)
        
        right_layout.addWidget(QLabel("Generated Map:"))
        right_layout.addWidget(self.image_label)

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
        self.country_combo.currentTextChanged.connect(self.check_cancer_map_availability) 
        self.resolution_combo.currentTextChanged.connect(self.check_cancer_map_availability) 
        self.map_type_combo.currentTextChanged.connect(self.check_cancer_map_availability)


    # ==== HELPER METHODS for setup_ui(): re-ordered by appearance ===

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

    def load_cancer_types(self, excel_path="b_cancer_incidence/cancer_type_radiotherapy.xlsx"):
        """
        Load and return a sorted list of unique cancer types from an Excel file.

        This helper reads the Excel sheet at `excel_path` using pandas, normalizes
        column names to lowercase (stripped), and looks for a column named
        "cancer type" (case-insensitive via normalization). If found, it extracts
        non-null values, de-duplicates them, sorts alphabetically, and returns the
        resulting list. If the column is missing or any error occurs while reading
        the file, an error dialog is shown and an empty list is returned.

        Args:
            excel_path (str): Path to the Excel file containing a "Cancer Type"
                column (name treated case-insensitively). Defaults to
                "b_cancer_incidence/cancer_type_radiotherapy.xlsx".

        Returns:
            list[str]: Sorted unique cancer type names, or an empty list on failure
            or if the expected column is absent.

        Side Effects:
            - Displays a critical QMessageBox on exceptions.
            - Performs file I/O synchronously; for large files consider offloading
            to a worker to keep the UI responsive.

        Requirements/Assumptions:
            - `pandas` is available as `pd`.
            - `self` is a QWidget (or subclass) so QMessageBox can parent to it.
            - The Excel file contains a "Cancer Type" column (any case).
        """
        try:
            df = pd.read_excel(excel_path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "cancer type" in df.columns:
                types = sorted(df["cancer type"].dropna().unique())
                return types
            else:
                return []
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load cancer types: {e}")
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
        for i in range(self.cancer_list.count()):
            item = self.cancer_list.item(i)
            item.setCheckState(check_state)

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

    def initiate_cancer_type_map_generate(self):
        """
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
            - `include_fraction`  ← True iff "Treated by Radiotherapy"
            - `include_optimal_fraction` ← True iff "Optimally Treated by Radiotherapy"
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
       


        if map_type_text in ["Population Density", "Effective Access (Population-Weighted)"]:
            selected_cancer_types = []  # no cancer types
        else:
            selected_cancer_types = []
            for i in range(self.cancer_list.count()):
                item = self.cancer_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected_cancer_types.append(item.text())

            if not selected_cancer_types:
                QMessageBox.critical(self, "Error", "Please select at least one cancer type.")
                return

        resolution = float(self.resolution_combo.currentText())
        map_type_text = self.map_type_combo.currentText()

        # Construct target file name
        safe_label = "_".join(ct.replace(" ", "_") for ct in selected_cancer_types)

        # Set flags
        include_fraction = map_type_text == "Treated by Radiotherapy"
        include_optimal_fraction = map_type_text == "Optimally Treated by Radiotherapy"

        if map_type_text == "Population Density":
            filename_prefix = "population"
            output_subfolder = "a_population_density/population_density_maps"

        elif map_type_text == "Effective Access (Population-Weighted)": 
            filename_prefix = "access_probability" 
            output_subfolder = "c_probability_of_access/access_probability_maps"
        
        elif include_optimal_fraction:
            filename_prefix = "optimally_treated"
            output_subfolder = "b_cancer_incidence/cancer_type_maps/optimally_treated"
        
        elif include_fraction:
            filename_prefix = "treated"
            output_subfolder = f"b_cancer_incidence/cancer_type_maps/{filename_prefix}_maps"
        
        else:
            filename_prefix = "incidence"
            output_subfolder = f"b_cancer_incidence/cancer_type_maps/{filename_prefix}_maps"


        os.makedirs(output_subfolder, exist_ok=True)

        try:
            country_code = countries.lookup(country).alpha_3.lower()
            population_raster_path = f"a_population_density/resampled/{country_code}_{resolution}km.tif"

            # NOW construct the correct target file path
            target_file = f"{output_subfolder}/{country_code}_{safe_label}_{resolution}km_{filename_prefix}_density.png"

            # THEN check for overwrite
            if os.path.exists(target_file):
                reply = QMessageBox.question(
                    self,
                    "Overwrite?",
                    f"{target_file} exists. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                overwrite = reply == QMessageBox.Yes
            else:
                overwrite = False

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

        
        elif map_type_text == "Effective Access (Population-Weighted)": 
            self.map_thread = AccessMapThread(country_code=country_code,
            resolution=resolution,
            population_raster_path=population_raster_path,
            output_dir=output_subfolder,
            overwrite_existing=overwrite
        )

        else:
            # Call thread for cancer maps
            self.map_thread = MapGenerationThread(
                country_code,
                selected_cancer_types,
                resolution,
                population_raster_path,
                overwrite,
                include_fraction,
                include_optimal_fraction
            )

            self.map_thread.finished.connect(self.cancer_type_map_completed)
            self.map_thread.error.connect(self.on_map_generation_error)
            self.map_thread.start()

    # End of main helper methods for setup_ui():

    # ==== Small Sub-Helpers ====

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
        Render an in-memory image into the preview area and report status.

        Loads `image_data` (expected to be bytes/bytes-like, e.g., PNG or JPEG)
        into a QPixmap, scales it to fit `self.image_label` while preserving aspect
        ratio (using smooth interpolation), and displays it. If no data is provided,
        clears the preview and shows a placeholder message. In both cases, appends
        a short status line via `update_status(...)`.

        Args:
            image_data (bytes | bytearray | memoryview | None): Encoded image data to
                display. If falsy, the preview is cleared instead.

        Side Effects:
            - Mutates `self.image_label` pixmap/contents.
            - Appends a message to the status log via `self.update_status`.

        Threading:
            - Call from the GUI thread. If originating in a worker thread, emit a
            signal and connect it to this slot so Qt marshals to the main thread.
        """
        if image_data:
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.width(), 
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.update_status("Map displayed successfully!")
        else:
            self.image_label.clear()
            self.image_label.setText("No image to display")
            self.update_status("Failed to generate map image.")

    def cancer_type_map_completed(self, image_data, tif_path, png_path):
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
        if image_data:
            self.display_image(image_data)
            self.update_status(f"Map generated successfully!\nTIFF: {tif_path}\nPNG: {png_path}")
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

    # End of small sub-helpers


if __name__ == "__main__":
    #ensuring proper shutdown
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    window = GeoSpacRadAccess()
    window.show()
    ret = app.exec_() 

    #Wait for background to finish (if still running) 
    if window.map_thread and window.map_thread.isRunning():
        window.map_thread.quit()
        window.map_thread.wait()

    sys.exit(ret)