# H3_GUI_polygon_only.py
import sys
from pathlib import Path
import pycountry
import xarray as xr
from PyQt5 import QtWidgets, QtCore, QtGui

from H3.download_h3_test import load_h3_population, download_H3_population_density_zipped
from H3.resample_h3_population_test import generate_population_density_map_only_h3
from H3.generate_cancer_type_map_H3_test import generate_cancer_type_map_h3_polygons

# ---------- Helpers ----------
def safe_lookup_country(name_or_code: str):
    """Return pycountry.Country object from a name or ISO code."""
    try:
        return pycountry.countries.lookup(name_or_code)
    except LookupError:
        for c in pycountry.countries:
            if c.name.lower() == name_or_code.lower():
                return c
        raise ValueError(f"Could not resolve country: {name_or_code}")

def get_available_countries():
    """Return sorted list of all ISO-recognized country names."""
    return sorted([c.name for c in pycountry.countries])

def get_cancer_types_from_tensor(xarray_path="b_cancer_incidence/globocan_xarray.nc"):
    """Return list of cancer type names from xarray DataArray."""
    da = xr.open_dataarray(xarray_path)
    if "Cancer" not in da.coords:
        raise ValueError("Tensor missing 'Cancer' coordinate.")
    return list(da.coords["Cancer"].values)

def get_country_gpkg_path(country_name, input_dir="H3_zipped_pop_density_maps"):
    """Return path to gzipped H3 GeoPackage for the country."""
    country_obj = safe_lookup_country(country_name)
    alpha2 = country_obj.alpha_2.upper()
    fname = f"{alpha2}_H3_population_density_map.gpkg.gz"
    return Path(input_dir) / fname


# ---------- Worker Thread ----------
class MapWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    image_signal = QtCore.pyqtSignal(bytes)
    finished_signal = QtCore.pyqtSignal(str)

    def __init__(self, country, cancers, linac_capacity, map_type):
        super().__init__()
        self.country = country
        self.cancers = cancers or []
        self.linac_capacity = linac_capacity
        self.map_type = map_type

    def run(self):
        try:
            country_obj = safe_lookup_country(self.country)
            gpkg_path = get_country_gpkg_path(self.country)

            # Download population data if missing
            if not gpkg_path.exists():
                self.log_signal.emit(f"🌍 Downloading population data for {self.country}...")
                success, msg = download_H3_population_density_zipped(self.country)
                self.log_signal.emit(msg)
                if not success:
                    raise RuntimeError(f"Failed to download H3 population data for {self.country}")

            # Population density map
            if self.map_type == "Population Density":
                self.log_signal.emit(f"▶ Generating population density map for {self.country}...")
                result = generate_population_density_map_only_h3(
                    country_name=self.country,
                    input_dir=str(gpkg_path.parent),
                    output_dir="h3_population_maps",
                    overwrite_existing=False,
                    return_image=True,
                    h3_gpkg_path=gpkg_path
                )

            # Cancer map
            else:
                if not self.cancers:
                    self.log_signal.emit("⚠️ No cancer types selected; aborting.")
                    return

                self.log_signal.emit(f"▶ Generating cancer map for {self.country} ({self.map_type})...")
                da = xr.open_dataarray("b_cancer_incidence/globocan_xarray.nc")

                include_optimal = "optim" in self.map_type.lower()
                include_actual = "treated" in self.map_type.lower() and not include_optimal
                include_capacity_weighted = "capacity" in self.map_type.lower()

                result = generate_cancer_type_map_h3_polygons(
                    country_iso3=country_obj.alpha_3,
                    h3_gpkg_path=gpkg_path,
                    da=da,
                    cancer_types=self.cancers,
                    include_RT_utilisation=include_actual,
                    include_optimal_RT_utilisation=include_optimal,
                    include_capacity_weighted=include_capacity_weighted,
                    linac_capacity=self.linac_capacity,
                    n_linacs=5,
                    output_dir="cancer_type_maps_h3",
                    return_image=True,
                    overwrite=False
                )

            # Display image
            img = result.get("image_bytes")
            if img:
                self.image_signal.emit(img)
            else:
                self.log_signal.emit("⚠️ No image preview available.")

            # Return GeoPackage path
            gpkg_out = result.get("gpkg_path") or result.get("gpkg")
            if gpkg_out:
                self.finished_signal.emit(str(gpkg_out))
            else:
                self.log_signal.emit("⚠️ GeoPackage path not returned by generator.")

        except Exception as e:
            self.log_signal.emit(f"❌ Error: {type(e).__name__}: {e}")


# ---------- GUI ----------
class CancerMapGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cancer Map Generator (H3) — Polygon-only")
        self.resize(900, 800)

        # Load countries and cancer types safely
        try:
            self.countries = get_available_countries()
        except Exception:
            self.countries = ["United States", "United Kingdom"]

        try:
            self.cancers = get_cancer_types_from_tensor()
        except Exception:
            self.cancers = ["All cancers"]

        layout = QtWidgets.QVBoxLayout(self)

        # Controls grid
        controls = QtWidgets.QGridLayout()
        row = 0

        controls.addWidget(QtWidgets.QLabel("Select country:"), row, 0)
        self.country_combo = QtWidgets.QComboBox()
        self.country_combo.addItems(self.countries)
        controls.addWidget(self.country_combo, row, 1)
        row += 1

        controls.addWidget(QtWidgets.QLabel("Select cancer types:"), row, 0)
        self.cancer_list = QtWidgets.QListWidget()
        self.cancer_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for c in self.cancers:
            item = QtWidgets.QListWidgetItem(str(c))
            item.setCheckState(QtCore.Qt.Unchecked)
            self.cancer_list.addItem(item)
        controls.addWidget(self.cancer_list, row, 1)
        row += 1

        controls.addWidget(QtWidgets.QLabel("Linac capacity (patients/year):"), row, 0)
        hbox_capacity = QtWidgets.QHBoxLayout()
        self.capacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.capacity_slider.setRange(50, 1000)
        self.capacity_slider.setValue(250)
        self.capacity_spin = QtWidgets.QSpinBox()
        self.capacity_spin.setRange(50, 1000)
        self.capacity_spin.setValue(250)
        self.capacity_slider.valueChanged.connect(self.capacity_spin.setValue)
        self.capacity_spin.valueChanged.connect(self.capacity_slider.setValue)
        hbox_capacity.addWidget(self.capacity_slider)
        hbox_capacity.addWidget(self.capacity_spin)
        controls.addLayout(hbox_capacity, row, 1)
        row += 1

        controls.addWidget(QtWidgets.QLabel("Select map to generate:"), row, 0)
        self.map_type_combo = QtWidgets.QComboBox()
        self.map_type_combo.addItems([
            "Cancer Incidence",
            "Treated by Radiotherapy",
            "Optimally Treated by Radiotherapy",
            "Population Density"
        ])
        controls.addWidget(self.map_type_combo, row, 1)
        row += 1

        layout.addLayout(controls)

        # Generate button
        self.run_btn = QtWidgets.QPushButton("Generate Map")
        self.run_btn.clicked.connect(self.run_maps)
        layout.addWidget(self.run_btn)

        # Map display & log
        content = QtWidgets.QHBoxLayout()

        left = QtWidgets.QVBoxLayout()
        self.map_label = QtWidgets.QLabel()
        self.map_label.setFixedSize(700, 500)
        self.map_label.setScaledContents(True)
        left.addWidget(self.map_label)
        content.addLayout(left)

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Log:"))
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log)
        content.addLayout(right)

        layout.addLayout(content)
        self.worker = None

    def run_maps(self):
        country = self.country_combo.currentText()
        cancers = [
            self.cancer_list.item(i).text()
            for i in range(self.cancer_list.count())
            if self.cancer_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        linac_capacity = self.capacity_spin.value()
        map_type = self.map_type_combo.currentText()

        if "Cancer" in map_type and not cancers:
            self.log.append("⚠️ Please select at least one cancer type.")
            return

        self.log.append(f"▶ Starting generation: {country} — {map_type}")

        self.worker = MapWorker(country, cancers, linac_capacity, map_type)
        self.worker.log_signal.connect(self._append_log)
        self.worker.image_signal.connect(self._display_image)
        self.worker.finished_signal.connect(self._finished)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)

    def _display_image(self, image_bytes):
        if not image_bytes:
            self.log.append("⚠️ No image bytes to display.")
            return
        pixmap = QtGui.QPixmap()
        pixmap.loadFromData(image_bytes)
        self.map_label.setPixmap(pixmap)

    def _finished(self, path):
        self.log.append(f"✅ Map saved: {path}")


# ---------- Run ----------
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = CancerMapGUI()
    gui.show()
    sys.exit(app.exec_())
