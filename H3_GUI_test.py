import sys
from pathlib import Path
import re
import pycountry
import xarray as xr
from PyQt5 import QtWidgets, QtCore, QtGui

from H3.download_h3_test import download_H3_population_density_zipped
from H3.resample_h3_population_test import generate_population_density_map_only_h3
from H3.generate_cancer_type_map_H3_test import generate_cancer_type_map_h3

# ---------- Helpers ----------
def get_available_countries():
    """Return list of all ISO-recognized country names."""
    return sorted([c.name for c in pycountry.countries])

def get_cancer_types_from_tensor(xarray_path="b_cancer_incidence/globocan_xarray.nc"):
    da = xr.open_dataarray(xarray_path)
    if "Cancer" not in da.coords:
        raise ValueError("Tensor missing 'Cancer' coordinate.")
    return list(da.coords["Cancer"].values)

def get_country_gpkg_path(country_name, input_dir="H3_zipped_pop_density_maps"):
    alpha2 = pycountry.countries.lookup(country_name).alpha_2.upper()
    fname = f"{alpha2}_population_density.gpkg.gz"
    return Path(input_dir) / fname

# ---------- Worker Thread ----------
class MapWorker(QtCore.QThread):
    log_signal = QtCore.pyqtSignal(str)
    image_signal = QtCore.pyqtSignal(bytes)
    finished_signal = QtCore.pyqtSignal(str)

    def __init__(self, country, cancers, linac_capacity, map_type):
        super().__init__()
        self.country = country
        self.cancers = cancers
        self.linac_capacity = linac_capacity
        self.map_type = map_type

    def run(self):
        try:
            gpkg_path = get_country_gpkg_path(self.country)
            alpha2 = pycountry.countries.lookup(self.country).alpha_2

            # Download if missing
            if not gpkg_path.exists():
                self.log_signal.emit(f"🌍 Downloading population data for {self.country}...")
                download_H3_population_density_zipped(alpha2)
                self.log_signal.emit("✅ Download complete.")
            else:
                self.log_signal.emit(f"Population data exists, using cached file.")

            # Generate population map if requested
            if self.map_type == "Population Density":
                self.log_signal.emit(f"▶ Generating population density map for {self.country}...")
                pop_gpkg, pop_png = generate_population_density_map_only_h3(self.country)
                with open(pop_png, "rb") as f:
                    self.image_signal.emit(f.read())
                self.finished_signal.emit(pop_png)
                return

            # Otherwise generate cancer map
            self.log_signal.emit(f"▶ Generating cancer map for {self.country} ({self.map_type})...")
            da = xr.open_dataarray("b_cancer_incidence/globocan_xarray.nc")
            result = generate_cancer_type_map_h3(
                country_iso3=pycountry.countries.lookup(self.country).alpha_3,
                cancer_types=self.cancers,
                da=da,
                include_RT_utilisation=True,
                include_optimal_RT_utilisation=True,
                include_capacity_weighted="capacity-weighted" in self.map_type.lower(),
                linac_capacity=self.linac_capacity,
                n_linacs=5,
            )
            # Send image bytes
            self.image_signal.emit(result["image_bytes"])
            self.finished_signal.emit(result["gpkg_path"])
        except Exception as e:
            self.log_signal.emit(f"❌ Error: {e}")

# ---------- GUI ----------
class CancerMapGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cancer Map Generator (H3)")
        self.resize(800, 700)

        self.countries = get_available_countries()
        self.cancers = get_cancer_types_from_tensor()

        layout = QtWidgets.QVBoxLayout(self)

        # Country dropdown
        self.country_combo = QtWidgets.QComboBox()
        self.country_combo.addItems(self.countries)
        layout.addWidget(QtWidgets.QLabel("Select country:"))
        layout.addWidget(self.country_combo)

        # Cancer checklist
        layout.addWidget(QtWidgets.QLabel("Select cancer types:"))
        self.cancer_list = QtWidgets.QListWidget()
        self.cancer_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for c in self.cancers:
            item = QtWidgets.QListWidgetItem(c)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.cancer_list.addItem(item)
        layout.addWidget(self.cancer_list)

        # Linac capacity slider + spinbox
        layout.addWidget(QtWidgets.QLabel("Linac capacity (patients/year):"))
        self.capacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.capacity_slider.setRange(50, 1000)
        self.capacity_slider.setValue(250)
        self.capacity_spin = QtWidgets.QSpinBox()
        self.capacity_spin.setRange(50, 1000)
        self.capacity_spin.setValue(250)

        cap_layout = QtWidgets.QHBoxLayout()
        cap_layout.addWidget(self.capacity_slider)
        cap_layout.addWidget(self.capacity_spin)
        layout.addLayout(cap_layout)

        self.capacity_slider.valueChanged.connect(self.capacity_spin.setValue)
        self.capacity_spin.valueChanged.connect(self.capacity_slider.setValue)

        # Map type dropdown
        layout.addWidget(QtWidgets.QLabel("Select map to generate:"))
        self.map_type_combo = QtWidgets.QComboBox()
        self.map_type_combo.addItems([
            "Cancer Incidence", "Treated by Radiotherapy",
            "Optimally Treated by Radiotherapy", "Population Density"
        ])
        layout.addWidget(self.map_type_combo)

        # Generate button
        self.run_btn = QtWidgets.QPushButton("Generate Map")
        layout.addWidget(self.run_btn)

        # Map display
        self.map_label = QtWidgets.QLabel()
        self.map_label.setFixedSize(700, 500)
        self.map_label.setScaledContents(True)
        layout.addWidget(self.map_label)

        # Log
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Log:"))
        layout.addWidget(self.log)

        self.run_btn.clicked.connect(self.run_maps)

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

        self.worker = MapWorker(country, cancers, linac_capacity, map_type)
        self.worker.log_signal.connect(self._append_log)
        self.worker.image_signal.connect(self._display_image)
        self.worker.finished_signal.connect(self._finished)
        self.worker.start()

    def _append_log(self, msg):
        self.log.append(msg)

    def _display_image(self, image_bytes):
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
