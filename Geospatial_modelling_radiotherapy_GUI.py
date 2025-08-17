import sys
import os
import subprocess
import pandas as pd 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog, QHBoxLayout, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from pycountry import countries
from a_population_density.download_worldpop import download_worldpop
from a_population_density.resample_population import resample_population

class ResampleThread(QThread): 
    finished = pyqtSignal(dict)  # Emits the full result dictionary

    def __init__(self, country_name, resolution):
        super().__init__()
        self.country_name = country_name
        self.resolution = resolution

    def run(self):
        result = resample_population(self.country_name, self.resolution)
        self.finished.emit(result)
    

class DownloadThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, country_name, output_dir, overwrite=False):
        super().__init__()
        self.country_name = country_name
        self.output_dir = output_dir
        self.overwrite = overwrite

    def run(self):
        try:
            def progress_callback(progress):
                self.progress_updated.emit(progress)

            success, message = download_worldpop(
                self.country_name,
                self.output_dir,
                progress_callback,
                self.overwrite
            )
            self.finished.emit(success, message)
        except Exception as e:
            self.finished.emit(False, str(e))

class WorldPopDownloader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def load_cancer_types(self, excel_path="b_cancer_incidence/cancer_type_radiotherapy.xlsx"):
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

    def setup_ui(self):
        self.setWindowTitle("WorldPop Data Processor")
        self.setFixedSize(500, 300)

        # Main layout
        main_layout = QVBoxLayout()

        # Download Group
        download_group = QGroupBox("Download Raw Data")
        download_layout = QVBoxLayout()
        
        self.country_label = QLabel("Select a country:")
        self.country_combo = QComboBox()
        self.country_combo.addItems(sorted([country.name for country in countries]))
        
        self.download_btn = QPushButton("Download")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        download_layout.addWidget(self.country_label)
        download_layout.addWidget(self.country_combo)
        download_layout.addWidget(self.download_btn)
        download_layout.addWidget(self.progress)
        download_group.setLayout(download_layout)

        # Resample Group
        resample_group = QGroupBox("Resample Data")
        resample_layout = QVBoxLayout()
        
        self.resolution_label = QLabel("Select resolution (km):")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["0.5", "1.0", "2.0", "5.0", "10.0", "50.0"])
        
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.setEnabled(False) # initially disabled
        self.check_resample_availability() # check if resampling is available
        
        resample_layout.addWidget(self.resolution_label)
        resample_layout.addWidget(self.resolution_combo)
        resample_layout.addWidget(self.resample_btn)
        resample_group.setLayout(resample_layout)

        # --- Cancer Type & Map Generation Group ---
        map_group = QGroupBox("Generate Cancer Type Map")
        map_layout = QVBoxLayout()

        self.cancer_label = QLabel("Select a cancer type:")
        self.cancer_combo = QComboBox()
        cancer_types = self.load_cancer_types()
        self.cancer_combo.addItems(cancer_types)

        self.generate_map_btn = QPushButton("Generate Map")
        self.generate_map_btn.setEnabled(False)  # initially disabled

        map_layout.addWidget(self.cancer_label)
        map_layout.addWidget(self.cancer_combo)
        map_layout.addWidget(self.generate_map_btn)
        map_group.setLayout(map_layout)

        main_layout.addWidget(map_group)

        # Add groups to main layout
        main_layout.addWidget(download_group)
        main_layout.addWidget(resample_group)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Signals
        self.download_btn.clicked.connect(self.initiate_download)
        self.resample_btn.clicked.connect(self.initiate_resample)
        self.country_combo.currentTextChanged.connect(self.check_resample_availability)
        self.generate_map_btn.clicked.connect(self.run_cancer_map_script)

    def check_resample_availability(self):
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

    def initiate_download(self):
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
                overwrite = True
            else:
                overwrite = False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't check file: {str(e)}")
            return

        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.download_btn.setEnabled(False)

        self.download_thread = DownloadThread(country, output_dir, overwrite)
        self.download_thread.progress_updated.connect(self.update_progress_bar)
        self.download_thread.finished.connect(self.download_complete)
        self.download_thread.start()

    def initiate_resample(self):
        country = self.country_combo.currentText()
        resolution = float(self.resolution_combo.currentText())
        
        self.resample_thread = ResampleThread(country, resolution)
        self.resample_thread.finished.connect(self.resample_complete)
        self.resample_thread.start()
        
        self.resample_btn.setEnabled(False)
        QMessageBox.information(self, "Processing", "Resampling in progress...")

    def update_progress_bar(self, value):
        self.progress.setValue(value)

    def download_complete(self, success, message):
        self.progress.setVisible(False)
        self.download_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
            self.check_resample_availability()
        else:
            QMessageBox.critical(self, "Error", message)

    def resample_complete(self, result):
        self.resample_btn.setEnabled(True)
        
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

    def run_cancer_map_script(self):
        country = self.country_combo.currentText()
        cancer_type = self.cancer_combo.currentText()
        resolution = self.resolution_combo.currentText()

        if not (country and cancer_type and resolution):
            QMessageBox.critical(self, "Error", "Missing inputs.")
            return

        try:
            from pycountry import countries
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3
        except:
            QMessageBox.critical(self, "Error", f"Invalid country name: {country}")
            return

        script_path = os.path.abspath("b_cancer_incidence/cancer_type_map.py")

        cmd = [
            sys.executable,  # Path to Python interpreter
            script_path,
            country_code,
            cancer_type,
            "--resolution", resolution
    ]

        try:
            subprocess.run(cmd, check=True)
            QMessageBox.information(self, "Success", f"Map generated for {cancer_type} in {country_code}.")
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Map generation failed:\n{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldPopDownloader()
    window.show()
    sys.exit(app.exec_())