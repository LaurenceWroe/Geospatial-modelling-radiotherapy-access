import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog, QHBoxLayout, QGroupBox
)
from PyQt5.QtCore import QThread, pyqtSignal
from pycountry import countries
from download_worldpop import download_worldpop
from resample_population import resample_population

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
        self.resolution_combo.addItems(["0.5", "1", "2", "5", "10", "50"])
        
        self.resample_btn = QPushButton("Resample")
        self.resample_btn.setEnabled(False) # initially disabled
        self.check_resample_availability() # check if resampling is available
        
        resample_layout.addWidget(self.resolution_label)
        resample_layout.addWidget(self.resolution_combo)
        resample_layout.addWidget(self.resample_btn)
        resample_group.setLayout(resample_layout)

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

    def check_resample_availability(self):
        country = self.country_combo.currentText()
        if not country:
            return
            
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            input_file = os.path.join("actual_data/raw_from_worldpop", f"{country_code}_raw.tif")
            self.resample_btn.setEnabled(os.path.exists(input_file))
        except:
            self.resample_btn.setEnabled(False)

    def initiate_download(self):
        country = self.country_combo.currentText()
        if not country:
            QMessageBox.critical(self, "Error", "Please select a country.")
            return

        output_dir = "actual_data/raw_from_worldpop"
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
        else:
            QMessageBox.critical(self, "Error", result['message'])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldPopDownloader()
    window.show()
    sys.exit(app.exec_())