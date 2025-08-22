import sys
import os
import subprocess
import pandas as pd 
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog, QHBoxLayout, QGroupBox, QSplitter
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QScrollArea, QTextEdit
import io
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from pycountry import countries
from a_population_density.download_worldpop import download_worldpop
from a_population_density.resample_population import resample_population
from b_cancer_incidence.generate_cancer_type_map import generate_cancer_type_map


# All threads below for resampling, downloading and mapping:

class ResampleThread(QThread): 
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


class MapGenerationThread(QThread):
    finished = pyqtSignal(bytes, str, str)  # image_data, tif_path, png_path
    error = pyqtSignal(str)

    def __init__(self, country_code, cancer_type, resolution, overwrite_cancer_type_map=False):
        super().__init__()
        self.country_code = country_code
        self.cancer_type = cancer_type
        self.resolution = resolution
        self.overwrite_cancer_type_map = overwrite_cancer_type_map

    def run(self):
        try:
            # Set matplotlib to use Agg backend in the thread, otherwise GUI crashes
            import matplotlib
            matplotlib.use('Agg')
            from b_cancer_incidence.generate_cancer_type_map import generate_cancer_type_map
            
            image_data, tif_path, png_path = generate_cancer_type_map(
                country_code=self.country_code,
                cancer_type=self.cancer_type,
                resolution=self.resolution,
                return_image=True,
                overwrite_cancer_type_map=self.overwrite_cancer_type_map
            )
            self.finished.emit(image_data, tif_path, png_path)
        except Exception as e:
            self.error.emit(str(e))





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
        self.setWindowTitle("Geospatial Modelling of Radiotherapy Access")
        self.setFixedSize(1200, 800)


        # Main splitter
        splitter = QSplitter(Qt.Horizontal)
    
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()

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
        self.check_resample_availability() # check if resampling is available, if so enable the button
        
        resample_layout.addWidget(self.resolution_label)
        resample_layout.addWidget(self.resolution_combo)
        resample_layout.addWidget(self.resample_btn)
        resample_group.setLayout(resample_layout)

        # Cancer Type & Map Generation Group
        map_group = QGroupBox("Generate Cancer Type Map")
        map_layout = QVBoxLayout()

        self.cancer_label = QLabel("Select a cancer type:")
        self.cancer_combo = QComboBox()
        cancer_types = self.load_cancer_types()
        self.cancer_combo.addItems(cancer_types)

        self.generate_map_btn = QPushButton("Generate Map")
        self.generate_map_btn.setEnabled(False)  # initially disabled
        self.check_cancer_map_availability() # check if cancer map generation is available, if so enable the button
        
        map_layout.addWidget(self.cancer_label)
        map_layout.addWidget(self.cancer_combo)
        map_layout.addWidget(self.generate_map_btn)
        map_group.setLayout(map_layout)


        # Add groups to left layout
        left_layout.addWidget(download_group)
        left_layout.addWidget(resample_group)
        left_layout.addWidget(map_group)
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)

        # Setting size of left panel
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(450)
        left_panel.setMinimumWidth(350)

        # Right panel for image display
        right_panel = QWidget()
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
        
        # Splitter add panels
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        container = QWidget()
        container_layout = QHBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)
        self.setCentralWidget(container)

        # Signals
        self.download_btn.clicked.connect(self.initiate_download)
        self.resample_btn.clicked.connect(self.initiate_resample)
        self.generate_map_btn.clicked.connect(self.initiate_cancer_type_map_generate)

        self.country_combo.currentTextChanged.connect(self.check_resample_availability) # Whenever the country changes, check if there exists a downloaded raw file for resampling
        self.country_combo.currentTextChanged.connect(self.check_cancer_map_availability) # Whenever the country changes, check if there exists a resampled file for map generation
        self.resolution_combo.currentTextChanged.connect(self.check_cancer_map_availability) # Whenever the resolution changes, check if there exists a resampled file for map generation


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

    def check_cancer_map_availability(self): # Check whether resampled file exists for map generation, if so enable the cancer map generate button
        country = self.country_combo.currentText()
        resolution = float(self.resolution_combo.currentText())
        if not country:
            return
        if not resolution:
            return
            
        try:
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3.lower()
            input_file = os.path.join("a_population_density/resampled", f"{country_code}_{resolution}km.tif")
            self.generate_map_btn.setEnabled(os.path.exists(input_file))
        except:
            self.generate_map_btn.setEnabled(False)

    def initiate_download(self): # Called when download button is clicked
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

    def initiate_resample(self): # Called when resample button is clicked
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
        QMessageBox.information(self, "Processing", "Resampling in progress...")

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



    def initiate_cancer_type_map_generate(self): # Called when generate map button is clicked
        country = self.country_combo.currentText()
        cancer_type = self.cancer_combo.currentText()
        resolution = float(self.resolution_combo.currentText())

        output_dir = "b_cancer_incidence/cancer_type_maps"
        if not output_dir:
            return

        if not (country and cancer_type and resolution):
            QMessageBox.critical(self, "Error", "Missing inputs.")
            return

        try:
            from pycountry import countries
            country_obj = countries.lookup(country)
            country_code = country_obj.alpha_3
            safe_cancer = cancer_type.replace(" ", "_") # replace spaces with underscores for naming
            # Check if file exists already
            target_file = os.path.join(output_dir, f"{country_code.lower()}_{safe_cancer.lower()}_{resolution}km_cancer_type_density.png")
            print(target_file)

            if os.path.exists(target_file):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"File already exists at:\n{target_file}\n\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                overwrite_cancer_type_map = True
            else:
                overwrite_cancer_type_map = False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Couldn't check file: {str(e)}")
            return


        self.update_status(f"Generating map for {cancer_type} in {country_code}...")
        self.generate_map_btn.setEnabled(False)  # Disable button during generation
        
        # Create and start the thread
        self.map_thread = MapGenerationThread(country_code, cancer_type, resolution, overwrite_cancer_type_map)
        self.map_thread.finished.connect(self.cancer_type_map_completed)
        self.map_thread.error.connect(self.on_map_generation_error)
        self.map_thread.start()


    def update_status(self, message):
        """Update status text area"""
        self.status_text.append(message)

    def display_image(self, image_data):
        """Display image data in the image label"""
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
        self.generate_map_btn.setEnabled(True)
        if image_data:
            self.display_image(image_data)
            self.update_status(f"Map generated successfully!\nTIFF: {tif_path}\nPNG: {png_path}")
        else:
            self.update_status("Map generated but no image data returned.")


    def on_map_generation_error(self, error_msg):
        self.generate_map_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Map generation failed:\n{error_msg}")
        self.update_status(f"Error: {error_msg}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldPopDownloader()
    window.show()
    sys.exit(app.exec_())