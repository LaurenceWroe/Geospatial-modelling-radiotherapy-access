import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QPushButton, QGroupBox, QFileDialog)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np
from pathlib import Path

# Import your existing functions (make sure they're in the same directory or adjust the import)
from resample_population import resample_country_population

class PopulationDensityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Population Density Visualizer")
        self.setGeometry(100, 100, 1000, 800)
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout(self.main_widget)
        
        # Control panel (left side)
        self.control_panel = QGroupBox("Controls")
        self.control_layout = QVBoxLayout()
        
        # Country selection
        self.country_label = QLabel("Select Country:")
        self.country_combo = QComboBox()
        self.country_combo.addItems(["Great Britain (GBR)", "Nigeria (NGA)"])
        
        # Resolution selection
        self.resolution_label = QLabel("Select Resolution (km):")
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["0.5", "1.0", "2.0", "5.0", "10.0"])
        
        # File selection
        self.file_button = QPushButton("Select Population File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        
        # Process button
        self.process_button = QPushButton("Process and Visualize")
        self.process_button.clicked.connect(self.process_data)
        
        # Add widgets to control panel
        self.control_layout.addWidget(self.country_label)
        self.control_layout.addWidget(self.country_combo)
        self.control_layout.addWidget(self.resolution_label)
        self.control_layout.addWidget(self.resolution_combo)
        self.control_layout.addWidget(self.file_button)
        self.control_layout.addWidget(self.file_label)
        self.control_layout.addWidget(self.process_button)
        self.control_layout.addStretch()
        self.control_panel.setLayout(self.control_layout)
        
        # Visualization panel (right side)
        self.viz_panel = QGroupBox("Population Density Visualization")
        self.viz_layout = QVBoxLayout()
        
        # Matplotlib figure
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.viz_layout.addWidget(self.canvas)
        self.viz_panel.setLayout(self.viz_layout)
        
        # Add panels to main layout
        self.layout.addWidget(self.control_panel, stretch=1)
        self.layout.addWidget(self.viz_panel, stretch=3)
        
        # Initialize variables
        self.selected_file = ""
        self.output_file = ""
        
    def select_file(self):
        """Open file dialog to select population file"""
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Select Population File", 
                                            "", "GeoTIFF Files (*.tif *.tiff);;All Files (*)", 
                                            options=options)
        if file:
            self.selected_file = file
            self.file_label.setText(f"Selected: {Path(file).name}")
    
    def process_data(self):
        """Process the selected data and update visualization"""
        # Get user selections
        country = "GBR" if self.country_combo.currentText().startswith("Great Britain") else "NGA"
        resolution = float(self.resolution_combo.currentText())
        
        # Use default file if none selected
        if not self.selected_file:
            default_files = {
                'GBR': 'data/raw/gbr_pd_2020_1km.tif',
                'NGA': 'data/raw/nga_pd_2020_1km_UNadj.tif'
            }
            self.selected_file = default_files.get(country, "")
            if not Path(self.selected_file).exists():
                self.file_label.setText("Error: Default file not found. Please select a file.")
                return
        
        # Set output path
        output_dir = Path("data/raw/resampled")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = str(output_dir / f"{country.lower()}_population_{resolution}km.tif")
        
        try:
            # Process the data using your existing function
            results = resample_country_population(
                country_code=country,
                raw_population_file=self.selected_file,
                output_file=self.output_file,
                target_resolution_km=resolution
            )
            
            # Update the visualization
            self.update_visualization(self.output_file)
            
        except Exception as e:
            self.file_label.setText(f"Error: {str(e)}")
    
    def update_visualization(self, tif_file):
        """Update the matplotlib visualization with the new data"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        try:
            with rasterio.open(tif_file) as src:
                data = src.read(1)
                # Replace nodata values with NaN for better visualization
                data[data == src.nodata] = np.nan
                
                # Create visualization
                im = ax.imshow(data, cmap='viridis')
                self.figure.colorbar(im, ax=ax, label='Population per km²')
                
                # Add title with country and resolution
                country = "GBR" if "gbr" in tif_file.lower() else "NGA"
                res = tif_file.split('_')[-1].replace('km.tif', '')
                ax.set_title(f"Population Density: {country} at {res} km resolution")
                
                # Remove axis ticks for cleaner look
                ax.set_xticks([])
                ax.set_yticks([])
                
                self.canvas.draw()
                
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading file:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PopulationDensityApp()
    window.show()
    sys.exit(app.exec_())