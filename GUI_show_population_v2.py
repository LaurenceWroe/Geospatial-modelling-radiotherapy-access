# SIMPLE GUI where user can select a country and download the WorldPop population TIF file for that country.
# Includes a select a country dropdown, a download button, and a progress bar.

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QComboBox, 
    QPushButton, QVBoxLayout, QWidget, QMessageBox,
    QProgressBar, QFileDialog
)
from PyQt5.QtCore import QThread, pyqtSignal
from pycountry import countries
from download_worldpop_tif import download_worldpop_tif  # Import your function

class DownloadThread(QThread):
    """Thread to handle the download without freezing the GUI."""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, country_name, output_dir):
        super().__init__()
        self.country_name = country_name
        self.output_dir = output_dir

    def run(self):
        try:
            success, result = download_worldpop_tif(self.country_name, self.output_dir)
            self.finished_signal.emit(success, result)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class WorldPopDownloader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WorldPop Data Downloader")
        self.setGeometry(100, 100, 400, 200)

        # Widgets
        self.label = QLabel("Select a country:")
        self.combo = QComboBox()
        self.combo.addItems([country.name for country in countries])
        self.download_btn = QPushButton("Download TIF")
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.combo)
        layout.addWidget(self.download_btn)
        layout.addWidget(self.progress)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Signals
        self.download_btn.clicked.connect(self.start_download)

    def start_download(self):
        country = self.combo.currentText()
        if not country:
            QMessageBox.critical(self, "Error", "Please select a country.")
            return

        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Download Directory", "actual_data/raw_from_worldpop"
        )
        if not output_dir:
            return

        self.progress.setVisible(True)
        self.download_btn.setEnabled(False)

        # Start download in a thread
        self.thread = DownloadThread(country, output_dir)
        self.thread.finished_signal.connect(self.on_download_finished)
        self.thread.start()

    def on_download_finished(self, success, result):
        self.progress.setVisible(False)
        self.download_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", f"File saved to:\n{result}")
        else:
            QMessageBox.critical(self, "Error", f"Download failed:\n{result}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldPopDownloader()
    window.show()
    sys.exit(app.exec_())
