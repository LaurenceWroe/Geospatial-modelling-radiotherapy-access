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
from download_worldpop_tif import download_worldpop_tif

class DownloadThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    download_size = 0
    downloaded = 0

    def __init__(self, country_name, output_dir):
        super().__init__()
        self.country_name = country_name
        self.output_dir = output_dir

    def run(self):
        try:
            result = download_worldpop_tif(self.country_name, self.output_dir, self.update_progress)
            if result[0]:
                self.finished.emit(True, result[1])
            else:
                self.finished.emit(False, result[1])
        except Exception as e:
            self.finished.emit(False, str(e))

class WorldPopDownloader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("WorldPop Data Downloader")
        self.setFixedSize(400, 200)

        # Widgets
        self.label = QLabel("Select a country:")
        self.combo = QComboBox()
        self.combo.addItems(sorted([country.name for country in countries]))
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
        self.download_btn.clicked.connect(self.initiate_download)

    def initiate_download(self):
        country = self.combo.currentText()
        if not country:
            QMessageBox.critical(self, "Error", "Please select a country.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Download Directory", "", 
            QFileDialog.ShowDirsOnly
        )
        if not output_dir:
            return

        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.download_btn.setEnabled(False)

        self.thread = DownloadThread(country, output_dir)
        self.thread.update_progress.connect(self.update_progress_bar)
        self.thread.finished.connect(self.download_complete)
        self.thread.start()

    def update_progress_bar(self, value):
        self.progress.setValue(value)

    def download_complete(self, success, message):
        self.progress.setVisible(False)
        self.download_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WorldPopDownloader()
    window.show()
    sys.exit(app.exec_())