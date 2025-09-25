import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import pycountry
import h3
import requests
import gzip, shutil
from PyQt5 import QtWidgets, QtCore, QtWebEngineWidgets
import pydeck as pdk
import numpy as np

# ----------------------- Utility Functions ----------------------- #

def unzip_gpkg(gz_path):
    gpkg_path = gz_path.with_suffix("")
    with gzip.open(gz_path, 'rb') as f_in, open(gpkg_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gpkg_path


def download_H3_population_density_gpkg(country_name, output_dir="H3_pop_density_maps", overwrite_download=False, progress_callback=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        selected_country = pycountry.countries.get(name=country_name)
        selected_country_alpha_2 = selected_country.alpha_2
    except LookupError:
        return False, None, f"Country {country_name} not found"

    base_url = (
        "https://geodata-eu-central-1-kontur-public.s3.amazonaws.com/kontur_datasets/"
        "kontur_population_{country_alpha_2}_20231101.gpkg.gz"
    )
    target_url = base_url.format(country_alpha_2=selected_country_alpha_2)
    gz_path = Path(output_dir) / f"{selected_country_alpha_2}_H3_population_density_map.gpkg.gz"
    output_file = gz_path.with_suffix("")

    if not overwrite_download and Path(output_file).exists():
        return True, output_file, f"File already exists at {output_file}"

    try:
        with requests.session() as session:
            with session.get(target_url, stream=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(gz_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=10 * 1024):
                        file.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size > 0:
                            progress = int(100 * downloaded / total_size)
                            progress_callback(progress)
    except Exception as e:
        if Path(gz_path).exists():
            Path(gz_path).unlink(missing_ok=True)
        return False, None, f"Download failed: {str(e)}"

    gpkg_path = unzip_gpkg(gz_path)
    Path(gz_path).unlink(missing_ok=True)
    return True, gpkg_path, f"File saved to {gpkg_path}"


def pop_density_calc(df):
    if df.columns.to_list()[0:2] != ["h3", "population"]:
        raise ValueError("Input DataFrame must start with ['h3', 'population']")
    area_km_calc_format = lambda x: h3.cell_area(x, "km^2")
    df["population/km^2"] = df["population"] / (df["h3"].map(area_km_calc_format))
    return df


def to_parent_aggregate_3(df, required_res):
    if required_res >= 8:
        raise ValueError("Resolution must be < 8")
    df_temp = df.copy(deep=True)
    get_res = lambda x: h3.get_resolution(x)
    res = pd.Series(data=df_temp["h3"].map(get_res), name="res")
    if not res.eq(8).all():
        raise ValueError("All input H3 hexes must be res 8")
    to_parent_format = lambda x: h3.cell_to_parent(x, required_res)
    df_temp["h3"] = df["h3"].map(to_parent_format)
    df_temp = df_temp.groupby("h3")["population"].sum().reset_index()
    df_temp = pop_density_calc(df_temp)
    return df_temp


def plot_pop_density_log_opacity(df, country, min_pop_dens=1, max_pop_dens=50000, filename="map.html"):
    lp = np.log(df["population/km^2"].clip(lower=min_pop_dens, upper=max_pop_dens))
    lp_min, lp_max = np.log(min_pop_dens), np.log(max_pop_dens)
    z = (lp - lp_min) / (lp_max - lp_min)
    df_temp = df.copy(deep=True)
    df_temp["alpha"] = (30 + z * (255 - 30)).round().astype(int)
    df_temp["population_dens_2dp"] = df_temp["population/km^2"].map(lambda x: round(x, 2))

    layer = pdk.Layer(
        "H3HexagonLayer",
        df_temp,
        get_hexagon="h3",
        get_fill_color="[0, 122, 255, alpha]",
        stroked=False,
        filled=True,
        extruded=False,
        pickable=True,
        auto_highlight=True,
    )

    country_name = pycountry.countries.lookup(country).name
    # Fallback to geopy for center
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent="h3_gui")
    location = geolocator.geocode(country_name)
    latitude, longitude = location.latitude, location.longitude

    view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=5, bearing=0, pitch=30)
    tooltip = {"html": "Population: {population}<br/>Density: {population_dens_2dp}/km^2"}

    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    r.to_html(filename, open_browser=False)
    return filename

# ----------------------- PyQt5 GUI ----------------------- #

class H3Gui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("H3 Map Generation")
        self.setGeometry(100, 100, 1200, 800)

        layout = QtWidgets.QVBoxLayout(self)

        # Controls
        control_layout = QtWidgets.QHBoxLayout()
        self.country_dropdown = QtWidgets.QComboBox()
        self.country_dropdown.setEditable(True)
        countries = sorted([c.name for c in pycountry.countries])
        self.country_dropdown.addItems(countries)

        self.download_btn = QtWidgets.QPushButton("Download Data")
        self.res_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.res_slider.setMinimum(0)
        self.res_slider.setMaximum(7)
        self.res_slider.setValue(8-1)
        self.res_label = QtWidgets.QLabel("Resolution: 7")
        self.generate_btn = QtWidgets.QPushButton("Generate Map")
        self.progress = QtWidgets.QProgressBar()

        control_layout.addWidget(self.country_dropdown)
        control_layout.addWidget(self.download_btn)
        control_layout.addWidget(self.res_label)
        control_layout.addWidget(self.res_slider)
        control_layout.addWidget(self.generate_btn)

        layout.addLayout(control_layout)
        layout.addWidget(self.progress)

        # Web view
        self.web_view = QtWebEngineWidgets.QWebEngineView()
        layout.addWidget(self.web_view)

        # Events
        self.download_btn.clicked.connect(self.download_data)
        self.res_slider.valueChanged.connect(self.update_res_label)
        self.generate_btn.clicked.connect(self.generate_map)

        # State
        self.df = None
        self.gpkg_file = None

    def update_res_label(self, value):
        self.res_label.setText(f"Resolution: {value}")

    def download_data(self):
        #country = self.country_input.text().strip()
        country = self.country_dropdown.currentText().strip()

        if not country:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter a country name")
            return

        def update_progress(val):
            self.progress.setValue(val)

        success, path, msg = download_H3_population_density_gpkg(
            country, progress_callback=update_progress
        )
        if success:
            self.gpkg_file = path
            gdf = gpd.read_file(path)
            self.df = gdf[["h3", "population"]]
            self.df = pop_density_calc(self.df)
            QtWidgets.QMessageBox.information(self, "Download Complete", msg)
        else:
            QtWidgets.QMessageBox.warning(self, "Error", msg)

    def generate_map(self):
        if self.df is None:
            QtWidgets.QMessageBox.warning(self, "Error", "No data loaded")
            return
        res = self.res_slider.value()
        df_res = to_parent_aggregate_3(self.df, res)
        filename = "map.html"
        #html_file = plot_pop_density_log_opacity(df_res, self.country_input.text(), filename=filename)
        html_file = plot_pop_density_log_opacity(df_res, self.country_dropdown.currentText(), filename=filename)
        self.web_view.load(QtCore.QUrl.fromLocalFile(str(Path(html_file).resolve())))

# ----------------------- Run App ----------------------- #

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = H3Gui()
    gui.show()
    sys.exit(app.exec_())
