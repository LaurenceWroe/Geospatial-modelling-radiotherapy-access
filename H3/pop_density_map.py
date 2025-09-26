#imports 
import pydeck as pdk
import pandas as pd
import numpy as np
import h3
import io
import requests
import geodatasets as gds
import geopandas as gpd
import numpy as np
import pycountry
from geopy.geocoders import Nominatim
from pathlib import Path
import gzip, shutil

# --- Helpers --- 
def get_country_coords(country_name):
    geolocator = Nominatim(user_agent="GeospatialCancerAccess2")
    try:
        location = geolocator.geocode(country_name)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- Population Density Calculation --- 
def pop_density_calc(df):
    """
    function takes a dataframe with H3 hexagons and populations, and adds a new column "population/km^2" 
    that gives the population density per square kilometer for each hex.
    """
    # First should probs check that it is the right dataframe input
    if df.columns.to_list()[0:2] != ["h3", "population"]:
        raise ValueError("Nu uh uh, wrong dataframe initial two columns name, should be ['h3', 'population', ...]")
    
    # To be honest, probs wont trip the above, but only problems will be if there's an alpha channel already or a population density already produced
    # Should probs add more error tripping
    
    area_km_calc_format = lambda x: h3.cell_area(x,"km^2")
    df["population/km^2"] = df["population"] / (df["h3"].map(area_km_calc_format))
    return df


# --- Plotting Population Density Map --- 
def plot_pop_density_log_opacity(
        df, 
        country,
        min_pop_dens=1, 
        max_pop_dens=50000,
        h3_column="h3",
        pickable=True,
        stroked=False,
        filled=True,
        extruded=False,
        get_elevation="population/km^2",
        elevation_scale=20,
        high_precision=True,
        auto_highlight=True,
        pitch=30,
        bearing=0,
        map_provider="carto",
        map_style="light", # road, dark, satellite, dark_no_labels, light_no_labels,
        map_output_folder="H3_pydeck_maps",
        map_name="log_opacity_map",
        open_browser=True,
        ):
    """ 
    function creates an interactive web map of H3 hexagons colored by population density, where opacity reflects density on a log scale.
    """

    lp = np.log(df["population/km^2"].clip(lower=min_pop_dens, upper=max_pop_dens))
    lp_min, lp_max = np.log(min_pop_dens), np.log(max_pop_dens)

    z = (lp - lp_min) / (lp_max - lp_min) # in [0, 1]

    df_temp= df.copy(deep=True)

    df_temp["alpha"] = (30 + z * (255 - 30)).round().astype(int)
    # Purely for formatting and displaying in the tooltip
    df_temp["population_dens_2dp"] = df_temp["population/km^2"].map(lambda x: round(x,2))

    layer = pdk.Layer(
        "H3HexagonLayer",
        df_temp,
        pickable=pickable,
        stroked=stroked,
        filled=filled,
        extruded=extruded,
        get_elevation=get_elevation,
        elevation_scale=elevation_scale,
        high_precision=high_precision,
        get_hexagon=h3_column,
        auto_highlight=auto_highlight,
        get_fill_color="[0, 122, 255, alpha]",  # data-driven transparency
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )

    country_name = pycountry.countries.lookup(country).name
    latitude, longitude = get_country_coords(country_name)

    view_state = pdk.ViewState(latitude=latitude, longitude=longitude, zoom=7, bearing=bearing, pitch=pitch)

    tooltip = {
        "html": "Population: {population},<br/>Population density: {population_dens_2dp}/km^2",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }


    r = pdk.Deck(layers=[layer], 
                initial_view_state=view_state,
                map_provider=map_provider,
                map_style=map_style,  # road, dark, satellite, dark_no_labels, light_no_labels
                tooltip=tooltip)

    Path(map_output_folder).mkdir(parents=True, exist_ok=True)
    
    filename = Path(map_output_folder) / Path(map_name + '.html')

    r.to_html(filename=filename, open_browser=open_browser, notebook_display=False)