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


def to_parent_aggregate_3(df, required_res):
    """
    function rolls up res-8 population data into larger H3 hexagons at a coarser resolution. 
    It sums the populations of all child cells inside each parent, and then computes the new 
    population density for those bigger hexes.
    """
    if required_res >= 8:
        raise ValueError("why are u doing that. The maximum resolution data we have is 8 so this is silly. Choose < 8")
    df_temp = df.copy(deep=True)

    # Check if all h3 res 8 cells
    get_res = lambda x: h3.get_resolution(x)
    res = pd.Series(data=df_temp["h3"].map(get_res),name="res")
    
    if not res.eq(8).all():
        raise ValueError(f"Wrong resolution for one/some input H3 hexes, supposed to be res. 8")
    
    to_parent_format = lambda x: h3.cell_to_parent(x, required_res)
    df_temp["h3"] = df["h3"].map(to_parent_format)
    df_temp = df_temp.groupby("h3")["population"].sum().reset_index()
    df_temp = pop_density_calc(df_temp)
    return df_temp