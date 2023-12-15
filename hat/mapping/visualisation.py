import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ipywidgets as widgets
from ipyleaflet import Map, GeoJSON, LayersControl, Popup, basemaps
import IPython.display as display
from ipywidgets import Layout


class GeoJSONLayerManager:
    def __init__(self, path, style_callback=None, point_style=None, name="Layer"):
        self.gdf = gpd.read_file(path)
        self.style_callback = style_callback
        self.point_style = point_style
        self.name = name
        self.layer = None

    def add_to_map(self, map_object):
        if self.style_callback:
            self.layer = GeoJSON(data=self.gdf.__geo_interface__, style_callback=self.style_callback, name=self.name)
        elif self.point_style:
            self.layer = GeoJSON(data=self.gdf.__geo_interface__, point_style=self.point_style, name=self.name)
        else:
            self.layer = GeoJSON(data=self.gdf.__geo_interface__, name=self.name)
        map_object.add_layer(self.layer)

class InteractiveMap:
    def __init__(self, center=(0, 0), zoom=2):
        self.map = Map(basemap=basemaps.CartoDB.Positron, center=center, zoom=zoom, layout=Layout(height='600px'))
        self.map.add_control(LayersControl())

    def add_layer(self, layer_manager):
        layer_manager.add_to_map(self.map)

    def show_map(self):  # Renamed from 'display' to 'show_map'
        display(self.map)

# Util functions
def create_gradient_legend(cmap, min_val, max_val):
    """Generates an HTML legend for the colormap with a label."""
    rgb_values = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, 256)]
    gradient_style = ", ".join(rgb_values)
    legend_html = f"""
    <div style="text-align: center; font-weight: bold;">Area Difference (%)</div>
    <div style="display: flex; flex-direction: column; align-items: center;">
        <div style="
            background: linear-gradient(to right, {gradient_style});
            height: 30px;
            width: 200px;
            border: 1px solid black;
        "></div>
        <div style="text-align: center; width: 100%;">
            <span>Low: {min_val:.1f}%</span> | <span>High: {max_val:.1f}%</span>
        </div>
    </div>
    """
    return widgets.HTML(legend_html)

def create_polygon_legend(colors, labels):
    """Creates an HTML legend for polygon colors."""
    items = []
    for color, label in zip(colors, labels):
        item_html = f"""
        <div>
            <span style='height:10px;width:10px;background-color:{color};display:inline-block;'></span>
            {label}
        </div>
        """
        items.append(item_html)
    legend_html = "<div style='padding:10px;background-color:white;opacity:0.8;'>" + "".join(items) + "</div>"
    return widgets.HTML(legend_html)

def make_line_click_handler(near_area_attr, new_area_attr, distance_attr, map_object):
    def line_click_handler(feature, **kwargs):
        near_area_diff = feature['properties'].get(near_area_attr, 'N/A')
        new_area_diff = feature['properties'].get(new_area_attr, 'N/A')
        distance_km = feature['properties'].get(distance_attr, 'N/A')

        # Format the popup message
        message_html = f"""
        <div>Near Area Difference (%): {near_area_diff:.1f}</div>
        <div>New Area Difference (%): {new_area_diff:.1f}</div>
        <div>Distance (km): {distance_km:.1f}</div>
        """
        message = widgets.HTML(message_html)

        # Extract latitude and longitude for the popup
        coords = feature['geometry']['coordinates'][0]
        latlng = (coords[1], coords[0])  # Convert to (lat, lon) format
        popup = Popup(location=latlng, child=message, close_button=True, auto_close=True, close_on_escape_key=True)
        map_object.add_layer(popup)

    return line_click_handler

def make_style_callback(attribute, cmap, norm):
    def style_callback(feature):
        """Style function for the GeoJSON layer based on a given attribute."""
        area_diff = feature['properties'][attribute]
        color = mcolors.to_hex(cmap(norm(area_diff))) if area_diff is not None else "#ffffff"
        return {'color': color, 'weight': 4}
    return style_callback

def vector_style(feature, color, opacity):
    return {
        'fillColor': color,   # Fill color
        'color': color,       # Border color
        'weight': 1,          # Border width
        'fillOpacity': opacity   # Opacity of the fill
    }

