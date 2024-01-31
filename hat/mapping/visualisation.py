import geopandas as gpd
import IPython.display as display
import ipywidgets as widgets
import matplotlib.colors as mcolors
import numpy as np
from ipyleaflet import (
    GeoJSON,
    LayersControl,
    Map,
    Marker,
    Popup,
    SearchControl,
    basemaps,
)
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
            self.layer = GeoJSON(
                data=self.gdf.__geo_interface__,
                style_callback=self.style_callback,
                name=self.name,
            )
        elif self.point_style:
            self.layer = GeoJSON(
                data=self.gdf.__geo_interface__,
                point_style=self.point_style,
                name=self.name,
            )
        else:
            self.layer = GeoJSON(data=self.gdf.__geo_interface__, name=self.name)
        map_object.add_layer(self.layer)


class InteractiveMap:
    def __init__(self, center=(0, 0), zoom=2):
        self.map = Map(
            basemap=basemaps.Esri.WorldImagery,
            center=center,
            zoom=zoom,
            layout=Layout(height="600px"),
            scroll_wheel_zoom=False,
        )
        self.map.add_control(LayersControl())

        search = SearchControl(
            position="topleft",
            url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
            zoom=12,
            marker=Marker(),
        )
        self.map.add_control(search)

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
    <div style="text-align: center; font-weight: bold;">Optimum Grid Area Difference (%)</div>
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
            <span style=
            'height:10px;width:10px;background-color:{color};display:inline-block;'>
            </span>
            {label}
        </div>
        """
        items.append(item_html)
    legend_html = (
        "<div style='padding:10px;background-color:white;opacity:0.8;'>"
        + "".join(items)
        + "</div>"
    )
    return widgets.HTML(legend_html)


def make_line_click_handler(
    station_name_attr,
    station_area_attr,
    near_area_attr,
    optimum_area_attr,
    optimum_dist_attr,
    map_object,
):
    def line_click_handler(feature, **kwargs):
        station_area = feature["properties"].get(station_area_attr, "N/A")
        near_area = feature["properties"].get(near_area_attr, "N/A")
        optimum_area = feature["properties"].get(optimum_area_attr, "N/A")
        optimum_distance_cells = feature["properties"].get(optimum_dist_attr, "N/A")

        # Format numbers with comma separators
        station_area = f"{station_area:,.1f}" if station_area != "N/A" else station_area
        near_area = f"{near_area:,.1f}" if near_area != "N/A" else near_area
        optimum_area = f"{optimum_area:,.1f}" if optimum_area != "N/A" else optimum_area
        optimum_distance_cells = (
            f"{optimum_distance_cells:,.1f}"
            if optimum_distance_cells != "N/A"
            else optimum_distance_cells
        )

        # Format the popup message with HTML
        message_html = f"""
        <table class="tg">
        <thead>
        <tr>
            <th class="tg-1wig">Station Name</th>
            <th class="tg-1wig">{feature['properties'][station_name_attr]}</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td class="tg-0lax">Station Ups. Area</td>
            <td class="tg-0lax">{station_area}km<sup>2</sup></td>
        </tr>
        <tr>
            <td class="tg-0lax">Nearest-Grid Ups. Area</td>
            <td class="tg-0lax">{near_area}km<sup>2</sup></td>
        </tr>
        <tr>
            <td class="tg-0lax"><b>Optimum-Grid</b> Ups. Area</td>
            <td class="tg-0lax">{optimum_area}km<sup>2</sup></td>
        </tr>
        <tr>
            <td class="tg-0lax"><b>Optimum-Grid</b> Distance</td>
            <td class="tg-0lax">{optimum_distance_cells} cells</td>
        </tr>
        </tbody>
        </table>
        """
        message = widgets.HTML(message_html)

        # Extract latitude and longitude for the popup
        coords = feature["geometry"]["coordinates"][0]
        latlng = (coords[1], coords[0])  # Convert to (lat, lon) format
        popup = Popup(
            location=latlng,
            child=message,
            close_button=True,
            auto_close=True,
            close_on_escape_key=True,
        )
        map_object.add_layer(popup)

    return line_click_handler


def make_style_callback(attribute, cmap, norm):
    def style_callback(feature):
        """Style function for the GeoJSON layer based on a given attribute."""
        area_diff = feature["properties"][attribute]
        color = (
            mcolors.to_hex(cmap(norm(area_diff)))
            if area_diff is not None
            else "#ffffff"
        )
        return {"color": color, "weight": 4}

    return style_callback


def vector_style(feature, color, opacity=0.5, weight=1):
    return {
        "fillColor": color,  # Fill color
        "color": color,  # Border color
        "weight": weight,  # Border width
        "fillOpacity": opacity,  # Opacity of the fill
    }


def attribute_based_style(row, attribute_name, threshold, color_above, color_below):
    """Style function for GeoJSON features based on an attribute value."""
    attribute_value = row.get(attribute_name)
    if attribute_value is not None:
        color = color_above if attribute_value > threshold else color_below
        return {
            "radius": 5,
            "color": color,
            "fillColor": color,
            "fillOpacity": 0.5,
        }
    else:
        return {
            "radius": 5,
            "color": "gray",
            "fillColor": "gray",
            "fillOpacity": 0.5,
        }
