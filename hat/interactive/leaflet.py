import glob
import json
import os

import jinja2
import ipyleaflet
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from ipyleaflet import Popup, WidgetControl
from ipywidgets import HTML, Button, HBox, Layout, Text


class LeafletMap:
    """
    A class for creating interactive leaflet maps.

    Parameters
    ----------
    basemap : ipyleaflet.basemaps, optional
        The basemap to use for the map. Default is
        ipyleaflet.basemaps.OpenStreetMap.Mapnik.

    """

    POPUP_MESS_TEMPLATE = """
    <div id="station-popup">
        <h4>{{ index }}</h4>
        <ul>
        {% for property_name, property_value in properties.items() -%}
        <li><b>{{ property_name }}:</b>
            {% if property_value is string %}
                {{ property_value }}
            {% else %}
                {{ property_value | round(0) }}
            {% endif %}
        </li>
        {%- endfor %}
        </ul>
    </div>
    """

    def __init__(self, basemap=ipyleaflet.basemaps.OpenStreetMap.Mapnik, **kwargs):
        self.map = ipyleaflet.Map(
            basemap=basemap,
            layout=ipywidgets.Layout(width="100%", height="600px"),
            **kwargs,
        )
        self.legend_widget = ipywidgets.Output()

    def _set_default_boundaries(self, stations_metadata, coord_names):
        """
        Compute the boundaries of the map based on the stations metadata.
        """
        lon_column = coord_names[0]
        lat_column = coord_names[1]

        lons = stations_metadata[lon_column].values
        lats = stations_metadata[lat_column].values

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        bounds = [
            (float(min_lat), float(min_lon)),
            (float(max_lat), float(max_lon)),
        ]
        self.map.fit_bounds(bounds)

    def _update_boundaries_from_station(self, station_id, metadata, coord_names, station_id_colname="station_id"):
        """
        Compute the boundaries of the map based on the stations metadata.
        """
        lon_column = coord_names[0]
        lat_column = coord_names[1]

        station_id_correct_dtype = metadata[station_id_colname].dtype.type(station_id)

        station_metadata = metadata[metadata[station_id_colname] == station_id_correct_dtype]
        lon = float(station_metadata[lon_column].values[0])
        lat = float(station_metadata[lat_column].values[0])

        shift = 1  # Adjust the shift value as needed
        min_lat = lat - shift
        max_lat = lat + shift
        min_lon = lon - shift
        max_lon = lon + shift

        bounds = [(float(min_lat), float(min_lon)), (float(max_lat), float(max_lon))]
        self.map.fit_bounds(bounds)

    def create_hover(self, widgets, min_zoom=7, property_names=None):
        def hover_handler(feature, **kwargs):
            if self.map.zoom >= min_zoom:
                index = widgets.index(feature, **kwargs)
                # Create a popup with the index as its content
                message = HTML()
                if property_names is not None:
                    items = {prop_name: feature["properties"].get(prop_name, "N/A") for prop_name in property_names}
                    message.value = jinja2.Template(self.POPUP_MESS_TEMPLATE).render(index=index, properties=items)
                else:
                    message.value = str(index)
                self.popup = Popup(
                    location=(
                        feature["geometry"]["coordinates"][1],
                        feature["geometry"]["coordinates"][0],
                    ),
                    child=message,
                    close_button=False,
                    auto_close=True,
                    close_on_escape_key=False,
                )
                # Add the popup to the map
                self.map.add(self.popup)

        return hover_handler

    def add_geolayer(
        self,
        geodata,
        colormap,
        widgets,
        coord_names=None,
        station_id_colname="station_id",
        name="Stations",
        property_names=None,
        # cluster=False
    ):
        """
        Add a geolayer to the map.

        Parameters
        ----------
        geodata : geopandas.GeoDataFrame
            The geodataframe containing the geospatial data.
        colormap : hat.PyleafletColormap
            The colormap to use for the geolayer.
        widgets : hat.WidgetsManager
            The widgets to use for the geolayer.
        coord_names : list of str, optional
            The names of the columns containing the spatial coordinates.
            Default is None.
        """
        geojson = ipyleaflet.GeoJSON(
            name=name,
            data=json.loads(geodata.to_json()),
            style={
                "radius": 7,
                "opacity": 0.5,
                "weight": 1.9,
                "dashArray": "2",
                "fillOpacity": 0.5,
            },
            hover_style={"radius": 10, "fillOpacity": 1},
            point_style={"radius": 5},
            style_callback=colormap.style_callback(),
        )
        # if cluster:
        #     markers = []
        #     for feat in geojson.data["features"]:
        #         marker = ipyleaflet.Marker(location=(feat["geometry"]["coordinates"][1], feat["geometry"]["coordinates"][0]))
        #         mess = HTML()
        #         if property_names is not None:
        #             items = {prop_name: feat["properties"].get(prop_name, "N/A") for prop_name in property_names}
        #             mess.value = self.POPUP_MESS_TEMPLATE.format(index=widgets.index(feat), properties=items)
        #             marker.tooltip = Popup(message=feat["properties"].get("name",""))
        #         markers.append(marker)

        #     SPIDERFY_MAX_ZOOM = 12
        #     marker_cluster = ipyleaflet.MarkerCluster(
        #         markers=markers,
        #         spiderfy_on_max_zoom=True,
        #         disable_clustering_at_zoom=SPIDERFY_MAX_ZOOM,
        #         zoom_to_bounds_on_click=True,
        #     )
        #     self.map.add(marker_cluster)
        #     geojson.visible = False

        #     def handle_zoom(**kwargs):
        #         if kwargs.get("type") == "zoom":
        #             zoom = self.map.zoom
        #         if zoom < SPIDERFY_MAX_ZOOM:
        #             marker_cluster.visible = True
        #             geojson.visible = False
        #         else:
        #             marker_cluster.visible = False
        #             geojson.visible = True

        #     self.map.on_interaction(handle_zoom)

        def update_widgets_from_click(*args, **kwargs):
            widgets.update(*args, **kwargs)

        geojson.on_click(update_widgets_from_click)
        geojson.on_hover(self.create_hover(widgets, property_names=property_names))
        self.map.add(geojson)

        if coord_names is not None:
            self._set_default_boundaries(geodata, coord_names)

        # Add the legend to the map
        legend_control = WidgetControl(widget=colormap.legend(), position="bottomleft")
        self.map.add(legend_control)

        # Add station selector with update button
        text_input = Text(placeholder="ID", description="Station:", disabled=False, layout=Layout(width="200px"))
        text_button = Button(description="Update", layout=Layout(width="100px"))

        # Define the update function
        def update_widgets_from_text(*args, **kwargs):
            station_id = text_input.value
            widgets.update(station_id)
            self._update_boundaries_from_station(station_id, geodata, coord_names, station_id_colname)

        text_button.on_click(update_widgets_from_text)

        # Add the station selector to the map
        widget_container = HBox([text_input, text_button])
        widget_control = WidgetControl(widget=widget_container, position="topright")
        self.map.add(widget_control)

    def output(self, layout={}):
        """
        Return the output widget.

        Parameters
        ----------
        layout : ipywidgets.Layout
            The layout of the widget.

        Returns
        -------
        ipywidgets.VBox
            The output widget.

        """
        output = ipywidgets.VBox([self.map, self.legend_widget], layout=layout)
        return output


class PyleafletColormap:
    """
    A base class handling the colormap of a pyleaflet map.

    Parameters
    ----------
    colormap : matplotlib.colors.Colormap
        The matplotlib colormap object to use for the map.
    """

    def __init__(
        self,
        colormap,
    ):
        self.colormap = colormap

    def legend(self):
        """
        Generates an HTML legend for the colormap.

        Returns
        -------
        ipywidgets.HTML
            An HTML widget containing the colormap legend.
        """
        # Convert the colormap to a list of RGB values
        rgb_values = [mpl.colors.rgb2hex(self.colormap(i)) for i in np.linspace(0, 1, 256)]

        # Create a gradient style using the RGB values
        gradient_style = ", ".join(rgb_values)
        gradient_html = f"""
        <div style="
            background: linear-gradient(to right, {gradient_style});
            height: 30px;
            width: 250px;
            border: 1px solid black;
        "></div>
        """
        if hasattr(self, "min_val"):
            self.vals = [f"{self.min_val:.1f}", f"{self.max_val:.1f}"]
        # Create labels
        labels_html = (
            '<div style="display: flex; justify-content: space-between;">'
            + "".join([f"<span>{x}</span>" for x in self.vals])
            + "</div>"
        )
        # Combine gradient and labels
        legend_html = gradient_html + labels_html

        return ipywidgets.HTML(legend_html)

    def style_callback(self):
        """
        Returns a function that can be used as input style for the ipyleaflet
        layer.

        Returns
        -------
        function
            A function that takes a dataframe feature as input and returns a
            dictionary of style options for the ipyleaflet layer.
        """
        return NotImplementedError


class StatsColormap(PyleafletColormap):
    """
    A class handling the colormap of a pyleaflet map colored by a statistic.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration options for the map.
    stats : xarray.Dataset or None, optional
        A dataset containing the data to be plotted on the map.
        If None, a default constant colormap will be used.
    colormap_style : str, optional
        The name of the matplotlib colormap to use. Default is 'viridis'.
    range : tuple of float, optional
        The minimum and maximum values of the colormap. If None, the
        minimum and maximum values in `stats` will be used.
    empty_color : str, optional
        The color to use for stations that are not in `stats`. Default is
        'white'.
    default_color : str, optional
        The color to use when statistics are not provided. Default is 'blue'.
    """

    def __init__(
        self,
        config={},
        stats=None,
        colormap_style="viridis",
        range=None,
        empty_color="white",
        default_color="blue",
        show_legend=True,
    ):
        self.config = config
        self.stats = stats
        self.empty_color = empty_color
        self.default_color = default_color
        self.show_legend = show_legend
        if self.stats is not None:
            assert "station_id_column_name" in self.config, 'Config must contain "station_id_column_name"'
            # Normalize the data for coloring
            if range is None:
                self.min_val = np.nanmin(self.stats.values)
                self.max_val = np.nanmax(self.stats.values)
            else:
                self.min_val = range[0]
                self.max_val = range[1]
        else:
            self.min_val = 0
            self.max_val = 1

        try:
            colormap = mpl.colormaps[colormap_style]
        except KeyError:
            raise KeyError(f"Colormap {colormap_style} not found. " f"Available colormaps are: {mpl.colormaps}")

        super().__init__(colormap)

    def style_callback(self):
        if self.stats is not None:
            norm = plt.Normalize(self.min_val, self.max_val)

            def map_color(feature):
                station_id = feature["properties"][self.config["station_id_column_name"]]
                if station_id in self.stats.station.values:
                    station_stats = self.stats.sel(station=station_id)
                    color = mpl.colors.rgb2hex(self.colormap(norm(station_stats.values)))
                else:
                    color = self.empty_color
                style = {
                    "color": "black",
                    "fillColor": color,
                }
                return style

        else:

            def map_color(feature):
                style = {
                    "color": "black",
                    "fillColor": self.default_color,
                }
                return style

        return map_color

    def legend(self):
        if self.show_legend:
            return super().legend()
        else:
            return ipywidgets.HTML("")


class PPColormap(PyleafletColormap):
    def __init__(self, config):
        self.station_id_column_name = config["station_id_column_name"]
        date = config["pp"]["date"]
        fc_dir = os.path.join(
            config["pp"]["forecast"],
            date.strftime("%Y%m"),
            f"PPR{date.strftime('%Y%m%d%H')}",
        )

        # Create a custom colormap with two colors: red and blue
        cmap_colors = ["red", "blue"]
        colormap = mpl.colors.ListedColormap(cmap_colors)
        self.min_val = 0
        self.max_val = 1

        degraded_file = os.path.join(fc_dir, "fail_list_*.txt")
        self.degraded_stations = self._get_degraded_stations(degraded_file)

        super().__init__(colormap)

    def _get_degraded_stations(self, files_regex):
        file_paths = glob.glob(files_regex)
        numbers_set = set()

        for file_path in file_paths:
            with open(file_path, "r") as file:
                contents = file.read()
                numbers = [number.strip() for number in contents.split(",") if number.strip() != ""]
                numbers_set.update(numbers)

        return numbers_set

    def style_callback(self):
        def map_style(feature):
            station_id = feature["properties"][self.station_id_column_name]
            if station_id in self.degraded_stations:
                color = "red"
            else:
                color = "green"
            style = {
                "color": "black",
                "fillColor": color,
            }
            return style

        return map_style


class ReportingPointsColormap(PyleafletColormap):
    """
    A class handling the colormap of a pyleaflet map colored by a statistic.

    Parameters
    ----------
    config : dict
        A dictionary containing configuration options for the map.
    stats : xarray.Dataset or None, optional
        A dataset containing the data to be plotted on the map.
        If None, a default constant colormap will be used.
    colormap_style : str, optional
        The name of the matplotlib colormap to use. Default is 'viridis'.
    range : tuple of float, optional
        The minimum and maximum values of the colormap. If None, the
        minimum and maximum values in `stats` will be used.
    empty_color : str, optional
        The color to use for stations that are not in `stats`. Default is
        'white'.
    default_color : str, optional
        The color to use when statistics are not provided. Default is 'blue'.
    """

    def __init__(
        self,
        config={},
        empty_color="black",
    ):
        self.config = config
        self.empty_color = empty_color

        cmap_colors = ["black", "gray", "yellow", "red", "purple"]
        colormap = mpl.colors.ListedColormap(cmap_colors)

        self.vals = ["Invalid", "Inactive", "Medium", "High", "Extreme"]

        super().__init__(colormap)

    def style_callback(self):
        def map_color(feature):
            station_id = feature["properties"][self.config["station_id_column_name"]]
            if station_id[1] == "I":  # inactive
                color = "gray"
            elif station_id[1] == "M":  # medium
                color = "yellow"
            elif station_id[1] == "H":  # high
                color = "red"
            elif station_id[1] == "E":  # extreme
                color = "purple"
            else:
                color = self.empty_color
            style = {
                "color": "black",
                "fillColor": color,
            }
            return style

        return map_color
