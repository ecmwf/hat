import json

import ipyleaflet
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _compute_bounds(stations_metadata, coord_names):
    """Compute the bounds of the map based on the stations metadata."""

    lon_column = coord_names[0]
    lat_column = coord_names[1]

    lons = stations_metadata[lon_column].values
    lats = stations_metadata[lat_column].values

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    return [(float(min_lat), float(min_lon)), (float(max_lat), float(max_lon))]


class LeafletMap:
    """
    A class for creating interactive leaflet maps.

    Parameters
    ----------
    basemap : ipyleaflet.basemaps, optional
        The basemap to use for the map. Default is
        ipyleaflet.basemaps.OpenStreetMap.Mapnik.

    """

    def __init__(
        self,
        basemap=ipyleaflet.basemaps.OpenStreetMap.Mapnik,
    ):
        self.map = ipyleaflet.Map(
            basemap=basemap, layout=ipywidgets.Layout(width="100%", height="600px")
        )
        self.legend_widget = ipywidgets.Output()

    def _set_boundaries(self, stations_metadata, coord_names):
        """
        Compute the boundaries of the map based on the stations metadata.
        """
        lon_column = coord_names[0]
        lat_column = coord_names[1]

        lons = stations_metadata[lon_column].values
        lats = stations_metadata[lat_column].values

        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        bounds = [(float(min_lat), float(min_lon)), (float(max_lat), float(max_lon))]
        self.map.fit_bounds(bounds)

    def add_geolayer(self, geodata, colormap, widgets, coord_names=None):
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
        geojson.on_click(widgets.update)
        self.map.add(geojson)

        if coord_names is not None:
            self._set_boundaries(geodata, coord_names)

        self.legend_widget = colormap.legend()

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
    A class handling the colormap of a pyleaflet map.

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
    """

    def __init__(
        self,
        config={},
        stats=None,
        colormap_style="viridis",
        range=None,
        empty_color="white",
        default_color="blue",
    ):
        self.config = config
        self.stats = stats
        self.empty_color = empty_color
        self.default_color = default_color
        if self.stats is not None:
            assert (
                "station_id_column_name" in self.config
            ), 'Config must contain "station_id_column_name"'
            # Normalize the data for coloring
            if range is None:
                self.min_val = self.stats.values.min()
                self.max_val = self.stats.values.max()
            else:
                self.min_val = range[0]
                self.max_val = range[1]
        else:
            self.min_val = 0
            self.max_val = 1

        try:
            self.colormap = mpl.colormaps[colormap_style]
        except KeyError:
            raise KeyError(
                f"Colormap {colormap_style} not found. "
                f"Available colormaps are: {mpl.colormaps}"
            )

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
        if self.stats is not None:
            norm = plt.Normalize(self.min_val, self.max_val)

            def map_color(feature):
                station_id = feature["properties"][
                    self.config["station_id_column_name"]
                ]
                if station_id in self.stats.station.values:
                    station_stats = self.stats.sel(station=station_id)
                    color = mpl.colors.rgb2hex(
                        self.colormap(norm(station_stats.values))
                    )
                else:
                    color = self.empty_color
                style = {
                    "color": "black",
                    "fillColor": color,
                }
                return style

        else:

            def map_color(feature):
                return {
                    "color": "black",
                    "fillColor": self.default_color,
                }

        return map_color

    def legend(self):
        """
        Generates an HTML legend for the colormap.

        Returns
        -------
        ipywidgets.HTML
            An HTML widget containing the colormap legend.
        """
        # Convert the colormap to a list of RGB values
        rgb_values = [
            mpl.colors.rgb2hex(self.colormap(i)) for i in np.linspace(0, 1, 256)
        ]

        # Create a gradient style using the RGB values
        gradient_style = ", ".join(rgb_values)
        gradient_html = f"""
        <div style="
            background: linear-gradient(to right, {gradient_style});
            height: 30px;
            width: 200px;
            border: 1px solid black;
        "></div>
        """

        # Create labels
        labels_html = f"""
        <div style="display: flex; justify-content: space-between;">
            <span>Low: {self.min_val:.1f}</span>
            <span>High: {self.max_val:.1f}</span>
        </div>
        """
        # Combine gradient and labels
        legend_html = gradient_html + labels_html

        return ipywidgets.HTML(legend_html)
