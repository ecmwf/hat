import numpy as np

from hat.station_mapping import metrics


class StationMapping:
    def __init__(self, config):
        self.max_search_distance = config.get("max_search_distance", 5)
        self.metric_error_func = getattr(metrics, config.get("metric_error_func", "mape"))
        self.distance_error_func = getattr(metrics, config.get("distance_error_func", "no_error"))
        self.lambd = config.get("lambda", 0)
        self.max_error = config.get("max_error", np.inf)
        self.min_error = config.get("min_error", 0)

    def conduct_mapping(
        self,
        station_coords1,
        station_coords2,
        grid_area_coords1,
        grid_area_coords2,
        station_metric=None,
        grid_metric=None,
    ):
        num_stations = len(station_coords1)

        indxs = np.empty(num_stations, dtype=int)
        indys = np.empty(num_stations, dtype=int)
        closest_indxs = np.empty(num_stations, dtype=int)
        closest_indys = np.empty(num_stations, dtype=int)
        errors = np.empty(num_stations, dtype=float)

        for i in range(num_stations):
            station_x, station_y = station_coords1[i], station_coords2[i]

            # get all grid cells within max_search_distance
            closest_idx = np.nanargmin(np.abs(grid_area_coords1[:, 0] - station_x))
            closest_idy = np.nanargmin(np.abs(grid_area_coords2[0, :] - station_y))

            searchbox_min_x = closest_idx - self.max_search_distance
            searchbox_max_x = closest_idx + self.max_search_distance + 1
            searchbox_min_y = closest_idy - self.max_search_distance
            searchbox_max_y = closest_idy + self.max_search_distance + 1

            # TODO: add option to wrap domain
            searchbox_min_x = max(0, searchbox_min_x)
            searchbox_max_x = min(grid_area_coords1.shape[0], searchbox_max_x)
            searchbox_min_y = max(0, searchbox_min_y)
            searchbox_max_y = min(grid_area_coords1.shape[1], searchbox_max_y)

            subset_x = grid_area_coords1[searchbox_min_x:searchbox_max_x, searchbox_min_y:searchbox_max_y]
            subset_y = grid_area_coords2[searchbox_min_x:searchbox_max_x, searchbox_min_y:searchbox_max_y]

            shape = subset_x.shape

            subset_x = subset_x.flatten()
            subset_y = subset_y.flatten()

            subset_coords = np.stack((subset_x, subset_y), axis=0)
            coords_vec = np.array([station_x, station_y])

            distance_error = self.distance_error_func(coords_vec, subset_coords)

            if station_metric is None:
                area_error = 0
            else:
                assert grid_metric is not None
                subset_metric = grid_metric[searchbox_min_x:searchbox_max_x, searchbox_min_y:searchbox_max_y].flatten()
                area_error = self.metric_error_func(station_metric[i], subset_metric)

            error = area_error + self.lambd * distance_error

            try:
                best_error_1d_index = np.nanargmin(error)
                min_index = np.unravel_index(best_error_1d_index, shape)
                best_error = error[best_error_1d_index]

                center_offset_x = closest_idx - searchbox_min_x
                center_offset_y = closest_idy - searchbox_min_y
                subset_width = searchbox_max_y - searchbox_min_y
                closest_1d_index = center_offset_x * subset_width + center_offset_y

                closest_error = error[closest_1d_index]

                if closest_error <= self.min_error:  # if nearest cell is good enough
                    indx = closest_idx
                    indy = closest_idy
                    best_error = closest_error
                elif best_error <= self.max_error:
                    indx = (min_index[0] + searchbox_min_x) % grid_area_coords1.shape[0]
                    indy = (min_index[1] + searchbox_min_y) % grid_area_coords1.shape[1]
                else:  # if best match is still too bad, revert to closest cell
                    indx = closest_idx
                    indy = closest_idy
                    best_error = closest_error
            except ValueError:
                center_offset_x = closest_idx - searchbox_min_x
                center_offset_y = closest_idy - searchbox_min_y
                subset_width = searchbox_max_y - searchbox_min_y
                closest_1d_index = center_offset_x * subset_width + center_offset_y

                closest_error = error[closest_1d_index]
                indx = closest_idx
                indy = closest_idy
                best_error = closest_error

            indxs[i] = indx
            indys[i] = indy

            closest_indxs[i] = closest_idx
            closest_indys[i] = closest_idy

            errors[i] = best_error

        return indxs, indys, closest_indxs, closest_indys, errors
