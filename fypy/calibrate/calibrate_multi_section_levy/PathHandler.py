from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR


class PathHandler:
    def __init__(self, disc_path: str, data_paths: dict):
        self._disc_path = disc_path
        self._data_paths = data_paths

    def get_data_path(self, ticker):
        if ticker in self._data_paths:
            return self._data_paths[ticker]
        else:
            COLOR.write("No data corresponding to this ticker.", color="RED")
