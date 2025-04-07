import datetime
import os
import pandas as pd
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR
import json
import numpy as np


class ResultHandler:
    def __init__(self, _verbose=True):
        self.path = self._make_and_get_path()
        self._verbose = _verbose


    def _make_and_get_path(self):
        path = os.path.join(os.getcwd(), "calibration_saved", "LS_CONS")
        os.makedirs(path, exist_ok=True)
        return path



    def to_csv(self, res_dict) -> pd.DataFrame:
        lines = []
        for ticker in res_dict:
            for model in res_dict[ticker]:
                for iter in res_dict[ticker][model]:
                    mape = res_dict[ticker][model][iter]["score"]["MAPE"]
                    rmse = res_dict[ticker][model][iter]["score"]["RMSE"]
                    params = res_dict[ticker][model][iter]["parameters"]
                    init_guess = res_dict[ticker][model][iter]["init_guess"]
                    frozen_parameters = res_dict[ticker][model][iter]["frozen_parameters"]
                    grid_values = res_dict[ticker][model][iter]["grid_values"]

                    # **Conversione da np.ndarray a liste Python**
                    def convert_ndarray(obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()  # Converte array NumPy in liste
                        elif isinstance(obj, dict):
                            return {k: convert_ndarray(v) for k, v in obj.items()}  # Ricorsione per dizionari
                        else:
                            return obj

                    frozen_parameters = convert_ndarray(frozen_parameters)
                    grid_values = convert_ndarray(grid_values)

                    # Ora possiamo convertire in JSON
                    frozen_parameters_str = json.dumps(frozen_parameters)
                    grid_values_str = json.dumps(grid_values)

                    line = [ticker, model, iter, mape, rmse, params, init_guess, frozen_parameters_str, grid_values_str]
                    lines.append(line)

        df = pd.DataFrame(
            lines,
            columns=[
                "Ticker",
                "Model",
                "Iter",
                "MAPE",
                "RMSE",
                "Params",
                "Guess",
                "FrozenParameters",
                "GridValues",
            ],
        )

        print(df)

        # Salvare il DataFrame in Parquet
        df.to_parquet(self.path + "/res.parquet")

        if self._verbose:
            print(f"Results saved in the folder {self.path}")

        return df  # Restituisco il DataFrame per debugging


