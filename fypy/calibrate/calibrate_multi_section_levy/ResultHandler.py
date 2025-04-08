import os
import pandas as pd
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

    def convert_ndarray(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        else:
            return obj
    def to_csv(self, res_dict) -> pd.DataFrame:
        lines = []
        for ticker in res_dict:
            for model in res_dict[ticker]:
                mape = res_dict[ticker][model]["MAPE"]
                rmse = res_dict[ticker][model]["RMSE"]
                params = res_dict[ticker][model]["parameters"]
                init_guess = res_dict[ticker][model]["init_guess"]
                frozen_parameters = res_dict[ticker][model]["frozen_parameters"]
                frozen_parameters_str = json.dumps(self.convert_ndarray(obj=frozen_parameters))

                line = [ticker, model, mape, rmse, params, init_guess, frozen_parameters_str]
                lines.append(line)

        df = pd.DataFrame(
            lines,
            columns=[
                "Ticker",
                "Model",
                "MAPE",
                "RMSE",
                "Params",
                "Guess",
                "FrozenParameters",
            ],
        )

        print(df)

        df.to_parquet(self.path + "/res.parquet")

        if self._verbose:
            print(f"Results saved in the folder {self.path}")

        return df


