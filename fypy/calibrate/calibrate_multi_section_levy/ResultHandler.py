import datetime
import os
import pandas as pd
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR


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
                    #foc = res_dict[ticker][model][iter]["FOC"]

                    line = [ticker, model, iter, mape, rmse, params, init_guess]# foc]
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
                #"FOC",
            ],
        )
        print(df)
        df.to_parquet(self.path + "/res.parquet")
        if self._verbose:
            COLOR.write(f"Results saved in the folder {self.path}")

        return
