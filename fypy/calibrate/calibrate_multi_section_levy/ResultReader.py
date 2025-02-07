import pandas as pd
import numpy as np

from fypy.calibrate.calibrate_multi_section_levy.PathHandler import PathHandler
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.sv.Heston import _HestonBase
import ast
import matplotlib.pyplot as plt
from fypy.calibrate.calibrate_multi_section_levy.IVHandler import IVHandler

# TODO IV Handler (marketinfo, results)


class ResultReader(PathHandler):
    def __init__(self, res_path: str, disc_path: str, data_paths: dict):
        super().__init__(disc_path, data_paths)
        self._res_path = res_path
        self._results_df = self._get_results()
        self._best_results = self._get_best_results()
        self._best_results_dict = self._get_best_results_dict()
        self.market_infos = MarketInfo(
            self._disc_path, self._data_paths, _verbose=False
        )
        self.iv_handler = IVHandler(
            self.market_infos,
            self._best_results_dict,
            self._best_results,
            self._res_path,
        )

    def _get_results(self) -> pd.DataFrame:
        df = pd.read_parquet(self._res_path + "/res.parquet")
        df = df[["Ticker", "Model", "Iter", "MAPE", "RMSE", "Params", "Guess"]]
        return df

    def _get_best_results(self):
        idx = self._results_df.groupby(["Ticker", "Model"])["MAPE"].idxmin()
        df = self._results_df.loc[idx]
        return df

    def _get_best_results_dict(self):
        tickers = self._best_results.Ticker
        models = self._best_results.Model
        params = self._best_results.Params
        res = {
            (ticker, model): param
            for (ticker, model, param) in zip(tickers, models, params)
        }
        return res

    def save_iv_fig(self):
        self.iv_handler.save_all_iv()
