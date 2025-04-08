from typing import Optional

from examples.visualization import calculate_error_metrics
from fypy.calibrate.MSLevyModelCalibrator import MSLevyModelCalibrator
from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
from fypy.calibrate.calibrate_multi_section_levy.ResultHandler import ResultHandler
from fypy.fit.Calibratable import Calibratable
from fypy.market.MarketSurface import MarketSurface
from fypy.pricing.StrikesPricer import StrikesPricer


class Calibrator(MarketInfo):

    def __init__(
            self,
            model_names: list[str],
            disc_path: str,
            data_paths: dict,
            init_guess_random : bool = False,
            verbose: bool = True,
    ):
        super().__init__(disc_path=disc_path, data_paths=data_paths, _verbose=verbose)
        self.model_names = model_names
        self.results = self._get_empty_result()
        self.results_hanlder = ResultHandler()
        self.init_guess_random = init_guess_random

    def calibrate(self):
        for ticker in self.tickers:
            for model_name in self.model_names:
                if self._verbose:
                    COLOR.write(f"Calibrating on {ticker}", style=["BOLD"], color="BLUE")
                    COLOR.write(f"Model {model_name} ", indent=1, color="YELLOW")
                self._calibrate_and_error(ticker, model_name)
        self.results_hanlder.to_csv(self.results)

    def _calibrate_and_error(self, ticker: str, model_name: str):
        calibrated_params, pricer, init_guess, frozen_parameters = self._calibrate_model(
            model_name, ticker
        )

        mape, rmse = calculate_error_metrics(
            pricer=pricer,
            surface=self.surfaces[ticker],
            ivc=self._ivcs[ticker],
        )
        print("Calibrated params : ", calibrated_params)
        print(calibrated_params)
        self._store_result(
            ticker, model_name, mape, rmse, calibrated_params, init_guess, frozen_parameters
        )
        return

    def _calibrate_model(self, model_name: str, ticker: str):
        model = Model(model_name, self.fwds[ticker], self.disc_curve)
        init_guess = model.determined_guess(model_name=model_name, ticker=ticker) if self.init_guess_random==False else model.random_guess()

        model.model.set_params(init_guess)

        res, pricer, frozen_parameters = self._calibration(
            model=model.model,
            model_name=model._model_name,
            surface=self.surfaces[ticker],
        )
        calibrated_params = model.model.get_params()
        return calibrated_params, pricer, init_guess, frozen_parameters

    def _store_result(
            self, ticker, model_name, mape, rmse, cal_params, init_guess, frozen_parameters
    ):
        self.results[ticker][model_name]["MAPE"] = mape
        self.results[ticker][model_name]["RMSE"] = rmse
        self.results[ticker][model_name]["parameters"] = cal_params
        self.results[ticker][model_name]["init_guess"] = init_guess
        self.results[ticker][model_name]["frozen_parameters"] = frozen_parameters

    def _calibration(
            self,
            model: Calibratable,
            model_name: str,
            surface: MarketSurface,
            pricer: Optional[StrikesPricer] = None,
    ):
        if model_name == "SABR":
            return SabrModelCalibrator(surface=surface).calibrate(
                model=model, pricer=pricer
            )
        return MSLevyModelCalibrator(
            surface=surface, do_vega_weight=True,
        ).calibrate(model=model, pricer=pricer)

    def _get_empty_result(self) -> dict:
        res = {}
        for ticker in self.tickers:
            res[ticker] = {}
            for model in self.model_names:
                res[ticker][model] = {}
        return res


