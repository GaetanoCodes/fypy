from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
from fypy.calibrate.MSLevyModelCalibrator import MSLevyModelCalibrator
from examples.visualization import calculate_error_metrics
from fypy.fit.Calibratable import Calibratable
from fypy.market.MarketSurface import MarketSurface
from typing import Optional
from fypy.pricing.StrikesPricer import StrikesPricer
from fypy.calibrate.SabrModelCalibrator import SabrModelCalibrator
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR

# model dict wih **args then
# TODO: Class that take a results (dict) and save it as csv Multi doss: time-> tickers
from fypy.calibrate.calibrate_multi_section_levy.ResultHandler import ResultHandler


class Calibrator(MarketInfo):

    def __init__(
        self,
        model_names: list[str],
        disc_path: str,
        data_paths: dict,
        verbose: bool = True,
    ):
        super().__init__(disc_path=disc_path, data_paths=data_paths, _verbose=verbose)
        self.model_names = model_names
        self.precision = [1e-7]
        self.num_iter = 1
        self.results = self._get_empty_result()
        self.results_hanlder = ResultHandler()



    def calibrate(self):
        for ticker in self.tickers:
            for model_name in self.model_names:
                if self._verbose:
                    COLOR.write(f"Calibrating on {ticker}", style=["BOLD"], color="BLUE")
                    COLOR.write(f"Model {model_name} ", indent=1, color="YELLOW")
                self._calibrate_and_error(ticker, model_name, iter=1)
        self.results_hanlder.to_csv(self.results)

    def _calibrate_and_error(self, ticker: str, model_name: str, iter: int):
        iter=1
        calibrated_params, pricer, init_guess = self._calibrate_model(
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
            ticker, model_name, iter, mape, rmse, calibrated_params, init_guess
        )
        return

    def _calibrate_model(self, model_name: str, ticker: str):
        model = Model(model_name, self.fwds[ticker], self.disc_curve)
        init_guess = model.random_guess()
        for ftol in self.precision:
            model.model.set_params(init_guess)

            res, pricer = self._calibration(
                model=model.model,
                model_name=model._model_name,
                surface=self.surfaces[ticker],
                ftol=ftol,
            )
            calibrated_params = model.model.get_params()

        return calibrated_params, pricer, init_guess#, res.optimality

    def _store_result(
        self, ticker, model_name, iter, mape, rmse, cal_params, init_guess#, foc
    ):
        self.results[ticker][model_name][iter]["score"]["MAPE"] = mape
        self.results[ticker][model_name][iter]["score"]["RMSE"] = rmse
        self.results[ticker][model_name][iter]["parameters"] = cal_params
        self.results[ticker][model_name][iter]["init_guess"] = init_guess
        #self.results[ticker][model_name][iter]["FOC"] = foc
        # print("FOC 3 ", self.results[ticker][model_name][iter]["FOC"], foc)

    def _calibration(
        self,
        model: Calibratable,
        model_name: str,
        surface: MarketSurface,
        pricer: Optional[StrikesPricer] = None,
        ftol: float = 1e-5,
    ):
        if model_name == "SABR":
            return SabrModelCalibrator(surface=surface).calibrate(
                model=model, pricer=pricer
            )
        return MSLevyModelCalibrator(
            surface=surface, do_vega_weight=True
        ).calibrate(model=model, pricer=pricer)

    def _get_empty_result(self) -> dict:
        res = {}
        for ticker in self.tickers:
            res[ticker] = {}
            for model in self.model_names:
                res[ticker][model] = {}
                res[ticker][model][1] = {
                    "score": {"RMSE": None, "MAPE": None},
                    "parameters": None,
                    "init_guess": None,
                }
        return res
