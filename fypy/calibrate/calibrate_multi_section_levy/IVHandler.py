import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from fypy.calibrate.calibrate_multi_section_levy.PathHandler import PathHandler
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.sv.Heston import _HestonBase


class IVHandler:
    def __init__(
        self, market_infos: MarketInfo, best_res_dict: dict, best_res, res_path: str
    ):
        self.market_infos = market_infos
        self._best_results_dict = best_res_dict
        self._best_results = best_res
        self._res_path = res_path

    def get_iv(self, ticker: str, strikes: np.array, model: Model, ttmy: float):
        pricer = self._get_pricer(model)
        prices = pricer.price_strikes(
            T=ttmy, K=strikes, is_calls=np.ones(len(strikes), dtype=bool)
        )

        ivs = self.market_infos._ivcs[ticker].imply_vols(
            strikes=strikes,
            prices=prices,
            is_calls=np.ones(len(strikes), dtype=bool),
            ttm=ttmy,
        )
        return ivs

    def get_iv_model_ticker_ttm(
        self, ticker: str, strikes: np.array, model_name: str, ttmy: float
    ):
        disc = self.market_infos.disc_curve
        fwd = self.market_infos.fwds[ticker]
        model = Model(model_name, fwd, disc)
        param = self._best_results_dict[(ticker, model_name)]
        model.model.set_params(param)
        ivs = self.get_iv(ticker, strikes, model, ttmy)
        return ivs

    def get_iv_ticker_ttm(self, ticker: str, ttmy: float, strikes: np.array):
        ivs = {}
        for model_name in self._best_results.Model.unique():
            iv = self.get_iv_model_ticker_ttm(ticker, strikes, model_name, ttmy)
            ivs[model_name] = iv
        return ivs

    def get_iv_ticker(self, ticker):
        ivs = {}
        rel_strikes = {}
        for ttm, ttmy in self.market_infos._maturities[ticker].items():
            strikes_market = self.market_infos.surfaces[ticker].slices[ttmy].strikes
            bounds = [min(strikes_market), max(strikes_market)]
            strikes_th = np.arange(bounds[0], bounds[1], 2)
            fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
            iv = self.get_iv_ticker_ttm(ticker, ttmy, strikes_th)
            ivs[ttm] = iv
            rel_strikes[ttm] = strikes_th / fwd
        return ivs, rel_strikes

    def forward_smile(self, ticker, td1, td2):
        k = np.arange(-0.5, 0.5, 0.01)
        ttmy1 = self.market_infos._maturities[ticker][td1]
        ttmy2 = self.market_infos._maturities[ticker][td2]
        fwd1 = self.market_infos.fwds[ticker].fwd_T(ttmy1)
        fwd2 = self.market_infos.fwds[ticker].fwd_T(ttmy2)
        strikest1 = np.exp(k) * fwd1
        strikest2 = np.exp(k) * fwd2

        s_mar1 = self.market_infos.surfaces[ticker].slices[ttmy1].strikes
        s_mar2 = self.market_infos.surfaces[ticker].slices[ttmy2].strikes
        ind1 = np.nonzero(np.in1d(s_mar1, s_mar2))[0]
        ind2 = np.nonzero(np.in1d(s_mar2, s_mar1))[0]
        s1 = s_mar1[ind1]

        vol_mar1 = self.market_infos.surfaces[ticker].slices[ttmy1].mid_vols[ind1]
        vol_mar2 = self.market_infos.surfaces[ticker].slices[ttmy2].mid_vols[ind2]
        # print(strikest1, s_mar1)
        ivt1_dict = self.get_iv_ticker_ttm(ticker, ttmy1, strikest1)
        ivt2_dict = self.get_iv_ticker_ttm(ticker, ttmy2, strikest2)
        for model in ivt1_dict:
            plt.plot(strikest1, ivt1_dict[model], label=model)
        plt.scatter(s_mar1, self.market_infos.surfaces[ticker].slices[ttmy1].mid_vols)
        plt.legend()
        plt.show()
        for model in ivt1_dict:
            plt.plot(strikest2, ivt2_dict[model], label=model)
        plt.scatter(s_mar2, self.market_infos.surfaces[ticker].slices[ttmy2].mid_vols)
        plt.legend()
        plt.show()
        # plt.plot(strikest1, ivt1_dict[model], label=model)
        for model in ivt1_dict:
            plt.plot(
                strikest1,
                (
                    (ivt2_dict[model] ** 2 * td2 - ivt1_dict[model] ** 2 * td1)
                    / (td2 - td1)
                )
                ** 0.5,
                label=model,
            )
        plt.scatter(
            s1,
            # vol_mar1,
            ((vol_mar2**2 * td2 - vol_mar1**2 * td1) / (td2 - td1)) ** 0.5,
        )
        # plt.scatter(s_mar1, self.market_infos.surfaces[ticker].slices[ttmy1].mid_vols)
        plt.plot()
        plt.legend()
        plt.grid()
        plt.show()

    def save_all_iv(self):
        for ticker in self._best_results.Ticker.unique():
            path_ticker = self._res_path + f"/{ticker}"
            os.mkdir(path_ticker)
            t0 = time.time()

            ivs, rel_strikes = self.get_iv_ticker(ticker)
            print(time.time() - t0)

            for ttmd, ttmy in self.market_infos._maturities[ticker].items():
                plt.figure(figsize=(10, 10))
                for model in ivs[ttmd]:
                    plt.plot(np.log(rel_strikes[ttmd]), ivs[ttmd][model], label=model)
                fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
                k = np.log(
                    self.market_infos.surfaces[ticker].slices[ttmy].strikes / fwd
                )
                plt.scatter(
                    k,
                    self.market_infos.surfaces[ticker].slices[ttmy].mid_vols,
                    label="Market",
                )
                plt.xlabel("log-Moneyness")
                plt.ylabel("IV")
                plt.grid()
                plt.legend()
                plt.savefig(path_ticker + f"/{ttmd}_{ticker}.png", dpi=300)
                plt.close()
        return

    def _get_pricer(self, model: Model):
        pricer = ProjEuropeanPricer(
            model=model.model,
            N=2**12,
            L=20 if isinstance(model.model, _HestonBase) else 15,
        )
        return pricer
