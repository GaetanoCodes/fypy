import pandas as pd
import numpy as np
import os
import time
from copy import deepcopy
from fypy.calibrate.calibrate_multi_section_levy.PathHandler import PathHandler
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.model.sv.Heston import _HestonBase
from fypy.calibrate.calibrate_multi_section_levy.SVI import SVI
import matplotlib.pyplot as plt
from fypy.calibrate.calibrate_multi_section_levy.IVHandler import IVHandler

# TODO: add other functions to ResultContainer
# refactor display smile, split into short functions
# refactorer le code de display smile
# regenerer les images vec titres et labels
# graph display en iv handler
# sous classe svi
# il faudrait que ResultDisplay ait une classe MarketInfos et la donne à graph display et à SVI
plt.style.use("ggplot")

class Pricer:
    def __init__(self, model: Model):
        self.model = model
        self._pricer = self._get_pricer()

    def _get_pricer(self):
        pricer = ProjEuropeanPricer(
            model=self.model.model,
            N=2**12,
            L=20 if isinstance(self.model.model, _HestonBase) else 15,
        )
        return pricer

    def price(self, strikes: np.array, ttmy: float):
        prices = self._pricer.price_strikes(
            T=ttmy, K=strikes, is_calls=np.ones(len(strikes), dtype=bool)
        )
        return prices


class ResultsContainer:

    def __init__(self, res_path: str):
        self._res_path = res_path
        self._results_df = self._get_results()
        self._best_results = self._get_best_results()
        self._best_results_dict = self._get_best_results_dict()
        self.tickers = self._results_df.Ticker.unique()
        self.models = self.get_models()

    def _get_results(self) -> pd.DataFrame:
        print(os.getcwd())
        df = pd.read_parquet(self._res_path + "/res.parquet")
        df = df[["Ticker", "Model", "MAPE", "RMSE", "Params", "Guess"]]
        return df

    def _get_best_results(self):
        # Get the index of the minimum MAPE value for each (Ticker, Model) group
        idx = self._results_df.groupby(["Ticker", "Model"])["MAPE"].idxmin()

        # Replace NaN values with a compatible placeholder (pd.NA)
        idx = idx.fillna(pd.NA)

        # Separate valid indices from NaN indices
        valid_idx = idx.dropna()
        nan_idx = idx[idx.isna()]

        # Retrieve valid rows from the DataFrame
        df_valid = self._results_df.loc[valid_idx]

        # Create a separate DataFrame to represent NaN results
        df_nan = pd.DataFrame({
            "Ticker": nan_idx.index.get_level_values(0),
            "Model": nan_idx.index.get_level_values(1),
            "MAPE": pd.NA  # Preserve NaN representation
        })

        # Combine valid results with NaN results into a single DataFrame
        return pd.concat([df_valid, df_nan], ignore_index=True)

    def _get_best_results_dict(self):
        tickers = self._best_results.Ticker
        models = self._best_results.Model
        params = self._best_results.Params
        res = {
            (ticker, model): param
            for (ticker, model, param) in zip(tickers, models, params)
        }
        return res

    def get_models(self):
        models = {ticker: [] for ticker in self.tickers}
        ticker_list = self._best_results.Ticker
        model_list = self._best_results.Model
        for ticker, model in zip(ticker_list, model_list):
            if model not in models[ticker]:
                models[ticker].append(model)
        return models


class ResultDisplay(PathHandler):

    def __init__(self, res_path: str, disc_path: str, data_paths: dict):
        super().__init__(disc_path, data_paths)
        self.results = ResultsContainer(res_path)
        self.iv = GraphDisplay(disc_path, data_paths, self.results, _verbose=False)


class GraphDisplay:

    def __init__(self, disc_path, data_paths, results: ResultsContainer, _verbose):
        self._res = results
        self._verbose = _verbose
        self._mkt_infos = MarketInfo(disc_path, data_paths, _verbose)
        self._svi = SVI(disc_path, data_paths, _verbose)
        return

    # To be used directly by the user
    def display_smile(
        self,
        ticker: str,
        ttmd: int,
        models: list[str],
        market: bool = True,
        svi: bool = True,
        save: bool = True,
        show: bool = False,
    ):
        ttmy = self._mkt_infos._maturities[ticker][ttmd]
        # IV of different models
        strikes, k = self._mkt_infos._get_strikes_th(ticker, ttmy)
        ivs = self._get_iv_ticker_ttm(ticker, ttmy, strikes)
        # IV of market
        k_market = self._mkt_infos._get_log_strikes(ticker, ttmy)
        ivs_mkt = self._mkt_infos.surfaces[ticker].slices[ttmy].mid_vols
        # plot
        figsize = (3, 3)
        # plt.figure(figsize=figsize)
        plt.figure()
        plt.xlim(min(k_market) - 0.05, max(k_market) + 0.05)
        # plt.xlim(-0.7, max(k_market) + 0.05)

        if "Hes-DE" in models:
            l = deepcopy(models)
            l.remove("Hes-DE")
            models = ["Hes-DE"] + l
        print(models)
        for model in models:
            if model == "BIG":
                plt.plot(k, ivs[model], label=model, color="yellow")
            else:
                plt.plot(k, ivs[model], label=model)

        if svi:
            plt.plot(k, self._svi.get_svi_iv(ticker, k, ttmd), "r--", label="SVI")
        if market:
            plt.scatter(k_market, ivs_mkt, alpha=0.3, label="Market", color="black")
        self._write_end_fig()
        if save:
            plt.savefig(self._res._res_path + f"/{ticker}/{ticker}_{ttmd}.png", dpi=300)
        plt.title(f"{ticker} | ttm:{ttmd}d")
        if show:
            plt.show()
        plt.close()
        return

    def display_surface(self, ticker: str, model: str):
        ttmy = np.linspace(0.02, 1, 50)
        k = np.linspace(-0.5, 0.5, 50)
        t_mesh, k_mesh = np.meshgrid(ttmy, k)
        ivs = []
        for ttm in ttmy:
            strikes = np.exp(k) * self._mkt_infos._get_fwd(ticker, ttm)
            iv = self._get_iv_model_ticker_ttm(ticker, strikes, model, ttm)
            ivs.append(iv)
        ivs = np.array(ivs)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.plot_surface(k_mesh, t_mesh, ivs.T, cmap="viridis")
        # ax.set_xlabel(r"$\log (K/F_t)$")
        # ax.set_ylabel("ttm")
        # ax.set_zlabel("IV")
        # plt.tight_layout()
        # plt.title(f"{ticker} | {model}")
        # plt.show()
        fig = plt.figure(figsize=(12, 8))  # Increase the figure size
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(k_mesh, t_mesh, ivs.T, cmap="viridis")

        # Set labels with increased padding
        ax.set_xlabel(r"$\log (K/F_t)$", labelpad=5, fontsize=12)
        ax.set_ylabel("ttm", labelpad=5, fontsize=12)
        ax.set_zlabel("IV", labelpad=5, fontsize=12)
        ax.zaxis.labelpad = -6
        # Manually adjust the subplot parameters to give more space for labels and title
        plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.2)

        # Set the title of the plot
        plt.suptitle(f"{ticker} | {model}", y=0.88)
        plt.savefig(f"{ticker}_{model}_3d_surf.png")

        plt.show()
        return

    def display_fwd_smile(
        self, ticker: str, models: list[str], ttmds: list, svi: bool = True
    ):
        ttmy1 = self._mkt_infos._maturities[ticker][ttmds[0]]
        ttmy2 = self._mkt_infos._maturities[ticker][ttmds[1]]
        print(ttmds, ttmy1, ttmy2)

        # strikes_t1, k_t1 = self._mkt_infos._get_strikes_th(ticker, ttmy1)
        # strikes_t2, k_t2 = self._mkt_infos._get_strikes_th(ticker, ttmy2)
        # strikes = np.intersect1d(strikes_t1, strikes_t2)
        spot = self._mkt_infos.surfaces[ticker].spot
        print("SPOT", spot)

        strikes = np.arange(0.5 * spot, 1.5 * spot, 1)
        ivs_t1 = self._get_iv_ticker_ttm(ticker, ttmy1, strikes)
        ivs_t2 = self._get_iv_ticker_ttm(ticker, ttmy2, strikes)
        log_money = np.log(strikes / spot)

        for model in models:
            iv_t1, iv_t2 = ivs_t1[model], ivs_t2[model]
            iv_fwd = ((iv_t2**2 * ttmy2 - iv_t1**2 * ttmy1) / (ttmy2 - ttmy1)) ** 0.5

            plt.plot(log_money, iv_fwd, label=model)
        if svi:
            fwd1 = self._mkt_infos._get_fwd(ticker, ttmy1)
            fwd2 = self._mkt_infos._get_fwd(ticker, ttmy2)
            k_t1, k_t2 = np.log(strikes / fwd1), np.log(strikes / fwd2)
            iv_svi_t1 = self._svi.get_svi_iv(ticker, k_t1, ttmds[0])
            iv_svi_t2 = self._svi.get_svi_iv(ticker, k_t2, ttmds[1])

            plt.plot(
                log_money,
                ((iv_svi_t2**2 * ttmy2 - iv_svi_t1**2 * ttmy1) / (ttmy2 - ttmy1))
                ** 0.5,
                label="SVI",
                linestyle="-",
                color="blue",
            )
        plt.title(f"Forward IV | {ticker} | ({ttmds[0]}, {ttmds[1]})")
        self._write_end_fig(xlabel=r"$k$")
        plt.show()
        plt.plot(k_t1, iv_svi_t1)
        plt.plot(k_t2, iv_svi_t2)
        plt.plot(
            k_t1,
            ((iv_svi_t2**2 * ttmy2 - iv_svi_t1**2 * ttmy1) / (ttmy2 - ttmy1)) ** 0.5,
        )
        plt.show()
        return

    def save_all_figs(self):
        for ticker in self._res.tickers:
            self._make_path(ticker)
            for ttmd, ttmy in self._mkt_infos._maturities[ticker].items():
                self.display_smile(ticker, ttmd, self._res.models[ticker], show=False)
        return

    ########################################
    ########## PRIVATE FUNCTIONS ###########
    ########################################

    def modified_params_iv(
        self, ticker: str, model_name: str, ttmy: float, param_idx: int
    ):
        modif_range = [0.5, 0.75, 1, 1.25, 1.5]
        model = self._get_best_model(ticker, model_name)
        fwd = model._fwd_curve.fwd_T(ttmy)
        strikes = np.arange(0.5 * fwd, 2 * fwd)
        ivs = self._get_iv(ticker, strikes, model, ttmy)
        iv_dict = self._get_modified_model_iv(
            ticker, model_name, param_idx, strikes, ttmy, modif_range
        )
        self._plot_modification(
            iv_dict, np.log(strikes / fwd), param_idx, modif_range, ticker, ttmy
        )
        return

    def _plot_modification(
        self,
        iv_dict: dict,
        moneyness: np.ndarray,
        param_idx: int,
        modif_range: list,
        ticker: str,
        ttmy: float,
    ):
        param = {
            0: r"V_0",
            1: r"\theta",
            2: r"\kappa",
            3: r"\sigma_v",
            4: r"\rho",
            5: r"\lambda",
            6: r"p_up",
            7: r"\eta_1",
            8: r"\eta_2",
        }
        factor_alpha = 10
        alpha_range = (
            1 + factor_alpha * np.arange(0, 1, 1 / len(modif_range))
        ) / np.max((1 + factor_alpha * np.arange(0, 1, 1 / len(modif_range))))
        plt.figure(figsize=(3, 3))
        for modif_idx in iv_dict:
            plt.plot(
                moneyness,
                iv_dict[modif_idx],
                label=rf"${np.round(modif_range[modif_idx],2)}\times{param[param_idx]}$",
                alpha=alpha_range[modif_idx],
                linestyle="-",
                color="blue",
            )
        plt.xlim(-0.6, 0.6)
        plt.legend()
        plt.grid()
        plt.ylabel("IV", fontsize=10)
        plt.xlabel("log-Moneyness", fontsize=10)
        plt.title(f"{ticker}|ttm: {np.round(ttmy,2)} year", fontsize=10)
        # plt.ylim(0.25, 0.7)
        plt.savefig(f"{ticker}_Hes-DE_{param_idx}_{np.round(ttmy,2)}.png")
        plt.show()
        return

    def _get_modified_model_iv(
        self,
        ticker: str,
        model_name: str,
        param_idx: int,
        strikes: np.ndarray,
        ttmy: float,
        modif_range: list,
    ):
        model = self._get_best_model(ticker, model_name)
        param = model.model.get_params()

        iv_dict = {}
        prices_dict = {}
        for modif_idx in range(len(modif_range)):
            modif = modif_range[modif_idx]
            param_modif = deepcopy(param)
            param_modif[param_idx] = modif * param_modif[param_idx]
            model.model.set_params(param_modif)
            ivs = self._get_iv(ticker, strikes, model, ttmy)
            iv_dict[modif_idx] = ivs
        return iv_dict

    def _make_path(self, ticker: str):
        path_ticker = self._res._res_path + f"/{ticker}"
        os.mkdir(path_ticker)
        return

    def _get_iv(self, ticker: str, strikes: np.array, model: Model, ttmy: float):
        pricer = Pricer(model)
        prices = pricer.price(strikes, ttmy)

        ivs = self._mkt_infos.get_iv_from_prices(ticker, strikes, prices, ttmy)
        return ivs

    def _get_best_model(self, ticker: str, model_name: str) -> Model:
        disc = self._mkt_infos.disc_curve
        fwd = self._mkt_infos.fwds[ticker]
        model = Model(model_name, fwd, disc)
        param = self._res._best_results_dict[(ticker, model_name)]
        model.model.set_params(param)
        return model

    def _get_iv_model_ticker_ttm(
        self, ticker: str, strikes: np.array, model_name: str, ttmy: float
    ):
        model = self._get_best_model(ticker, model_name)
        ivs = self._get_iv(ticker, strikes, model, ttmy)
        return ivs

    def _get_iv_ticker_ttm(self, ticker: str, ttmy: float, strikes: np.array):
        ivs = {}
        for model_name in self._res.models[ticker]:
            iv = self._get_iv_model_ticker_ttm(ticker, strikes, model_name, ttmy)
            ivs[model_name] = iv
        return ivs

    def _get_iv_ticker(self, ticker: str):
        ivs = {}
        rel_strikes = {}
        for ttm, ttmy in self._mkt_infos._maturities[ticker].items():
            strikes_th = self._mkt_infos._get_strikes_th(ticker, ttmy)
            fwd = self._mkt_infos.fwds[ticker].fwd_T(ttmy)
            iv = self._get_iv_ticker_ttm(ticker, ttmy, strikes_th)
            ivs[ttm] = iv
            rel_strikes[ttm] = strikes_th / fwd
        return ivs, rel_strikes

    def _write_end_fig(self, xlabel: str = "log-Moneyness", ylabel: str = "IV"):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.legend()
        return
