import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from scipy.optimize import minimize
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import MarketInfo
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer

from fypy.pricing.analytical.black_scholes import black76_price_strikes
from fypy.calibrate.calibrate_multi_section_levy.Model import Model
import numpy.typing as npt
from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess

# intégrer SVI dans la classe calibration pour pouvoir afficher les smiles SVI ET les smiles forwards
import numpy as np
from scipy.stats import norm

N = norm.cdf


def bs_call(S, K, T, r, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 10
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if abs(diff) < PRECISION:
            return sigma
        sigma = sigma + diff / vega  # f(x) / f'(x)
    # if np.isnan(sigma):
    #     print("NONE #####")
    #     raise ValueError
    return sigma  # value wasn't found, return best guess so far


class SVI:
    def __init__(self, disc_path: str, data_paths: dict, _verbose_SVI: bool = True):
        self._verbose = _verbose_SVI
        self.market_infos = MarketInfo(disc_path, data_paths, _verbose=False)
        self._bounds = self._get_bounds()
        self._args_ls = self._get_ls_args()
        self.results = self.calibrate_svi()
        self.fwd = SVIForwardIV(self.results, self.market_infos)
        return

    def _get_bounds(self):
        bounds = {
            "svi": [
                (-np.inf, np.inf),
                (0, np.inf),
                (-1, 1),
                (-np.inf, np.inf),
                (0, np.inf),
            ]
        }
        return bounds

    def _get_ls_args(self) -> dict:
        args = {"bounds": self._bounds["svi"], "tol": 1e-6}
        return args

    def calibrate_svi(self):
        if self._verbose:
            COLOR.write("Calibrating SVI", style=["BOLD"], color="DARKCYAN")
        svi = {}
        for ticker in self.market_infos.tickers:
            svi[ticker] = self.calibrate_svi_ticker(ticker)
            if self._verbose:
                COLOR.write(f"Ticker {ticker}: done!", color="GREEN", indent=1)
        return svi

    def get_svi_iv(self, ticker, k: np.array, ttmd: int):
        params_t1 = self.results[ticker][self.market_infos._maturities[ticker][ttmd]]
        a, b, rho, m, sigma = params_t1
        iv = (a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)) ** 0.5
        return iv

    def calibrate_svi_ticker(self, ticker: str):
        svi = {}
        for ttmy, _ in self.market_infos.surfaces[ticker].slices.items():
            cal_params = self.calibrate_slice_svi(ticker, ttmy)
            svi[ttmy] = cal_params
        return svi

    def constraint_svi(self, svi_param):
        a, b, rho, m, sigma = svi_param
        cons = a + b * sigma * (1 - rho**2) ** 0.5
        return cons

    def calibrate_slice_svi(self, ticker, ttmy):
        x0 = [-6.14244784, 5.57900399, -0.38657939, 0.5454216, 1.20557006]
        loss_fun = partial(self.loss_svi_slice, ticker, ttmy)
        constraint = {"type": "ineq", "fun": self.constraint_svi}
        res = minimize(loss_fun, x0, **self._args_ls, constraints=constraint)
        # self.display_res(res.x, ttmy, ticker)
        return res.x

    def loss_svi_slice(self, ticker: str, ttmy: float, svi_param: np.array):
        # voir si on peut pas faire les appels de vols, k, etc avant et passer en partial
        a, b, rho, m, sigma = svi_param
        fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
        strikes = self.market_infos.surfaces[ticker].slices[ttmy].strikes
        iv = self.market_infos.surfaces[ticker].slices[ttmy].mid_vols
        k = np.log(strikes / fwd)
        iv_svi = (a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)) ** 0.5
        loss = ((1 / len(strikes)) * np.power(iv - iv_svi, 2).sum()) ** 0.5
        return loss

    def display_res(self, x, ttmy, ticker):
        fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
        strikes = self.market_infos.surfaces[ticker].slices[ttmy].strikes
        iv = self.market_infos.surfaces[ticker].slices[ttmy].mid_vols
        k = np.log(strikes / fwd)
        a, b, rho, m, sigma = x
        iv_svi = (a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)) ** 0.5

        plt.plot(k, iv, label="market")
        plt.plot(k, iv_svi, label="SVI")
        plt.title(f"{ticker}-{ttmy}")
        plt.legend()
        plt.show()
        return

    def display_iv_clouds(self):

        for ticker in self.market_infos.tickers:
            ivs = []
            strikes = []
            ttmys = []
            for ttmy, slice in self.market_infos.surfaces[ticker].slices.items():
                strike = slice.strikes
                iv = slice.mid_vols
                ttm = len(strike) * [ttmy]
                strikes += list(strike)
                ivs += list(iv)
                ttmys += ttm

            plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter3D(ttmys, strikes, ivs, c=ivs)
            plt.show()
            plt.close()
        return

    def forward_smile(self, ticker, td1, td2):
        print(self.market_infos._maturities)
        params_t1 = self.results[ticker][self.market_infos._maturities[ticker][td1]]
        params_t2 = self.results[ticker][self.market_infos._maturities[ticker][td2]]

        def iv_t1(k):
            a, b, rho, m, sigma = params_t1
            iv = (a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)) ** 0.5
            return iv

        def iv_t2(k):
            a, b, rho, m, sigma = params_t2
            iv = (a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)) ** 0.5
            return iv

        k = np.arange(-0.5, 0.5, 0.01)
        ivt1, ivt2 = iv_t1(k), iv_t2(k)
        # plt.plot(k, ivt1)
        # plt.plot(k, ivt2)
        plt.plot(k, ((ivt2 * td2 - ivt1 * td1) / (td2 - td1)) ** 0.5)
        plt.title(f"Forward Smile {ticker} | {td2, td1}")
        plt.show()
        return


class SVIForwardIV:
    def __init__(self, results: dict, market_infos=MarketInfo):
        self.results = results
        self.market_infos = market_infos
        return

    def _price_fwd_option_PROJ(
        self,
        model: Model,
        strike: npt.NDArray | float,
        ticker: str,
        ttmy1: int,
        ttmy2: int,
    ):
        pricer = ProjForwardStartingOption(model.model)
        spot = self.market_infos.surfaces[ticker].spot
        # ttmy1 = self.market_infos._maturities[ticker][td1]
        # ttmy2 = self.market_infos._maturities[ticker][td2]
        # print(ttmy2 - ttmy1)
        price = pricer.price_strikes_fill(
            ttmy1, ttmy2 - ttmy1, spot, strike=strike
        )  # include strike AND vectorize along strikes in the code of price
        eur_pricer = ProjEuropeanPricer(model=model.model, N=2**14)
        strikes = np.array([strike * spot])
        eur_price = eur_pricer.price_strikes(ttmy2, strikes, np.ones(len(strikes)))
        # print(f"FWD price:  {price} Eur price {eur_price}")
        return price

    def _obj_function(
        self,
        model: Model,
        ticker: str,
        k: npt.NDArray | float,
        ttmy1: int,
        ttmy2: int,
        # vol: npt.NDArray,
    ):
        # ttmy1 = self.market_infos._maturities[ticker][td1]
        # ttmy2 = self.market_infos._maturities[ticker][td2]
        tau = ttmy2 - ttmy1
        mean = self.market_infos.fwds[ticker].fwd_T(
            ttmy1
        )  # self._get_mean_gath(ticker, td1)
        proj_price = self._price_fwd_option_PROJ(model, k, ticker, ttmy1, ttmy2)
        new_price = np.array([proj_price / mean])
        disc = self.market_infos.disc_curve.discount_T(tau)
        # print(new_price, 1 / disc - k, disc, k, 1 / disc)
        iv = implied_volatility_from_a_transformed_rational_guess(
            new_price, 1 / disc, k, tau, 1
        )
        print("FWD", proj_price)
        # r = self.market_infos.disc_curve.implied_rate(ttmy2)
        # iv = find_vol(new_price, 1, k, tau, r)
        return iv

    def _get_best_model(self, model_name: str, ticker: str):
        model = Model(
            model_name, self.market_infos.fwds[ticker], self.market_infos.disc_curve
        )
        # params = self.results[ticker][model_name]
        params = np.array([0.26536192, 0.37982436, 0.08937994, 8.79446411, 4.5674848])
        model.model.set_params(params)
        return model

    def _compute_Kirkby_iv(
        self,
        strike: float,
        model_name: str,
        ticker: str,
        td1: int,
        td2: int,
    ):
        model = self._get_best_model(model_name, ticker)
        k = np.array([strike])
        # obj_fun = partial(self._obj_function, model, ticker, k, td1, td2)
        iv = []
        rel_strikes = np.linspace(0.5, 1.5, 100)
        for k in rel_strikes:
            # try:
            iv_kirkby = self._obj_function(model, ticker, k, td1, td2)
            # print(float(iv_kirkby))
            iv.append(float(iv_kirkby))
            # except:
            #     print("None")
            #     iv.append(0)
        # print(iv)
        print(iv)
        plt.scatter(np.log(rel_strikes), iv)
        plt.ylim((0, 1))
        plt.show()
        return

    # def _iv_gath(
    #     self, k: npt.NDArray | float, cal_params: npt.NDArray, derivat_ord: int = 0
    # ):
    #     a, b, rho, m, sigma = cal_params
    #     match derivat_ord:
    #         case 0:
    #             return a + b * (rho * (k - m) + ((k - m) ** 2 + sigma**2) ** 0.5)
    #         case 1:
    #             den = (k - m) ** 2 + sigma**2
    #             return b * ((rho * den**0.5 + k - m) / den**0.5)
    #         case 2:
    #             den = (k - m) ** 2 + sigma**2
    #             return sigma**2 * b / (den**1.5)

    # def _get_d_gath(
    #     self, k: npt.NDArray | float, cal_params: npt.NDArray, pm: str = "+"
    # ):
    #     w_sqrt = self._iv_gath(k, cal_params) ** 0.5
    #     if pm == "+":
    #         return -k / w_sqrt + w_sqrt / (2**0.5)
    #     else:
    #         return -k / w_sqrt - w_sqrt / (2**0.5)

    # def _get_g(self, k: npt.NDArray | float, cal_params: npt.NDArray):
    #     w = self._iv_gath(k, cal_params)
    #     w1 = self._iv_gath(k, cal_params, derivat_ord=1)
    #     w2 = self._iv_gath(k, cal_params, derivat_ord=2)
    #     member1 = (1 - k * w1 / (2 * w)) ** 2
    #     member2 = 0.25 * w1**2 * (0.25 + 1 / w)
    #     member3 = 0.5 * w2
    #     return member1 - member2 + member3

    # def _get_density_function(self, k: npt.NDArray | float, cal_params: npt.NDArray):
    #     d_minus = self._get_d_gath(k, cal_params, pm="-")
    #     w = self._iv_gath(k, cal_params)
    #     g = self._get_g(k, cal_params)
    #     member1 = g / (2 * np.pi * w) ** 0.5
    #     member2 = np.exp(-0.5 * d_minus**2)
    #     density = member1 * member2
    #     return density

    # def display_density(self, ticker: str, td1: int):
    #     ttmy = self.market_infos._maturities[ticker][td1]
    #     cal_params = self.results[ticker][ttmy]
    #     fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
    #     S_T = np.linspace(50, 400, 400)
    #     real_space = np.log(S_T / fwd)
    #     density = self._get_density_function(real_space, cal_params)
    #     # plt.plot(real_space, density)
    #     # plt.title(f"Gatheral Density")
    #     # plt.show()
    #     # print("Integral of density ", np.sum(density) * np.diff(real_space).mean())
    #     # print("Mean ", (K * density * np.diff(K).mean()).sum())

    #     real_space = np.arange(50, 400, 1)
    #     log_space = np.log(real_space / fwd)
    #     ds = np.mean(np.diff(real_space))
    #     density = self._get_density_function(log_space, cal_params)
    #     print("Density", ds * (density.sum()))

    #     return

    # def _get_mean_gath(self, ticker: str, td1: int):
    #     ttmy = self.market_infos._maturities[ticker][td1]
    #     fwd = self.market_infos.fwds[ticker].fwd_T(ttmy)
    #     cal_params = self.results[ticker][ttmy]
    #     real_space = np.arange(0.2 * fwd, 1.8 * fwd, 1)
    #     log_space = np.log(real_space / fwd)
    #     ds = np.mean(np.diff(real_space))
    #     density = self._get_density_function(log_space, cal_params)
    #     mean = ds * density.sum()
    #     print("Mean", mean)
    #     return mean
