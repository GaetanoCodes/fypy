import datetime
import numpy as np
import pandas as pd
# pandas options
pd.options.mode.chained_assignment = None  # default='warn'
from fypy.model.levy.BilateralGamma import BilateralGammaMotion
from fypy.model.levy import *
from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import DiscountCurve
from typing import Dict


class Model:

    def __init__(
            self, model_name: str, fwd_curve: ForwardCurve, disc_curve: DiscountCurve
    ):

        ##################
        ## Private args ##
        ##################
        self._model_name = model_name
        self._saving_path = self._set_saving_path()
        self._fwd_curve = fwd_curve
        self._disc_curve = disc_curve
        self._name_to_model()
        self._bounds = self._get_bounds()

    def _set_saving_path(self) -> str:
        saving_path = "calibration_saved/" + datetime.datetime.now().strftime(
            "%d_%m_%Y__%H_%M_%S"
        )
        return saving_path

    def _get_bounds(self) -> list[tuple]:
        bounds = self.model.param_bounds()
        return bounds

    def _name_to_model(self):
        args = {"forwardCurve": self._fwd_curve, "discountCurve": self._disc_curve}
        model_dict = {
            "BS": BlackScholes(
                forwardCurve=self._fwd_curve, discountCurve=self._disc_curve, sigma=0.2
            ),
            "VG": VarianceGamma(
                sigma=0.18574156,
                theta=-0.92537499,
                nu=0.0485687,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            ),
            "BG": BilateralGamma(
                alpha_p=1.18,
                lambda_p=10.57,
                alhpa_m=1.44,
                lambda_m=5.57,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve
            ),
            "BGM": BilateralGammaMotion(
                alpha_p=1.18,
                lambda_p=10.57,
                alhpa_m=1.44,
                lambda_m=5.57,
                sigma=0.2,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            ),
            "NIG": NIG(
                alpha=38.66193053,
                beta=-15.19012801,
                delta=2.50232471,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            ),
            "CGMY": CMGY(
                C=1.75853947e-02,
                G=1.25202553,
                M=2.38905437e02,
                Y=1.78055594e00,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            ),
            "MJD": MertonJD(
                sigma=0.25074416,
                lam=1.09311656,
                muj=-0.11622712,
                sigj=0.08925924,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            ),
            "KDE": KouJD(
                sigma=0.247715067,
                lam=2.53789904,
                p_up=2.43434855e-08,
                eta1=81.2851952,
                eta2=13.9978289,
                forwardCurve=self._fwd_curve,
                discountCurve=self._disc_curve,
            )
        }
        self.model = model_dict.get(self._model_name)

    def _coordinate_from_bound(self, bound: tuple) -> np.array:
        bound_inf, bound_sup = bound[0], bound[1]
        if bound_inf == -np.inf:
            bound_inf = -10
        if bound_sup == np.inf:
            bound_sup = 10
        rd_point = np.random.uniform(bound_inf, bound_sup)
        return rd_point

    def _initial_guess_from_bounds(self) -> np.array:
        vector = []
        for bound in self._bounds:
            rd_point = self._coordinate_from_bound(bound)
            vector.append(rd_point)
        vector = np.array(vector)
        return vector

    def random_guess(self):
        print("* Initializing with a random guess.")
        is_good_guess = False
        while not is_good_guess:
            guess = self._initial_guess_from_bounds()
            self.model.set_params(guess)
            cumul = self.model.cumulants(1)
            if (
                    abs(cumul.c1) <= 0.8 and abs(cumul.c2) <= 0.3 and cumul.c2 >= 0
            ):  # skew entre -2 et 0
                is_good_guess = True
        print("Good guess [Mean, Var]: ", np.round(cumul.c1, 2), np.round(cumul.c2, 2))
        return guess

    def deterministic_guess(self, model_name: str, ticker: str, init_guess: Dict):
        return init_guess[model_name][ticker]
