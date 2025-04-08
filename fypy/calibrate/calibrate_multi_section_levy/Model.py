import numpy as np
import datetime
import pandas as pd

# pandas options
pd.options.mode.chained_assignment = None  # default='warn'

from fypy.model.levy.BilateralGamma import BilateralGammaMotion

from fypy.model.levy import *
from fypy.model.sv.Heston import Heston
from fypy.model.sv.Bates import Bates
from fypy.model.sv.HestonDEJumps import HestonDEJumps

from fypy.termstructures.ForwardCurve import ForwardCurve
from fypy.termstructures.DiscountCurve import InterpolatedDiscountCurve, DiscountCurve


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
        self.model= model_dict.get(self._model_name)

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
                abs(cumul.c1) <= 1.2 and abs(cumul.c2) <= 0.8 and cumul.c2 >= 0
            ):  # skew entre -2 et 0
                is_good_guess = True
        print("Good guess [Mean, Var]: ", np.round(cumul.c1, 2), np.round(cumul.c2, 2))
        return guess

    def determined_guess(self, model_name:str, ticker:str):
        init_guess = {
            "BG": {
                "AMZN": [
                    8.282347723593748,
                    17.858771407518166,
                    5.8240080835482875,
                    9.905827595501975
                ],
                "NFLX": [
                    11.117912842869966,
                    18.370930200077566,
                    9.311314777418316,
                    12.667384168138435
                ],
                "SHOP": [
                    2.48286933, 9.31425759, 4.55954031, 4.80252968
                ],
                "SPOT": [
                    15.847510314294311,
                    19.40207900145105,
                    12.277489307322103,
                    13.054295305526097
                ]
            },
            "BGM": {
                "AMZN": [
                    11.554916085257439,
                    97.63320384766116,
                    1.2895276940060632,
                    7.686025315096277,
                    0.24886029003586005
                ],
                "NFLX": [
                    1.052598365319495,
                    17.455756040886893,
                    0.3246503720866783,
                    5.141516688654363,
                    0.26813530112505324
                ],
                "SHOP": [
                    7.154279626354302,
                    9.29755097594311,
                    3.0878424677592595,
                    4.858713055422091,
                    0.2553592334808104
                ],
                "SPOT": [
                    7.58934471, 8.08271683, 0.27062367, 4.65247138, 0.62783253
                ]
            },
            "CGMY": {
                "AMZN": [
                    0.23990732483925287,
                    6.043111810162788,
                    13.08964823388836,
                    1.1237800403162597
                ],
                "NFLX": [
                    0.0023497052580397026,
                    0.00015567971503998248,
                    17.369871309300773,
                    1.948687242163479
                ],
                "SHOP": [
                    0.03561935240492091,
                    0.000948163601259239,
                    3.190807690880335,
                    1.7473286592579147
                ],
                "SPOT": [
                    2.4759713791132407,
                    10.531585594116438,
                    16.608052545840298,
                    0.5758995810762103
                ]
            },
            "KDE": {
                "AMZN": [
                    0.2510629333402113,
                    1.3254690597255143,
                    0.8901042165809533,
                    25.346699889874245,
                    5.498482651541336
                ],
                "NFLX": [
                    0.27458358528162463,
                    1.834633424233265,
                    0.912441418768996,
                    34.05609345567448,
                    6.226074658794586
                ],
                "SHOP": [
                    0.4254439075037811,
                    1.5889110434949827,
                    0.9829536177531447,
                    11.561413581584642,
                    0.198079882714954
                ],
                "SPOT": [
                    0.31622793492780865,
                    0.44033234177016195,
                    0.711952146383646,
                    25.850446869498136,
                    5.443603790669857
                ]
            },
            "MJD": {
                "AMZN": [
                    0.24480305403021724,
                    0.4078688242133251,
                    -0.10340827311452377,
                    0.19493284190638938
                ],
                "NFLX": [
                    0.2814534448016537,
                    0.06116302786742476,
                    -0.19809167178083667,
                    0.2707625956546219
                ],
                "SHOP": [
                    0.4501575131813136,
                    0.039016884070270756,
                    -1.5570678863625098,
                    1.6264961339164963
                ],
                "SPOT": [
                    0.3202397527990224,
                    0.029986750724225145,
                    -0.33144014509198994,
                    0.3504797450949462
                ]
            },
            "NIG": {
                "AMZN": [
                    12.959025671908396,
                    -4.197257573196975,
                    0.8839211126531135
                ],
                "NFLX": [
                    7.3990297, -0.03585549,  4.02227187
                ],
                "SHOP": [
                    6.142113011068288,
                    -2.154528989502321,
                    1.3823434293756214
                ],
                "SPOT": [
                    12.46736617869119,
                    -2.80907735201137,
                    1.3338596944724737
                ]
            }
        }
        return init_guess[model_name][ticker]
