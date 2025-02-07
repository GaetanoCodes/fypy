"""test ms levy chf"""
import unittest
from copy import deepcopy

import numpy as np

from fypy.model.levy import LevyModel
from fypy.model.levy.BlackScholes import BlackScholes
from fypy.termstructures.DiscountCurve import DiscountCurve_ConstRate
from fypy.termstructures.EquityForward import EquityForward


class MSLevyTest(unittest.TestCase):
    """MSLevyTest"""

    def test_manual_vs_levy_module_chf(self):
        """test of multi section chf"""
        s0 = 100
        r = 0.05
        q = 0.01
        xi = np.arange(-1, 1, 1e-2)
        errors = []
        disc_curve = DiscountCurve_ConstRate(rate=r)
        div_disc = DiscountCurve_ConstRate(rate=q)
        fwd = EquityForward(S0=s0, discount=disc_curve, divDiscount=div_disc)

        model = BlackScholes(forwardCurve=fwd,
                             discountCurve=disc_curve)
        times = np.arange(1e-2, 13.1, 1e-2)
        ms_params = get_multi_section_params(model)
        for t in times:
            model.set_multi_section(
                multi_section=True, frozen_params=ms_params)
            chf_ms_module = model.chf(t, xi)
            chf_manual = get_manual_chf(model, ms_params, t, xi, verbose=False)
            errors.append(np.max(np.abs(chf_ms_module-chf_manual)))

        print(f"Error L_inf:{np.max(errors)}")
        self.assertAlmostEqual(np.max(errors), 0)


def get_multi_section_params(model):
    """get multi section params"""
    times = [0.1, 1, 1.2, 2, 10]  # np.arange(0.2, 2, 0.1)
    if isinstance(model, BlackScholes):
        params = {tenor: [np.round(np.random.rand(), 2)]
                  for tenor in times}
        # print(params)
        return params
    else:
        raise NotImplementedError


def distribute_time(time: float, times: list) -> list:
    """distribute time"""
    result = np.zeros(len(times))
    remaining = time
    len_times = len(times)
    for i in range(len_times):
        if i == 0:
            max_fill = times[0]
        else:
            max_fill = times[i] - times[i - 1]

        if remaining > 0:
            result[i] = min(remaining, max_fill)
            remaining -= result[i]
        else:
            break
    assert abs(sum(result)+remaining -
               time) < 1e-6, f"Error: sum={sum(result)}, expected={time}"

    return result.tolist(), remaining


def get_manual_chf(model_orig: LevyModel, ms_params: dict, t: float, xi: np.ndarray, verbose: bool = True):
    """get manual chf"""
    model = deepcopy(model_orig)
    times_ms, params_values = list(ms_params.keys()), list(ms_params.values())
    distributed_times, remaining_time = distribute_time(t, times_ms)
    model.set_multi_section(False)
    chf = model.chf(remaining_time, xi)
    if verbose:
        print("params:", model.get_params(),
              "during:", remaining_time)
    for time_interval, param in zip(distributed_times, params_values):
        # print("-")
        model.set_params(param)
        chf *= model.chf(time_interval, xi)
        if verbose:
            print("params:", model.get_params(),
                  "during:", time_interval)
    return chf


if __name__ == "__main__":
    unittest.main()
