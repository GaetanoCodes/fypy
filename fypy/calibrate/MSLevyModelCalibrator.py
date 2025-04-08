from typing import Optional, Tuple, Dict, Callable, Any
import numpy as np

from fypy.calibrate.FourierModelCalibrator import FourierModelCalibrator
from fypy.calibrate.utils.TargetsWithSmallPriceErr import TargetsWithSmallPriceErr
from fypy.market.MarketSlice import MarketSlice
from fypy.market.MarketSurface import MarketSurface
from fypy.model.levy.LevyModel import LevyModel
from fypy.model.sv.Heston import _HestonBase
from fypy.pricing.fourier.ProjPricer import ProjPricer
from fypy.pricing.fourier.ProjEuropeanPricer import ProjEuropeanPricer
from fypy.fit.Minimizer import LeastSquares, Minimizer
from fypy.calibrate.calibrate_multi_section_levy.MarketInfo import FilterTTM

class MSLevyModelCalibrator(FourierModelCalibrator):
    def __init__(self,
                 surface: MarketSurface,
                 minimizer: Optional[Minimizer] = None,
                 reduced_surface_range: Tuple= (0.02, 0.085),
                 do_vega_weight: bool = True,
                 tol_single_slice:float =1e-08,
                 max_nfev_single_slice:int=1000,
                 alpha_pricer: float= 20):
        super(MSLevyModelCalibrator, self).__init__(surface=surface, minimizer=minimizer, do_vega_weight=do_vega_weight)
        self._reduced_surface_range= reduced_surface_range
        self._single_slice_minimizer= LeastSquares(max_nfev=max_nfev_single_slice, ftol=tol_single_slice, xtol=tol_single_slice, gtol=tol_single_slice, verbose=0)
        self._alpha_pricer=alpha_pricer




    def calibrate(self, model: LevyModel, pricer: Optional[ProjPricer] = None)-> Tuple:

        result, pricer= self._calibrate_reduced_surface(model=model, pricer=pricer)
        remaining_slices = {maturity: market_slice for maturity, market_slice in self.surface.slices.items() if maturity > self._reduced_surface_range[1]}

        # Second stage: calibrate remaining maturities with lower precision
        self._multiple_slice_calibration(model, pricer, remaining_slices)

        # Last slice is calibrated separately without updating frozen parameters
        slices_list = list(self.surface.slices.items())
        maturity, market_slice = slices_list[-1]
        result, pricer = self._single_slice_calibration(model, maturity, market_slice, pricer, update_frozen_values=False)



        return result, pricer, pricer.get_model().get_frozen_params()





    def _multiple_slice_calibration(self, model: LevyModel, pricer: ProjPricer, slices: Dict)-> None:
        """ Calibrates a set of market slices, updating frozen parameters accordingly. """
        self._minimizer=self._single_slice_minimizer
        for maturity, market_slice in slices.items():
            _, _ = self._single_slice_calibration(model, maturity, market_slice, pricer)

    def _calibrate_generic(self,
                                model: LevyModel,
                                pricer: ProjPricer,
                                target_preparation_fn: Callable[
                                    [ProjPricer], Tuple[np.ndarray, np.ndarray, Callable[[], np.ndarray]]],
                                update_model_fn: Optional[Callable[[LevyModel, Any], None]] = None
                                ) -> Tuple[Any, ProjPricer]:
        """
        Calibrates a model using a flexible target preparation and update logic.

        Parameters:
            model: The model to calibrate.
            pricer: A pre-configured pricer (should not be None).
            target_preparation_fn: Function that returns (target_prices, weights, targets_pricer).
            update_model_fn: Function to update the model after calibration (optional).

        Returns:
            Tuple of (CalibrationResult, pricer)
        """
        target_prices, weights, targets_pricer = target_preparation_fn(pricer)

        calibrator = self._configure_calibrator(
            model=model,
            target_prices=target_prices,
            targets_pricer=targets_pricer,
            weights=weights
        )

        result = self._calibrate(calibrator)

        if update_model_fn is not None:
            update_model_fn(pricer.get_model(), result)

        return result, pricer

    def _calibrate_reduced_surface(self, model: LevyModel, pricer: Optional[ProjPricer] = None):
        self._reduced_surface = self._filter_market_surface_by_maturity(market_surface=self.surface)

        if pricer is None:
            pricer = ProjEuropeanPricer(
                model=model,
                N=2 ** 13,
                L=20 if isinstance(model, _HestonBase) else 15,
                alpha_override=self._alpha_pricer
            )
        else:
            pricer.set_alpha_override(alpha_override=self._alpha_pricer)

        def prepare_targets(p: ProjPricer):
            return self._prepare_targets_and_pricer_surface(surface=self._reduced_surface, pricer=p)

        def update_model(model: LevyModel, result: Any):
            model.set_multi_section(multi_section=True)
            model.update_frozen_values(
                T=self._reduced_surface_range[1],
                parameters=result.params,
                alph=pricer.get_alpha_override(),
                N=pricer.get_N()
            )

        return self._calibrate_generic(model=model, pricer=pricer,
                                            target_preparation_fn=prepare_targets,
                                            update_model_fn=update_model)

    def _single_slice_calibration(self, model: LevyModel, maturity: float,
                                  market_slice: MarketSlice, pricer: ProjPricer,
                                  update_frozen_values: bool = True):
        """
        Performs calibration for a single tenor using an already initialized pricer.
        """

        def prepare_targets(p: ProjPricer):
            return self._prepare_targets_and_pricer_slice(maturity=maturity, market_slice=market_slice, pricer=p)

        def update_model(model: LevyModel, result):
            if update_frozen_values:
                model.update_frozen_values(
                    T=maturity,
                    parameters=result.params,
                    alph=pricer.get_alpha_override(),
                    N=pricer.get_N()
                )

        return self._calibrate_generic(model=model, pricer=pricer,
                                            target_preparation_fn=prepare_targets,
                                            update_model_fn=update_model)

    def _make_targets_and_pricer_generic(
            self,
            get_targets_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
            pricing_fn: Callable[[np.ndarray], None]
    ):

        target_prices, weights = get_targets_fn()
        all_prices = np.empty_like(target_prices, dtype=float)

        def targets_pricer() -> np.ndarray:
            try:
                pricing_fn(all_prices)
            except Exception as e:
                print(f'Error getting prices, filling with NaN: {e}')
                all_prices[:] = np.nan
            return all_prices

        return target_prices, weights, targets_pricer

    def _prepare_targets_and_pricer_slice(self, maturity: float, market_slice: MarketSlice, pricer: ProjPricer):
        def get_targets():
            return self._make_all_targets_MSlevy(market_slice=market_slice)

        def pricing_fn(output: np.ndarray):
            pricer.price_strikes_fill(
                T=maturity,
                K=market_slice.strikes,
                is_calls=market_slice.is_calls,
                output=output
            )

        return self._make_targets_and_pricer_generic(get_targets_fn=get_targets, pricing_fn=pricing_fn)

    def _prepare_targets_and_pricer_surface(self, surface: MarketSurface, pricer: ProjPricer):
        def get_targets():
            return self._make_all_targets(surface=surface)

        def pricing_fn(output: np.ndarray):
            left = 0
            for ttm, market_slice in surface.slices.items():
                num_strikes = len(market_slice.strikes)
                pricer.price_strikes_fill(
                    T=ttm,
                    K=market_slice.strikes,
                    is_calls=market_slice.is_calls,
                    output=output[left:left + num_strikes]
                )
                left += num_strikes

        return self._make_targets_and_pricer_generic(get_targets_fn=get_targets, pricing_fn=pricing_fn)

    def _configure_calibrator(self, model: LevyModel, target_prices: np.ndarray, targets_pricer: Callable,
                              weights: np.ndarray):

        # Initialize the calibrator for the model
        calibrator = self._init_calibrator(model=model)

        # Define the small price penalty multiplier based on the model type
        small_price_penalty_mult = 10 if isinstance(model, LevyModel) else 3

        # Create targets for the calibrator
        targets = TargetsWithSmallPriceErr(target_prices, targets_pricer, weights=weights,
                                           small_price_penalty_mult=small_price_penalty_mult)
        calibrator.add_objective("Targets", targets)

        return calibrator


    def _make_all_targets_MSlevy(self, market_slice: MarketSlice=None) -> Tuple[np.ndarray, np.ndarray]:
        target_prices = []
        weights = []

        # push back the target prices to fit to
        target_prices.append(market_slice.mid_prices)

        # Use inverse vega weighting
        weights.append(self._make_weights(market_slice))

        # Full set of market target prices
        target_prices = np.concatenate(target_prices)
        weights = np.concatenate(weights)

        return target_prices, weights




    def _filter_market_surface_by_maturity(self, market_surface: MarketSurface) -> MarketSurface:
        """
        Filters a MarketSurface to keep only slices with maturity up to max_maturity.

        :param market_surface: MarketSurface, the original market surface.
        :return: Filtered MarketSurface.
        """
        maturity_filter = FilterTTM(min_ttm=self._reduced_surface_range[0], max_ttm=self._reduced_surface_range[1])
        return market_surface.filter_slices(slice_filter=maturity_filter)