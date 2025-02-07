import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR
from fypy.calibrate.calibrate_multi_section_levy.PathHandler import PathHandler
from fypy.calibrate.calibrate_multi_section_levy.Color import COLOR
from fypy.market.loader.YahooFinanceSurfaceLoader import YahooFinanceLoader
from fypy.termstructures.DiscountCurve import InterpolatedDiscountCurve, DiscountCurve
from fypy.termstructures.ForwardCurve import ForwardCurve

from fypy.market.MarketSlice import (
    OTMStrikeFilter,
    StrikeFilters,
    MidPriceFilter,
    RelativeStrikeFilter,
    MarketSlice,
)
from fypy.market.MarketSurface import MarketSlice, MarketSurface, SliceFilter

from fypy.volatility.implied.ImpliedVolCalculator import ImpliedVolCalculator_Black76

# TODO surfaces handler disc_curve data_path


class FilterTTM(SliceFilter):
    def __init__(self, min_ttm, max_ttm):
        self._min_ttm = min_ttm
        self._max_ttm = max_ttm

    def keep(self, market_slice: MarketSlice) -> bool:
        if market_slice.T < self._min_ttm:
            return False
        if market_slice.T > self._max_ttm:
            return False
        return True


class MarketInfo(PathHandler):
    def __init__(
        self, disc_path: str, data_paths: dict, _verbose=True, OTM: bool = True
    ):
        super().__init__(disc_path=disc_path, data_paths=data_paths)
        self._verbose = _verbose
        # row surfaces
        self.disc_curve = self._get_disc()
        # filtered surfaces
        self._filters = self._get_filters()
        self.surfaces = self._get_surfaces()
        # maturities
        self.tickers = self._get_tickers()
        self.fwds = self._get_fwds()

        self._maturities = self._get_maturities_days()
        # Implied volatilities
        self._ivcs = self._get_ivcs()
        self._fill_iv()

    def _get_surfaces(self) -> dict[str:MarketSurface]:  # -> Surface
        if self._verbose:
            print("* Getting surfaces:")
        surfaces = {}
        for ticker, ticker_path in self._data_paths.items():
            loader = YahooFinanceLoader()
            surface = loader.load_from_file(
                fpath=ticker_path, fit_discount=False, disc_curve=self.disc_curve
            )
            filt_surfaces = surface.filter_slices(
                slice_filter=self._filters["slice"],
                strike_filter=self._filters["strikes"],
            )
            surfaces[ticker] = filt_surfaces
        if self._verbose:
            COLOR.write("-done!", color="GREEN", indent=1)
        return surfaces

    def _get_filters(self):
        filter = {"slice": FilterTTM(0.01, 1)}
        filter["strikes"] = StrikeFilters(
            filters=[
                OTMStrikeFilter(),
                RelativeStrikeFilter(min_relative=0.2, max_relative=5),
                MidPriceFilter(min_price=0.01),
            ]
        )
        return filter

    def _get_disc(self) -> DiscountCurve:
        if self._verbose:
            print("* Getting discount curve:")
        discs = pd.read_csv(self._disc_path, sep=";")
        disc_curve = InterpolatedDiscountCurve.from_log_linear(
            ttms=discs["ttm"].values, discounts=discs["discount"].values
        )
        if self._verbose:
            COLOR.write("-done!", color="GREEN", indent=1)
        return disc_curve

    def _get_maturities_days(self) -> dict[int:float]:
        mat = {}
        for ticker in self.tickers:
            surface = self.surfaces.get(ticker)
            mat[ticker] = {int(ttm * 365.25): ttm for ttm in surface.slices}
        return mat

    def _get_tickers(self) -> list:
        tickers = list(self.surfaces.keys())
        return tickers

    def _get_fwds(self) -> dict[str:ForwardCurve]:
        fwds = {ticker: self.surfaces[ticker].forward_curve for ticker in self.tickers}
        return fwds

    def _get_ivcs(self):
        ivcs = {
            ticker: ImpliedVolCalculator_Black76(
                disc_curve=self.disc_curve, fwd_curve=self.fwds[ticker]
            )
            for ticker in self.tickers
        }
        return ivcs

    def _fill_iv(self):
        if self._verbose:
            print("* Filling Implied Volatility surfaces:")
        for ticker in self.tickers:
            self.surfaces[ticker].fill_implied_vols(self._ivcs[ticker])

        if self._verbose:
            COLOR.write("-done!", color="GREEN", indent=1)

    def get_iv_from_prices(
        self, ticker: str, strikes: np.array, prices: np.array, ttmy: float
    ):
        ivs = self._ivcs[ticker].imply_vols(
            strikes=strikes,
            prices=prices,
            is_calls=np.ones(len(strikes), dtype=bool),
            ttm=ttmy,
        )
        return ivs

    def _get_fwd(self, ticker: str, ttmy: float):
        return self.fwds[ticker].fwd_T(ttmy)

    def _get_strikes(self, ticker: str, ttmy: float):
        return self.surfaces[ticker].slices[ttmy].strikes

    def _get_log_strikes(self, ticker: str, ttmy: float):
        strikes = self._get_strikes(ticker, ttmy)
        fwd = self._get_fwd(ticker, ttmy)
        return np.log(strikes / fwd)

    def _get_strikes_th(self, ticker: float, ttmy: float):
        strikes_market = self.surfaces[ticker].slices[ttmy].strikes
        bounds = [min(strikes_market), max(strikes_market)]
        strikes_th = np.arange(bounds[0] * 0.5, bounds[1] * 1.5, 2)
        fwd = self._get_fwd(ticker, ttmy)
        log_mon = np.log(strikes_th / fwd)
        return strikes_th, log_mon
