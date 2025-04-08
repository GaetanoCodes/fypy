from fypy.calibrate.calibrate_multi_section_levy.Calibrator import Calibrator
from fypy.calibrate.calibrate_multi_section_levy.ResultReader2 import ResultDisplay
import os
import pandas as pd
from data.init_guess import init_guess
print(os.getcwd())

output_excel = f"results.xlsx"
with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    for model_name in ["CGMY", "BG","BGM", "NIG", "MJD", "KDE"]:
        for i in range(0,1):
            date = 20240220
            tickers = ["AMZN", "NFLX", "SHOP", "SPOT"]
            discount_path = f"data/USD_discounts_{date}.csv"
            data_path = {ticker: f"data/\{ticker}_{date}.csv" for ticker in tickers}
            calibrator = Calibrator([model_name], disc_path=discount_path, data_paths=data_path, init_guess=init_guess)
            calibrator.calibrate()
            print("\n\n\n\n\n\n\n\n\nfine calibrate examples")
            res_path = "calibration_saved/LS_CONS"
            res = ResultDisplay(res_path, discount_path, data_path)
            df = pd.read_parquet(os.path.join(res_path, "res.parquet"))
            df.to_excel(writer, sheet_name=f"{model_name}_{i}", index=False)
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", 1000)
            pd.set_option("display.max_colwidth", None)
            print(df)

