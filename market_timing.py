import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from qdb import QdbApi

class MarketTiming(QdbApi):
    def __init__(
        self, 
        bmk_id='000300.CSI', 
        start_date=dt.date(2015, 1, 5), 
        end_date=dt.date(2018, 12, 28),
        backtest_end_date=dt.date(2022, 5, 5),
        user=None, password=None
    ):
        super().__init__(user=user, password=password)
        
        self.bmk_id = bmk_id
        self.start_date = time_util.str2dtdate(start_date) if type(start_date) == str else start_date
        self.end_date = time_util.str2dtdate(end_date) if type(end_date) == str else end_date
        self.backtest_start_date = self.next_tdate(self.end_date)
        self.backtest_end_date = time_util.str2dtdate(backtest_end_date) if type(backtest_end_date) == str else backtest_end_date
        self.bmk_daily = self.get_index_daily(self.bmk_id, self.start_date, self.end_date)
        self.bmk_daily.set_index("date", inplace=True)

    def signal(self, input_signal, window_size=50):
        self.original_signal = input_signal
        self.original_signal.index.name = 'date'
        self.window_size = window_size
        print(f'对择时信号进行滑动窗口标准化，winndow_size = {self.window_size}\n得到预测序列fcst')
        self.fcst = self._standard_scaler(self.original_signal, self.window_size)
        self.fcst.name = 'fcst'
        print(self.fcst.describe())
    
    def _standard_scaler(self, X, window_size):
        X_rolling = X.rolling(window_size)
        mu = X_rolling.mean()
        sigma = X_rolling.std()
        return (X - mu) / sigma

    def _adf_test(self, X):
        return sm.tsa.stattools.adfuller(X.dropna())[1]

    def plot_fcst(self):
        fcst = self.fcst
        adf_test = self._adf_test(fcst)
        fcst.plot(figsize=[16, 5], grid=True, label=f'fcst (P-value (ADF test) = {adf_test:.4f})')
        plt.legend()
        plt.show()

    def dot_product_analysis(self):
        df = pd.DataFrame(self.fcst)
        df["open"], df["close"] = self.bmk_daily["open"], self.bmk_daily["close"]

        df["ret_1"] = df["close"].shift(-1) / df["open"].shift(-1) - 1
        df["ret_5"] = (df["close"].shift(-5) / df["open"].shift(-1) - 1) / 5
        df["ret_10"] = (df["close"].shift(-10) / df["open"].shift(-1) - 1) / 10
        df["ret_21"] = (df["close"].shift(-21) / df["open"].shift(-1) - 1) / 21
        df["ret_63"] = (df["close"].shift(-63) / df["open"].shift(-1) - 1) / 63

        df["ret_2_5"] = (df["close"].shift(-5) / df["open"].shift(-2) - 1) / 4
        df["ret_6_10"] = (df["close"].shift(-10) / df["open"].shift(-6) - 1) / 5
        df["ret_11_21"] = (df["close"].shift(-21) / df["open"].shift(-11) - 1) / 11
        df["ret_22_63"] = (df["close"].shift(-63) / df["open"].shift(-22) - 1) / 41
        self._dot_product_df = df
        self.plot_dot_product_result(mode='cum')
        self.plot_dot_product_result(mode='seg')

    def plot_dot_product_result(self, mode='cum'):
        # cum: 累积时间间隔; seg: 不重叠时间间隔
        assert mode in ['cum', 'seg']
        df = self._dot_product_df
        fig, (ax_0, ax_1) = plt.subplots(nrows=2, ncols=1, figsize=(16, 10), dpi=200)
        
        if mode == 'cum':
            ret_cols = ["ret_1", "ret_5", "ret_10", "ret_21", "ret_63"]
            for col in ret_cols:
                _, h = col.split('_')
                ax_0.plot((df["fcst"] * df[col]).cumsum(), label=f"$fcst^T \cdot ret({h})$")
        elif mode == 'seg':
            ret_cols = ["ret_1", "ret_2_5", "ret_6_10", "ret_11_21", "ret_22_63"]
            for col in ret_cols:
                if col != "ret_1":
                    _, h_1, h_2 = col.split('_')
                    ax_0.plot((df["fcst"] * df[col]).cumsum(), label=f"$fcst^T \cdot ret(t+{h_1}, t+{h_2})$")
                else:
                    ax_0.plot((df["fcst"] * df[col]).cumsum(), label=f"$fcst^T \cdot ret_1$")

        ax_0.grid()
        ax_0.legend()

        short_mean_ret = [df.loc[df['fcst']<0, col].mean() for col in ret_cols]
        long_mean_ret = [df.loc[df['fcst']>0, col].mean() for col in ret_cols]

        xticks = np.append(-np.linspace(1, 5, 5), np.linspace(1, 5, 5))
        ax_1.bar(
            x=xticks, 
            height=short_mean_ret+long_mean_ret,
            width=0.5
        )
        ax_1.spines["top"].set_color("none")
        ax_1.spines["right"].set_color("none")
        ax_1.spines["bottom"].set_position(("data", 0))
        ax_1.spines["left"].set_position(("data", 0))
        ax_1.grid(axis='y')
        ax_1.set_xticks(
            xticks,
            ["$fcst<0$\n" + col for col in ret_cols] + ["$fcst>0$\n" + col for col in ret_cols]
        )
        fig.show()