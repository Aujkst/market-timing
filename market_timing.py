import datetime as dt
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.simplefilter('ignore', UserWarning)

from qdb import QdbApi
import qdb.util.time_func as time_util

class MarketTiming:
    def __init__(
        self, 
        bmk_id='000300.CSI',
        history_start_date=None,
        history_end_date=None,
        test_start_date=dt.date(2019, 1, 2),
        test_end_date=dt.date(2022, 5, 31)
    ):
        q = QdbApi()
        self.bmk_id = bmk_id
        self.test_start_date = time_util.str2dtdate(test_start_date) if type(test_start_date) == str else test_start_date
        self.history_start_date = time_util.str2dtdate(history_start_date) if type(history_start_date) == str else history_start_date
        self.history_end_date = time_util.str2dtdate(history_end_date) if type(history_end_date) == str else history_end_date
        self.history_end_date = q.prev_tdate(self.test_start_date) if history_end_date is None else history_end_date
        assert self.history_end_date < self.test_start_date
        self.test_end_date = time_util.str2dtdate(test_end_date) if type(test_end_date) == str else test_end_date
        self.bmk_name = q.get_index_list(self.bmk_id).squeeze()['full_name']
        self.bmk_daily = q.get_index_daily(self.bmk_id)
        self.bmk_daily.set_index("date", inplace=True)
        print('=================================')
        print(f'对股指：{self.bmk_name}进行择时信号的评估与回测，\n从{self.test_start_date}到{self.test_end_date}的数据用于外推效果验证')
        self.backtest_result = {}

    def signal_input(self, original_signal, window_size=50):
        self.original_signal = original_signal
        self.original_signal.index.name = 'date'
        if self.history_start_date is None or self.history_start_date < self.original_signal.index[0]:
            self.history_start_date = self.original_signal.index[0]
        self.window_size = window_size
        print(f'对择时信号进行滑动窗口标准化，winndow_size = {self.window_size}\n得到预测序列fcst')
        self.fcst = self._standard_scaler(self.original_signal, self.window_size)
        self.fcst.name = 'fcst'
        print(self.fcst[self.history_start_date:self.history_end_date].describe())
        self.plot_fcst(self.history_start_date, self.history_end_date)
    
    def _standard_scaler(self, X, window_size):
        X_rolling = X.rolling(window_size)
        mu = X_rolling.mean()
        sigma = X_rolling.std()
        return (X - mu) / sigma

    def _adf_test(self, X):
        return sm.tsa.stattools.adfuller(X.dropna())[1]

    def plot_fcst(self, start_date, end_date):
        fcst = self.fcst[start_date: end_date]
        adf_test = self._adf_test(fcst)
        fcst.plot(figsize=[16, 5], grid=True, label=f'fcst (P-value (ADF test) = {adf_test:.4f})')
        plt.legend()
        plt.show()

    def dot_product_analysis(self):
        # 内积评估
        df = pd.DataFrame(self.fcst[self.history_start_date:self.history_end_date])
        df["open"], df["close"] = self.bmk_daily["open"], self.bmk_daily["close"]

        df["ret(1)"] = df["close"].shift(-1) / df["open"].shift(-1) - 1
        df["ret(5)"] = (df["close"].shift(-5) / df["open"].shift(-1) - 1) / 5
        df["ret(10)"] = (df["close"].shift(-10) / df["open"].shift(-1) - 1) / 10
        df["ret(21)"] = (df["close"].shift(-21) / df["open"].shift(-1) - 1) / 21
        df["ret(63)"] = (df["close"].shift(-63) / df["open"].shift(-1) - 1) / 63

        df["ret(2,5)"] = (df["close"].shift(-5) / df["open"].shift(-2) - 1) / 4
        df["ret(6,10)"] = (df["close"].shift(-10) / df["open"].shift(-6) - 1) / 5
        df["ret(11,21)"] = (df["close"].shift(-21) / df["open"].shift(-11) - 1) / 11
        df["ret(22,63)"] = (df["close"].shift(-63) / df["open"].shift(-22) - 1) / 41
        self._dot_product_df = df
        self.plot_dot_product_result(mode='cum')
        self.plot_dot_product_result(mode='seg')

    def plot_dot_product_result(self, mode='cum'):
        # cum: 累积时间间隔; seg: 不重叠时间间隔
        assert mode in ['cum', 'seg']
        if mode == 'cum':
            ret_cols = ["ret(1)", "ret(5)", "ret(10)", "ret(21)", "ret(63)"] 
        else: 
            ret_cols = ["ret(1)", "ret(2,5)", "ret(6,10)", "ret(11,21)", "ret(22,63)"]
        df = self._dot_product_df
        fig, (ax_0, ax_1, ax_2, ax_3) = plt.subplots(nrows=4, ncols=1, figsize=(16, 20), dpi=200)
        for col in ret_cols:
            ax_0.plot((df["fcst"] * df[col]).cumsum(), label=f"$fcst^T \cdot {col}$")
        ax_0.grid()
        ax_0.legend()

        for threshold, ax_i in zip([0, 0.5, 1], [ax_1, ax_2, ax_3]):
            short_ret_mean = [df.loc[df['fcst']<threshold, col].mean() for col in ret_cols]
            long_ret_mean = [df.loc[df['fcst']>threshold, col].mean() for col in ret_cols]
            ind = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
            ax_i.bar(
                x=ind, 
                height=short_ret_mean[::-1] + long_ret_mean,
                width=0.5,
            )
            ax_i.set_xticks(ind)
            ax_i.set_xticklabels([f"$fcst<{threshold}$\n{col})" for col in ret_cols][::-1] + [f"$fcst>{threshold}$\n{col})" for col in ret_cols])
            ax_i.spines["top"].set_color("none")
            ax_i.spines["right"].set_color("none")
            ax_i.spines["left"].set_position(("data", 0))
            ax_i.axhline(y=0, xmin=min(ind), xmax=max(ind), color='black')
            ax_i.grid(axis='y')

        fig.show()

    def corr_analysis(self):
        # 目前只支持计算与动量因子的相关性
        start_date = self.history_start_date
        end_date = self.history_end_date
        
        mtm = self.bmk_daily['close'].pct_change(20)
        mtm = self._standard_scaler(mtm, window_size=60)
        mtm.name = 'mtm'
        corr_df = self.fcst[start_date: end_date].to_frame().join(mtm)
        corr_df = corr_df.dropna()
        corr, p_value = stats.pearsonr(corr_df['fcst'], corr_df['mtm'])
        self._corr_df = corr_df

        fig, ax = plt.subplots(figsize=(16, 5))
        ax.grid()
        ax.plot(corr_df['fcst'], color="#752100", label='fcst')
        ax.set_ylabel('fcst')
        ax.legend(loc="upper left")

        ax_twinx = ax.twinx()
        ax_twinx.plot(corr_df['mtm'], color="#d8a373", label='mtm')
        ax_twinx.set_ylabel('mtm')
        ax_twinx.legend(loc="upper right")

        ax.set_title(f"Pearson's r = {corr:.4f} and p-value = {p_value:.4f}")
        fig.show()
    
    def backtest(
        self, long_threshold=1, long_start=1, long_period=1,
        short_threshold=None, short_start=None, short_period=None,
        mode='history'
    ):
        assert mode in ['history', 'test', 'all']
        short_threshold = -long_threshold if short_threshold is None else short_threshold
        short_start = long_start if short_start is None else short_start
        short_period = long_period if short_period is None else short_period

        if mode == 'history':
            start_date, end_date = self.history_start_date, self.history_end_date
        elif mode == 'test':
            start_date, end_date = self.test_start_date, self.test_end_date
        elif mode == 'all':
            start_date, end_date = self.history_start_date, self.test_end_date
        backtest_df = self.bmk_daily.loc[start_date:end_date].copy()
        dates_list = backtest_df.index.tolist()
        backtest_df['pos_long'], backtest_df['pos'] = 0, 0
        for i, idx in enumerate(dates_list):
            fcst_i = self.fcst[idx] if idx in self.fcst.index else None
            if fcst_i is None:
                continue
            if fcst_i > long_threshold:
                for h in range(long_period):
                    if i + long_start + h >= len(dates_list):
                        break
                    long_date = dates_list[i+long_start+h]
                    backtest_df.loc[long_date, 'pos_long'] = 1
                    backtest_df.loc[long_date, 'pos'] = 1
            elif fcst_i < short_threshold:
                for h in range(short_period):
                    if i + short_start + h >= len(dates_list):
                        break
                    short_date = dates_list[i+short_start+h]
                    backtest_df.loc[short_date, 'pos'] = -1

        backtest_df['overnight_ret'] = backtest_df['open'] / backtest_df['close'].shift(1) - 1
        backtest_df.loc[backtest_df.index[0], 'overnight_ret'] = 0
        backtest_df['daytime_ret'] = backtest_df['close'] / backtest_df['open'] - 1

        backtest_df['stgy_ret'] = (1 + backtest_df["pos"].shift(1) * backtest_df['overnight_ret']) * (1 + backtest_df["pos"] * backtest_df['daytime_ret']) - 1
        backtest_df['stgy_long_ret'] = (1 + backtest_df["pos_long"].shift(1) * backtest_df['overnight_ret']) * (1 + backtest_df["pos_long"] * backtest_df['daytime_ret']) - 1
        backtest_df.loc[backtest_df.index[0], 'stgy_ret'] = 0
        backtest_df.loc[backtest_df.index[0], 'stgy_long_ret'] = 0
        backtest_df['stgy_nav'] = (1 + backtest_df["stgy_ret"]).cumprod()
        backtest_df['stgy_long_nav'] = (1 + backtest_df["stgy_long_ret"]).cumprod()
        backtest_df['stgy'] = backtest_df['stgy_nav'] * backtest_df.loc[start_date, 'open']
        backtest_df['stgy_long'] = backtest_df['stgy_long_nav'] * backtest_df.loc[start_date, 'open']
        self.backtest_result[mode] = backtest_df

        backtest_perf = self.cal_period_perf_indicator(backtest_df[['close', 'stgy_long', 'stgy']])
        backtest_eval = self.backtest_evaluate(mode=mode)
        display(backtest_perf)
        display(backtest_eval)
        self.plot_backtest_result(backtest_df, self.fcst)

    def plot_backtest_result(self, df, fcst):
        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(3,1,1)
        color_dict = {'close': '#CE9461', 'stgy': '#5534A5', 'stgy_long': '#A85CF9'}
        for col, color in color_dict.items():
            df[col].plot.line(ax=ax1, color=color)
        ax1.set_title('price')
        ax1.grid()
        ax1.legend()
        ax2 = fig.add_subplot(3,1,2)
        df.loc[:, 'pos'].plot(ax=ax2, grid=True, title='pos')
        ax3 = fig.add_subplot(3,1,3)
        fcst[df.index[0]: df.index[-1]].plot(ax=ax3, grid=True, title='fcst')

    def backtest_evaluate(self, mode='history'):
        assert mode in ['history', 'test', 'all']
        # 做多/空日期
        df_bt = self.backtest_result[mode]
        df_bt_eval = pd.DataFrame(index=['做多做空', '只做多'], columns=['总胜率', '看多胜率', '看空胜率'])

        # 只做多
        long_dates = df_bt[df_bt["pos_long"] == 1].index.to_list()
        df_bt_eval.loc['只做多', '看多胜率'] = (df_bt.loc[long_dates, "stgy_ret"] > 0).sum() / len(long_dates)
        df_bt_eval.loc['只做多', '总胜率'] = df_bt_eval.loc['只做多', '看多胜率']

        # 做多做空
        short_dates = df_bt[df_bt["pos"] == -1].index.to_list()
        long_dates = df_bt[df_bt["pos"] == 1].index.to_list()
        df_bt_eval.loc['做多做空', '总胜率'] = (df_bt.loc[short_dates+long_dates, "stgy_ret"] > 0).sum() / len(short_dates+long_dates)
        df_bt_eval.loc['做多做空', '看空胜率'] = (df_bt.loc[short_dates, "stgy_ret"] > 0).sum() / len(short_dates)
        df_bt_eval.loc['做多做空', '看多胜率'] = (df_bt.loc[long_dates, "stgy_ret"] > 0).sum() / len(long_dates)
        return df_bt_eval

    def cal_period_perf_indicator(self, df):
        """
        计算区间业绩指标(高级版)
        Input
            df: 价格或净值序列，DataFrame, index是datetime.date，每列是一只基金
        Output
            ['AnnRet1', 'AnnRet2', 'AnnVol', 'SR', 'MaxDD', 'Calmar'] 
        """

        assert type(df)==pd.DataFrame
        assert type(df.index[0])==dt.date
        
        indicators = ['AnnRet', 'AnnRet_Simple', 'AnnVol', 'SR', 'MaxDD', 'Calmar']
        res = pd.DataFrame(index=df.columns, columns=indicators)
        date_ordinal = pd.Series([dt.date.toordinal(e) for e in df.index])
        time_diff = date_ordinal - date_ordinal.shift(1) # 相对上一次净值的日期间隔
        for col in df:    
            p = df[col] # 单个资产价格或净值序列, pd.Series
            r = p.pct_change() # 涨幅
            annret = (p[-1] / p[0]) ** (365/(p.index[-1]-p.index[0]).days) - 1 # 复利年化收益
            r1 = r.values / time_diff # 日均收益
            annret1 = np.nanmean(r1) * 365 # 单利年化收益
            r2 = r.values / np.sqrt(time_diff) # 波动率调整后涨幅   
            annvol = np.nanstd(r2) * np.sqrt(365) # 年化波动率
            sr = (annret - 0.025) / annvol # 夏普比率
            mdd = np.min(p/p.cummax() - 1) # 最大回撤
            calmar = annret / -mdd
            res.loc[col] = [annret, annret1, annvol, sr, mdd, calmar]

        return res