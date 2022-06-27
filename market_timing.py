import datetime as dt
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

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
        self.bmk_name = q.get_index_list(self.bmk_id).squeeze()['index_name']
        self.bmk_daily = q.get_index_daily(self.bmk_id)
        self.bmk_daily.set_index("date", inplace=True)

        print('=================================')
        print(f'对股指：{self.bmk_name}（{self.bmk_id}）进行择时信号的评估与回测')
        print(f'从{self.test_start_date}到{self.test_end_date}的数据用于外推效果验证')
        print('=================================')
        self.backtest_result = {}

    def add_signal(self, original_signal, window_size=200):
        self.original_signal = original_signal
        self.original_signal.index.name = 'date'
        if self.history_start_date is None or (self.history_start_date < self.original_signal.index[0]):
            self.history_start_date = self.original_signal.index[0]
        self.window_size_ = window_size
        print(f'对择时信号进行滑动窗口标准化，winndow_size = {self.window_size_}\n得到预测序列fcst')
        if self.window_size_ != 0:
            self.fcst = self._standard_scaler(self.original_signal, self.window_size_)
            self.fcst.name = 'fcst'
        else:
            self.fcst = self.original_signal.copy()
            self.fcst.name = 'fcst'
        print(self.fcst[self.history_start_date:self.history_end_date].describe())
        self.plot_fcst(self.history_start_date, self.history_end_date)
    
    def _standard_scaler(self, X, window_size):
        X_rolling = X.rolling(window_size, min_periods=1)
        mu = X_rolling.mean()
        sigma = X_rolling.std()
        sigma[X.index[0]] = 1
        return (X - mu) / sigma

    def _adf_test(self, X):
        return sm.tsa.stattools.adfuller(X.dropna())[1]

    def plot_fcst(self, start_date, end_date):
        fcst = self.fcst[start_date: end_date]
        adf_test = self._adf_test(fcst)
        fcst.plot(figsize=[16, 5], grid=True, label=f'fcst (P-value (ADF test) = {adf_test:.4f})')
        plt.legend()
        plt.show()

    def dot_product_analysis(self, cut_off_points=[0, 0.5, 1]):
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
        self.plot_dot_product_result(mode='cum', cut_off_points=cut_off_points)
        self.plot_dot_product_result(mode='seg', cut_off_points=cut_off_points)

    def plot_dot_product_result(self, cut_off_points=[0, 0.5, 1], mode='cum'):
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

        base_point = cut_off_points[0]
        for threshold, ax_i in zip(cut_off_points, [ax_1, ax_2, ax_3]):
            threshold_prime = 2*base_point - threshold
            short_ret_mean = [df.loc[df['fcst']<threshold_prime, col].mean() for col in ret_cols]
            long_ret_mean = [df.loc[df['fcst']>threshold, col].mean() for col in ret_cols]
            ind = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
            ax_i.bar(
                x=ind, 
                height=short_ret_mean[::-1] + long_ret_mean,
                width=0.5,
            )
            ax_i.set_xticks(ind)
            ax_i.set_xticklabels([f"$fcst<{threshold_prime:.2f}$\n{col}" for col in ret_cols][::-1] + [f"$fcst>{threshold:.2f}$\n{col}" for col in ret_cols])
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

    def generate_stgy(
        self, long_threshold=1, long_start=1, long_period=1,
        short_threshold=None, short_start=None, short_period=None, default_pos=0
    ):

        assert default_pos >= 0, '暂不支持负向偏移仓位'
        short_threshold = -long_threshold if short_threshold is None else short_threshold
        short_start = long_start if short_start is None else short_start
        short_period = long_period if short_period is None else short_period
        stgy_df = self.fcst.to_frame()
        fcst_index_list = stgy_df.index.tolist()
        stgy_df['pos'], stgy_df['pos_long'] = 0, 0
        for i, date in enumerate(fcst_index_list):
            fcst_i = self.fcst[date]
            if fcst_i > long_threshold:
                for h in range(long_period):
                    if i + long_start + h >= len(fcst_index_list):
                        break
                    long_date = fcst_index_list[i+long_start+h]
                    stgy_df.loc[long_date, 'pos_long'] = 1
                    stgy_df.loc[long_date, 'pos'] = 1
            elif fcst_i < short_threshold:
                for h in range(short_period):
                    if i + short_start + h >= len(fcst_index_list):
                        break
                    short_date = fcst_index_list[i+short_start+h]
                    stgy_df.loc[short_date, 'pos'] = -1

        stgy_df['pos'] = stgy_df['pos'].apply(lambda x: min(1, x+default_pos))
        stgy_df['pos_long'] = stgy_df['pos_long'].apply(lambda x: min(1, x+default_pos))
        self.stgy_df = stgy_df

    def evaluate_stgy(self, etf_id='510300#1', fut_id='IF'):
        self.etf_id = etf_id
        self.etf_name = q.get_etf_list(self.etf_id).squeeze()['name']
        self.etf_daily = q.get_etf_daily(self.etf_id, '1990-01-01', '2023-01-01')
        self.etf_daily.set_index("date", inplace=True)
        fut_prod_list = q.get_fut_prod_list()
        self.fut_id = fut_id
        self.fut_name = fut_prod_list[fut_prod_list['abbr']==self.fut_id].squeeze()['full_name']
        self.fut_daily = q.get_main_price(self.fut_id, fut_type='main')
        self.fut_daily.set_index("date", inplace=True)
        print(f'使用{self.etf_name}（{self.etf_id}）和{self.fut_name}（{self.fut_id}）进行实盘模拟')

        self.stgy_nav, self.stgy_long_nav = {}, {}
        self._simulate_stgy_nav(asset='index')
        self._simulate_stgy_nav(asset='etf')
        self._simulate_stgy_nav(asset='fut')
        print('胜率评估（相对基准指数）')
        self.cal_win_rate(period_type='history')
        self.cal_win_rate(period_type='test')
        self.cal_win_rate(period_type='all')
        self._show_stgy_result(stgy_type='pos')
        self._show_stgy_result(stgy_type='pos_long')


    def _simulate_stgy_nav(self, asset):
        assert asset in ['index', 'etf', 'fut']
        if asset == 'index':
            asset_df = self.bmk_daily[['open', 'close']]
            asset_name = f'{self.bmk_name}({self.bmk_id})'
        elif asset == 'etf':
            asset_df = self.etf_daily[['open', 'close']]
            asset_name = f'{self.etf_name}({self.etf_id})'
        else:
            asset_df = self.fut_daily[['open', 'close']]
            asset_name = f'{self.fut_name}({self.fut_id})'
            
        smlt_df = self.stgy_df[['pos', 'pos_long']].copy()
        smlt_df = smlt_df.loc[self.history_start_date:self.test_end_date]
        smlt_df = smlt_df.merge(asset_df, how='left', left_index=True, right_index=True)
        smlt_df['overnight_ret'] = smlt_df['open'] / smlt_df['close'].shift(1) - 1
        smlt_df.loc[smlt_df.index[0], 'overnight_ret'] = 0
        smlt_df['daytime_ret'] = smlt_df['close'] / smlt_df['open'] - 1
        smlt_df['stgy_ret'] = (1 + smlt_df["pos"].shift(1) * smlt_df['overnight_ret']) * (1 + smlt_df["pos"] * smlt_df['daytime_ret']) - 1
        smlt_df['stgy_long_ret'] = (1 + smlt_df["pos_long"].shift(1) * smlt_df['overnight_ret']) * (1 + smlt_df["pos_long"] * smlt_df['daytime_ret']) - 1
        smlt_df.loc[smlt_df.index[0], 'stgy_ret'] = 0
        smlt_df.loc[smlt_df.index[0], 'stgy_long_ret'] = 0
        smlt_df['stgy_nav'] = (1 + smlt_df["stgy_ret"]).cumprod()
        smlt_df['stgy_long_nav'] = (1 + smlt_df["stgy_long_ret"]).cumprod()
        self.stgy_nav[asset_name] = smlt_df['stgy_nav']
        self.stgy_long_nav[asset_name] = smlt_df['stgy_long_nav']

    def cal_win_rate(self, period_type='history'):
        assert period_type in ['history', 'test', 'all']
        if period_type == 'history':
            period_name = '历史训练时期'
            start_date, end_date = self.history_start_date, self.history_end_date
        elif period_type == 'test':
            period_name = '外推测试时期'
            start_date, end_date = self.test_start_date, self.test_end_date
        elif period_type == 'all':
            period_name = '全部'
            start_date, end_date = self.history_start_date, self.test_end_date
        # 做多/空日期
        df_bt = self.stgy_df.loc[start_date:end_date, ['pos', 'pos_long']]
        df_bt['bmk_ret'] = self.bmk_daily['close'].pct_change()
        df_bt_eval = pd.DataFrame(index=['做多做空', '只做多'], columns=['总胜率', '看多胜率', '看空胜率'])

        # 只做多
        long_dates = df_bt[df_bt["pos_long"] == 1].index.to_list()
        df_bt_eval.loc['只做多', '看多胜率'] = (df_bt.loc[long_dates, "bmk_ret"] > 0).sum() / len(long_dates)
        df_bt_eval.loc['只做多', '总胜率'] = df_bt_eval.loc['只做多', '看多胜率']

        # 做多做空
        short_dates = df_bt[df_bt["pos"] < 0].index.to_list()
        long_dates = df_bt[df_bt["pos"] == 1].index.to_list()
        df_bt_eval.loc['做多做空', '总胜率'] = (df_bt.loc[short_dates+long_dates, "bmk_ret"] > 0).sum() / len(short_dates+long_dates)
        df_bt_eval.loc['做多做空', '看空胜率'] = (df_bt.loc[short_dates, "bmk_ret"] > 0).sum() / len(short_dates)
        df_bt_eval.loc['做多做空', '看多胜率'] = (df_bt.loc[long_dates, "bmk_ret"] > 0).sum() / len(long_dates)
        print(f'period = {period_name}, 从{start_date}到{end_date}')
        display(df_bt_eval)
        
    def _show_stgy_result(self, stgy_type='pos'):
        assert stgy_type in ['pos', 'pos_long']
        eval_df = pd.concat(self.stgy_nav, axis=1) if stgy_type=='pos' else pd.concat(self.stgy_long_nav, axis=1)
        index_name = self.bmk_name
        eval_df[index_name] = self.bmk_daily['close'].copy()
        # 首一化和计算价格比
        stgy_names = eval_df.columns.tolist()[:-1]
        eval_df[index_name] = eval_df[index_name] / eval_df[index_name][0] # 基准指数首一化
        for stgy_name in stgy_names:
            t0 = eval_df.index[~eval_df[stgy_name].isna()][0] # 第一个有效值index
            eval_df[stgy_name] = eval_df[stgy_name] / eval_df[stgy_name][t0] * eval_df[index_name][t0] # 首一化【对齐到基准指数】
            eval_df[stgy_name+'/'+index_name] = eval_df[stgy_name] / eval_df[index_name]
        price_ratio_names = [d+'/'+index_name for d in stgy_names]
        eval_df.ffill(inplace=True)
        pr_df = eval_df[[index_name]+stgy_names+price_ratio_names].copy()
        result = self.cal_period_perf_indicator(pr_df)
        for col in result:
            if col in ['SR', 'Calmar']:
                result[col] = result[col].apply(lambda x: format(x, '.3') if not pd.isna(x) else '')
            else:
                result[col] = result[col].apply(lambda x: format(x, '.2%') if not pd.isna(x) else '')
        display(result)
        
        fig = plt.figure(figsize=(16, 12))
        ax1 = fig.add_subplot(3,1,1)
        # color_dict = {index_name: '#CE9461', 'stgy_nav': '#5534A5', 'stgy_long_nav': '#A85CF9'}
        # for col, color in color_dict.items():
        #     pr_df[col].plot.line(ax=ax1, color=color)
        plt.plot(pr_df[index_name], color='k', linestyle='--')
        plt.plot(pr_df.loc[:,stgy_names])
        last_index = pr_df.loc[:, stgy_names].index[-1]
        last_value = pr_df.loc[last_index, stgy_names]
        for name, value in zip(stgy_names, last_value):
            plt.text(last_index, value, name, ha='left', va='center')
        ymin, ymax = ax1.get_ylim()
        ax1.vlines(self.test_start_date, ymin=ymin, ymax=ymax, color='black')
        ax1.set_title(f'实盘模拟{"（做多做空）" if stgy_type=="pos" else "（只做多）"}：净值走势')
        ax1.grid('on')
        ax1.legend(
            stgy_names,
            loc='upper center', bbox_to_anchor=(0.5, -0.05), 
            fancybox=True, shadow=True, ncol=4, title='交易标的')
        ax2 = fig.add_subplot(3,1,2)
        self.stgy_df.loc[self.history_start_date:self.history_end_date, stgy_type].plot(ax=ax2, grid=True, title='pos')
        ax3 = fig.add_subplot(3,1,3)
        ax3.plot(eval_df.loc[:,price_ratio_names])
        ymin, ymax = ax3.get_ylim()
        ax3.vlines(self.test_start_date, ymin=ymin, ymax=ymax, color='black')
        ax3.legend(
            price_ratio_names, 
            loc='upper center', bbox_to_anchor=(0.5, -0.05), 
            fancybox=True, shadow=True, ncol=4, title='交易标的/基准')
        ax3.grid('on')
        ax3.set_title('价格比走势')
        # 在图上增加文字标明曲线
        last_index = eval_df.loc[:, price_ratio_names].index[-1]
        last_value = eval_df.loc[last_index, price_ratio_names]
        for name, value in zip(stgy_names, last_value):
            plt.text(last_index, value, name, ha='left', va='center')
        plt.tight_layout()
        plt.show()

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

        indicators = ['AnnRet', 'AnnRet_Simple', 'AnnVol', 'SR', 'MaxDD', 'Calmar', 'DD']
        res = pd.DataFrame(index=df.columns, columns=indicators)
        date_ordinal = pd.Series([dt.date.toordinal(e) for e in df.index])
        time_diff = date_ordinal - date_ordinal.shift(1) # 相对上一次净值的日期间隔
        for col in df: 
            p = df[col] # 单个资产价格或净值序列, pd.Series
            dd = (p/p.cummax() - 1)[-1] # 当前回撤
            r = p.pct_change() # 涨幅
            p.dropna(inplace=True)
            annret = (p[-1] / p[0]) ** (365/(p.index[-1]-p.index[0]).days) - 1 # 复利年化收益
            
            r1 = r.values / time_diff # 日均收益
            rb = df[df.columns[0]].pct_change().values / time_diff # 基准日均收益
            annret1 = np.nanmean(r1) * 365 # 单利年化收益
            r2 = r.values / np.sqrt(time_diff) # 波动率调整后涨幅   
            annvol = np.nanstd(r2) * np.sqrt(365) # 年化波动率
            sr = (annret - 0.025) / annvol # 夏普比率
            mdd = np.min(p/p.cummax() - 1) # 最大回撤
            calmar = annret / -mdd
            res.loc[col] = [annret, annret1, annvol, sr, mdd, calmar, dd]

        return res

    def search_etf(self):
        etf_list = q.search_mfp(f'{self.bmk_name}ETF')
        etf_list = etf_list[~etf_list.name.str.contains('联接')]
        etf_list = etf_list[(etf_list['maturity_date'] > dt.date.today()) | (etf_list['maturity_date'].isna())]
        etf_list = etf_list[etf_list['found_date'] < self.history_start_date]
        print(f'存续时间满足要求的ETF共{len(etf_list)}个')
        mfp_nav_df = q.get_mfp_asset_alloc(etf_list.index.tolist(), start_date='2021-12-31', end_date='2021-12-31')
        return pd.concat([etf_list[['name', 'found_date']], mfp_nav_df.set_index('mfp_id')[['mfp_nav']]], axis=1)