# coding=utf-8
import pandas as pd



def xyz(p1, p2, tp):
    """
    (p1.x - p2.x) / (p1.y - p2.y) * (p.y - p2.y) + p2.x
    点在线的左边还是右边
    左=true, 右=false

    """

    x1, y1 = p1
    x2, y2 = p2

    x, y = tp

    #tmp = (y1 – y2) *x + (x2 – x1) *y + x1 * y2 – x2 * y1
    tmpx = (x1-x2) / (y1-y2) * (y-y2) + x2
    if tmpx > x: return True
    return False

class ZB(object):

    @classmethod
    def kdj(cls, stock_data):
        """
        RSV:=(CLOSE-LLV(LOW,N))/(HHV(HIGH,N)-LLV(LOW,N))*100;
        K:SMA(RSV,M1,1);
        D:SMA(K,M2,1);
        J:3*K-2*D;

        :param stock_data:
        :return:
        """
        df = stock_data.loc[:, ["close", "low", "high"]]
        min_low = df['low'].rolling(window=9, min_periods=1).min()
        max_high = df['high'].rolling(window=9, min_periods=1).max()

        rsv = (df["close"] - min_low) / (max_high - min_low) * 100

        df["kdj_k"] = rsv.ewm(com=2).mean()  # 加权移动平均
        df["kdj_d"] = df["kdj_k"].ewm(com=2).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

        return df.loc[:, ["kdj_k", "kdj_d", "kdj_j"]]

    @classmethod
    def _cross(cls, s1, s2):
        df_temp = pd.DataFrame({"a": s1, "b": s2})

        df_temp["cross"] = False
        df_temp.ix[(df_temp['a'].shift(1) <= df_temp['b'].shift(1)) & (df_temp['a'] > df_temp['b']), "cross"] = True
        return df_temp["cross"]

    @classmethod
    def _max(cls, s1, s2):
        df_temp = pd.DataFrame({"a": s1, "b": s2})
        df_temp['max'] = 0

        df_temp.ix[df_temp['a'] >= df_temp['b'], 'max'] = df_temp['a']
        df_temp.ix[df_temp['a'] < df_temp['b'], 'max'] = df_temp['b']

        return df_temp['max']

    @classmethod
    def _round(cls, df, names, d=4):
        for name in names:
            df[name] = df[name].round(decimals=d)

    @classmethod
    def mdi(cls, stock_data, n=14, m=6):
        """
        MTR:=SUM(MAX(MAX(HIGH-LOW,ABS(HIGH-REF(CLOSE,1))),ABS(REF(CLOSE,1)-LOW)),N);
        HD :=HIGH-REF(HIGH,1);
        LD :=REF(LOW,1)-LOW;
        DMP:=SUM(IF(HD>0&&HD>LD,HD,0),N);
        DMM:=SUM(IF(LD>0&&LD>HD,LD,0),N);
        PDI: DMP*100/MTR;
        MDI: DMM*100/MTR;
        ADX: MA(ABS(MDI-PDI)/(MDI+PDI)*100,M);
        ADXR:(ADX+REF(ADX,M))/2;

        :param stock_data:
        :param n:
        :param m:
        :return:
        """

        df = stock_data.loc[:, ["close", "low", "high", "open"]]

        h_l = df['high'] - df['low']
        h_ref_c = abs(df['high'] - df['close'].shift(1))
        ref_c_l = abs(df["close"].shift(1) - df['low'])

        tt1 = cls._max(cls._max(h_l, h_ref_c), ref_c_l)

        MTR = tt1.rolling(center=False, min_periods=1, window=n).sum()
        HD = df["high"] - df["high"].shift(1)
        LD = df["low"].shift(1) - df["low"]

        df_tmp = pd.DataFrame({"MTR": MTR, "HD": HD, "LD": LD})

        df_tmp["DMP"] = 0
        df_tmp.ix[(df_tmp['HD'] > 0) & (df_tmp['HD'] > df_tmp['LD']), "DMP"] = df_tmp['HD']
        df_tmp["DMP"] = df_tmp["DMP"].rolling(center=False, min_periods=1, window=n).sum()

        df_tmp["DMM"] = 0
        df_tmp.ix[(df_tmp['LD'] > 0) & (df_tmp['LD'] > df_tmp['HD']), "DMM"] = df_tmp['LD']
        df_tmp["DMM"] = df_tmp["DMM"].rolling(center=False, min_periods=1, window=n).sum()

        df_tmp['PDI'] = df_tmp["DMP"] * 100 / df_tmp['MTR']
        df_tmp['MDI'] = df_tmp["DMM"] * 100 / df_tmp['MTR']

        df_tmp['ADX'] = abs(df_tmp['MDI'] - df_tmp['PDI']) / (df_tmp['MDI'] + df_tmp['PDI']) * 100
        df_tmp['ADX'] = df_tmp['ADX'].rolling(center=False, min_periods=1, window=m).mean()  # 简单移动平均

        df_tmp['ADXR'] = (df_tmp['ADX'] + df_tmp['ADX'].shift(m)) / 2

        return df_tmp.loc[:, ['PDI', 'MDI', 'ADX', 'ADXR']]


    @classmethod
    def ma(cls, stock_data, m1=5, m2=10, m3=20, m4=60):
        """
        MA1:MA(CLOSE,M1);
        MA2:MA(CLOSE,M2);
        MA3:MA(CLOSE,M3);
        MA4:MA(CLOSE,M4);

        :param stock_data:
        :return:
        """
        df = stock_data.loc[:, ["close"]]
        names = []
        for m in [m1, m2, m3, m4]:
            if m <= 0: continue
            name = 'ma%d' % m
            names.append(name)
            df[name] = df['close'].rolling(center=False, min_periods=1, window=m).mean()

        return df.loc[:, names]

    @classmethod
    def vol(cls, stock_data, m1=3, m2=5, m3=10):
        """
        :param stock_data:
        :return:
        """
        df = stock_data.loc[:, ["volume"]]
        df['VOL'] = df['volume']
        names = []
        for m in [m1, m2, m3]:
            if m <= 0: continue
            name = 'VMA%d' % m
            names.append(name)
            df[name] = df['volume'].rolling(center=False, min_periods=1, window=m).mean()

        return df.loc[:, names + ['VOL']]

    @classmethod
    def ma_diff_desc(cls, stock_data):
        """
        均线系统描述及打分
        :param stock_data:
        :return:
        """
        df = stock_data.loc[:, ["close", "ma5", "ma10", "ma20"]]
        ma5_10 = (df['ma5'] - df['ma10']) / df['ma10'] * 100
        ma10_20 = (df['ma10'] - df['ma20']) / df['ma20'] * 100
        ma5_20 = (df['ma5'] - df['ma20']) / df['ma20'] * 100
        ma_desc = (ma5_10 + ma10_20 + (ma5_20 / 2)) / 3

        df["_t1"] = 0.0
        df.ix[(df['ma5'] > df['ma5'].shift(1)), '_t1'] = 1
        df.ix[(df['ma5'] < df['close']), '_t1'] += 0.5

        df["_t2"] = 0.0
        df.ix[(df['ma10'] > df['ma10'].shift(1)), '_t2'] = 1
        df.ix[(df['ma10'] < df['close']), '_t2'] += 0.5

        df["_t3"] = 0.0
        df.ix[(df['ma20'] > df['ma20'].shift(1)), '_t3'] = 1
        df.ix[(df['ma20'] < df['close']), '_t3'] += 0.5

        ma_score = df['_t1'] + df['_t2'] + df['_t3']

        df_tmp = pd.DataFrame({
            "ma5_10": ma5_10, "ma10_20": ma10_20, "ma5_20": ma5_20,
            "ma_desc": ma_desc, "ma_score": ma_score
        })

        cls._round(df_tmp, names=["ma5_10", "ma10_20", "ma5_20", "ma_desc", "ma_score"])
        return df_tmp


    @classmethod
    def simple_duokong(cls, stock_data):
        df = stock_data.loc[:, ["close", "open", "volume", "p_change"]]

        df["dk_flag"] = 0
        df.ix[df['close'] > df['open'], 'dk_flag'] = 1
        df.ix[(df['close'] == df['open']) & (df['close'] >= df['close'].shift(1)), 'dk_flag'] = 1

        df["mrate5"] = df["dk_flag"].rolling(center=False, min_periods=1, window=5).sum() / 5.0
        df["mrate10"] = df["dk_flag"].rolling(center=False, min_periods=1, window=10).sum() / 10.0

        df["mrate_avg"] = (df["mrate5"] + df["mrate10"]) / 2
        df["mrate_diff"] = df["mrate5"] - df["mrate10"]

        cls._round(df, names=["mrate5", "mrate10", "mrate_avg", "mrate_diff"])
        return df.loc[:, ["mrate5", "mrate10", "mrate_avg", "mrate_diff"]]


    @classmethod
    def vol_duokong(cls, stock_data):
        df = stock_data.loc[:, ["close", "open", "volume"]]

        df["dk_flag"] = 0
        df.ix[df['close'] > df['open'], 'dk_flag'] = 1
        df.ix[(df['close'] == df['open']) & (df['close'] >= df['close'].shift(1)), 'dk_flag'] = 1

        df['red_v'] = 0
        df.ix[df['dk_flag'] == 1, 'red_v'] = df['volume']

        df['sum3'] = df["volume"].rolling(center=False, min_periods=1, window=3).sum()
        df['red_v_sum3'] = df["red_v"].rolling(center=False, min_periods=1, window=3).sum()
        df['red_v_rate3'] = df['red_v_sum3'] / df['sum3'] * 100

        df['sum5'] = df["volume"].rolling(center=False, min_periods=1, window=5).sum()
        df['red_v_sum5'] = df["red_v"].rolling(center=False, min_periods=1, window=5).sum()
        df['red_v_rate5'] = df['red_v_sum5'] / df['sum5'] * 100

        df['sum10'] = df["volume"].rolling(center=False, min_periods=1, window=10).sum()
        df['red_v_sum10'] = df["red_v"].rolling(center=False, min_periods=1, window=10).sum()
        df['red_v_rate10'] = df['red_v_sum10'] / df['sum10'] * 100

        df['rate_diff_5_10'] = df['red_v_rate5'] - df['red_v_rate10']
        df['rate_diff_3_10'] = df['red_v_rate3'] - df['red_v_rate10']

        df['rate_sum_5_10'] = df['sum5'] / df['sum10'] * 100
        df['rate_sum_3_10'] = df['sum3'] / df['sum10'] * 100

        names = [
            "red_v_rate3", "red_v_rate5", "red_v_rate10",
            "rate_diff_5_10", "rate_diff_3_10",
            "rate_sum_5_10", "rate_sum_3_10"
        ]
        cls._round(df, names=names)
        return df.loc[:, names]

    @classmethod
    def zf_target(cls, stock_data, days=5):
        df = stock_data.loc[:, ["close", "open", "p_change"]]

        # 计算未来N天的涨幅
        df['fzf'] = df["p_change"].shift(-days).rolling(center=False, min_periods=1, window=days).sum()

        df['target'] = 1
        df.ix[df['fzf'] > 2, 'target'] = 3
        df.ix[(df['fzf'] > -2) & (df['fzf'] <= 2), 'target'] = 2

        # 负-1，平-2，涨-3
        return df.loc[:, ["target"]]

    @classmethod
    def buy_signal(cls, stock_data):
        kdj_df = cls.kdj(stock_data)

        kdj_df['signal'] = 0
        kdj_df['k_ref1'] = kdj_df['kdj_k'].shift(1)
        kdj_df['k_ref2'] = kdj_df['kdj_k'].shift(2)

        kdj_df.ix[(kdj_df['kdj_k'] <= 40) & (kdj_df['k_ref2'] > kdj_df['k_ref1']) & (kdj_df['kdj_k'] > kdj_df['k_ref1']), 'signal'] = 1

        # tmp_df = pd.concat([df, kdj_df], axis=1) # 横向合并
        return kdj_df.loc[:, ["signal"]]

    @classmethod
    def ma_buy_signal2(cls, df):

        names = ["ma5_10", "ma10_20", "ma5_20", "ma_desc"]
        df = df.ix[:, names]
        df['fii1'] = 0

        p1 = (-25, -3.5)
        p2 = (-5.5, -19)

        for idd, row in df.iterrows():
            tp = (row['ma10_20'], row['ma5_10'])
            flg = xyz(p1, p2, tp)
            if flg:
                df.loc[idd, "fii1"] = 1

        signal_df = df.ix[(df['fii1'] == 1) | (df['ma5_20'] <= -25) | (df['ma_desc'] <= -13), names]

        signal_df['signal2'] = 1

        return signal_df













