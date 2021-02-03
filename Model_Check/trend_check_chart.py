def SMA(stock_code, term=14, start="2018-01-01", end=dt.datetime.now()):
    df = yf.download(stock_code, start, end, interval='1d') 

    # calculation SMA
    short_sma = 20
    long_sma = 50
    SMAs = [short_sma, long_sma]
    for i in SMAs:
        df["SMA_"+str(i)] = df.iloc[:,4].rolling(window=i).mean()

    # judge Up trend or Down trend by SMA
    # Basics: if shorter SMA is higher than longer SMA, its now in up-trend area. Down-trend area if it's in the opposite condition.
    position = 0 # 1 means we ave already entered position, 0 means not already entered.
    counter = 0
    up_trend_days = []
    down_trend_days = []
    percentChange = []
    for i in df.index:
        SMA_short = df['SMA_20']
        SMA_long = df['SMA_50']

        if (SMA_short[i] > SMA_long[i]):
            up_trend_days.append(i)

        elif (SMA_short[i] < SMA_long[i]):
            down_trend_days.append(i)


    up_signals = []
    down_signals = []
    for i in df.index:
        if i in up_trend_days:
            up_signals.append(df['Adj Close'][i] + 10)
        else:
            up_signals.append(np.nan)
        if i in down_trend_days:
            down_signals.append(df['Adj Close'][i] + 10)
        else:
            down_signals.append(np.nan)
    return { 'df': df, 'up_trend_days': up_trend_days, 'down_trend_days':down_trend_days, 'up_signals':up_signals, 'down_signals': down_signals } 

# plot
stock_code = 'UAA' 
result = SMA(stock_code)
adp = [
    mpf.make_addplot(result['up_signals'], type='scatter', markersize=200, marker='^', color="b"),
    mpf.make_addplot(result['down_signals'], type='scatter', markersize=200, marker='v', color="r"),
]
fig, ax = mpf.plot(result['df'], type='candle', figratio=(45, 15),
        mav=(20, 50),
        addplot=adp,
        volume=True, title=str(stock_code)+" SMA",
        style='starsandstripes', returnfig=True)
legend = ['Short_SMA(20)', 'Long_SMA(50)']
ax[0].legend(legend, fontsize=16)
fig.savefig('./sma.png')
