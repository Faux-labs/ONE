from tvDatafeed import TvDatafeed, Interval

def fetch_tv_data(symbol="ETHUSDT", exchange="BINANCE", bars=100):
    # Initialize (username/pass optional but recommended for more data)
    tv = TvDatafeed() 
    
    df = tv.get_hist(
        symbol=symbol, 
        exchange=exchange, 
        interval=Interval.in_1_hour, 
        n_bars=bars
    )
    return df

# Usage
# Note: You must know the exact exchange TV uses (e.g. 'BINANCE', 'COINBASE')
data = fetch_tv_data()
print(data)