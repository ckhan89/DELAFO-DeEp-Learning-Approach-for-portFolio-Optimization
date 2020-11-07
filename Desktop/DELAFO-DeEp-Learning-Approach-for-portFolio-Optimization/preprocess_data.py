import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
def prepair_data(path,window_x,window_y,tickers=[],period=[]):
    df = pd.read_csv(path)
    df['date'] = df.date.apply(pd.Timestamp)

    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    ## max 500 tickers
    num_tickers = df['ticker'].unique()
    num_tickers = num_tickers[:500]
    df = df[df['ticker'].isin(num_tickers)]
    #---------
    if len(tickers)>0:
        df = df[df['ticker'].isin(tickers)]
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)

    df = df.pivot_table(index='date', columns='ticker')
    ## select date
    if len(period)!=0:
        df = df[(df.index> period[0]) & (df.index < period[1])]

    ## select tickers not nan in final day
    columns = df.close.columns[~df.close.iloc[-1].isna()]
    df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]
    


    df.volume = df.volume.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
    df.close = df.close.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)


    close = df.close
    daily_return = ((close.shift(-1) - close)/close).shift(1)

    daily_return = daily_return.interpolate(method='linear',limit_area="inside",limit_direction='both', axis=0)
    # daily_return = daily_return.fillna(daily_return.min(axis=0),axis=0)
    # daily_return = daily_return.fillna(daily_return.min(axis=0),inplace=True)

    # daily_return.fillna(daily_return.min(axis=0), inplace=True)

    tickers = df.close.columns
    date = df.index

    X = df.values.reshape(df.shape[0],2,-1)
    ## Using for max scaling data
    # X_max = X.max(axis=0)[np.newaxis,np.newaxis,:,:]
    y = daily_return.values

    ## fill X
    ##fill nan by 0.0
    X[np.isnan(X)] = 0.0

    ## fill y
    y[np.isnan(y)] = -1e2
    # y[np.isnan(y)] = 0

    # X1 = rolling_array(X[window_x:],stepsize=1,window=window_y)

    X = rolling_array(X[:-window_y],stepsize=1,window=window_x)
    y = rolling_array(y[window_x:],stepsize=1,window=window_y)
    dates_X = rolling_array(date[:-window_y],stepsize=1,window=window_x)
    dates_y = rolling_array(date[window_x:],stepsize=1,window=window_y)
    X = np.moveaxis(X,-1,1)
    # X_max = np.moveaxis(X_max,-1,1)
    # X1 = np.moveaxis(X1,-1,1)
    y = np.swapaxes(y,1,2)

    return X,y,tickers,dates_X,dates_y

def rolling_array(a, stepsize=1, window=60):
    n = a.shape[0]
    return np.stack((a[i:i + window:stepsize] for i in range(0,n - window + 1)),axis=0)
