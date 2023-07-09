'''
Generate dataset for carbon credit price prediction from various sources
    1. carbon credit price data, 2017-2023, from EEX.com
    2. electricity price data, 2015-2023, from ember-climate.org
    3. other explanatory data, 2018-2023, from Yahoo Finance
'''

import pandas as pd
import matplotlib.pyplot as plt

# prepare Cprice data from several source files
def prepare_Cprice(rootdir='data/source/'):
    df_Cprice = pd.DataFrame()
    for i in range(2023,2016,-1):
        filename = 'emission-spot-primary-market-auction-report-{}-data.xlsx'.format(i)
        df = pd.read_excel(rootdir+filename)
        df.columns = df.iloc[4]
        df = df.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df.index[0:5])
        df_Cprice = df_Cprice.append(df)
    # # # we won't use the data a long time ago
    # # for i in range(2016,2012,-1):
    # #     filename = 'emission-spot-primary-market-auction-report-{}-data.xlsx'.format(i)
    # #     df = pd.read_excel(rootdir+filename)
    # #     df.columns = df.iloc[1]
    # #     if i == 2016:
    # #         df.rename(columns={'Auction Price EUR/tCO2':'Auction Price €/tCO2', 'Minimum Bid EUR/tCO2':'Minimum Bid €/tCO2', 'Maximum Bid EUR/tCO2':'Maximum Bid €/tCO2','Mean Price EUR/tCO2':'Mean €/tCO2','Median Price EUR/tCO2':'Median €/tCO2'}, inplace=True)
    # #     df = df.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df.index[0:5])
    # #     df_Cprice = df_Cprice.append(df)
    df_Cprice.loc[:,'Date'] = pd.to_datetime(df_Cprice['Date'])
    df_Cprice.rename(columns={'Auction Price €/tCO2':'Cprice', 'Minimum Bid €/tCO2':'Cprice_min','Maximum Bid €/tCO2':'Cprice_max','Mean €/tCO2':'Cprice_mean','Median €/tCO2':'Cprice_median'}, inplace=True)
    df_Cprice = df_Cprice.loc[:,['Date', 'Cprice']] # take the "auction price", FIXME or take the "mean price"?
    # df_Cprice.sort_values(by='Date', ascending=False, inplace=True)
    # df_Cprice = df_Cprice.reset_index(drop=True)
    return df_Cprice


# prepare Eprice data from source file, take the average of different countries
def prepare_Eprice(rootdir='data/source/'):
    filename = 'european_wholesale_electricity_price_data_daily-5.csv'
    df_Eprice = pd.read_csv(rootdir+filename)
    df_Eprice = df_Eprice.groupby(by=['Date']).mean().reset_index()
    df_Eprice.loc[:,'Date'] = pd.to_datetime(df_Eprice['Date'])
    df_Eprice.rename(columns={'Price (EUR/MWhe)':'Eprice'}, inplace=True)
    # df_Eprice.sort_values(by='Date', ascending=False, inplace=True)
    return df_Eprice


# prepare explanatory data from source file
def prepare_Xvar(rootdir='data/source/'):
    rootdir = rootdir + 'Xvar/'
    filedict = {
        'Brent Crude Oil Last Day Financ (BZ=F).xlsx': 'BrentOil',
        'Crude Oil Aug 23 (CL=F).xlsx': 'CrudeOilF',
        'Dutch TTF Natural Gas Calendar (TTF=F).xlsx': 'TTF-NatGas',
        'Natural Gas Aug 23 (NG=F).xlsx': 'NatGasF',
        'Coal (API2) CIF ARA (ARGUS-McCl (MTF=F).xlsx': 'Coal',
        'RBOB Gasoline Aug 23 (RB=F).xlsx': 'GasolineF',
        'Dow Jones Industrial Average (^DJI).xlsx': 'DJI',
        'S&P 500 (^GSPC).xlsx': 'S&P500',
        'USD-EUR (EUR=X).xlsx': 'USD-EUR',
    }
    dfX = pd.DataFrame(columns=['Date'])
    for filename,colname in filedict.items():
        df = pd.read_excel(rootdir+filename)
        df = df.loc[:,['Date','Close_']]
        df.rename(columns={'Close_':colname}, inplace=True)
        dfX = pd.merge(dfX, df, on='Date', how='outer')
    dfX['Date'] = pd.to_datetime(dfX['Date'])
    # dfX.sort_values(by='Date', ascending=False, inplace=True)
    # dfX.to_excel(rootdir+'Xvar.xlsx', index=False)
    return dfX


# merge all data: Cprice, Eprice, Xvar
def prepare_all(rootdir='data/source/'):
    df_Cprice = prepare_Cprice(rootdir)
    df_Eprice = prepare_Eprice(rootdir)
    dfX = prepare_Xvar(rootdir)

    df = pd.merge(left=df_Cprice, right=df_Eprice, how='left', on='Date', sort=False)
    df = pd.merge(left=df, right=dfX, how='left', on='Date', sort=False)
    df = df.iloc[26:1100] # FIXME hardcode '2018-07-09'~'2023-05-30'
    for col in df.columns[1:]:
        if  df[col].dtype != 'float64': # if dtype=object, mainly because there is ',' as thousand separator
            df[col] = df[col].astype(str).str.replace(',','').astype(float)
        df[col] = df[col].interpolate(method='linear', axis=0)
    
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':

    rootdir='data/source/'
    df = prepare_all(rootdir)
    df.to_excel(rootdir+'df.xlsx', index=False)

    print(df.info())
    print(df.head())

    # normalize to make the plot more readable
    df_norm = df.set_index('Date')
    df_norm = df_norm.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    # df_norm = df_norm.apply(lambda x: (x-x.min()) / (x.max()-x.min()), axis=0)
    df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
    df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
    plt.show()