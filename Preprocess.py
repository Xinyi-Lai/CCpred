'''
Generate dataset for carbon credit price prediction from various sources
'''

import pandas as pd
import matplotlib.pyplot as plt

rootdir = 'data/source'


def preprocess_source(rootdir):

    # prepare CC price data from several source files
    filename = r'/primary_auction_report_20230418_41186423.xlsx'
    df_Cprice = pd.read_excel(rootdir+filename)
    df_Cprice.columns = df_Cprice.iloc[4]
    df_Cprice = df_Cprice.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df_Cprice.index[0:5])

    for i in range(2022,2016,-1):
        filename = '/emission-spot-primary-market-auction-report-{}-data.xlsx'.format(i)
        df1 = pd.read_excel(rootdir+filename)
        df1.columns = df1.iloc[4]
        df1 = df1.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df1.index[0:5])
        df_Cprice = df_Cprice.append(df1)

    for i in range(2016,2012,-1):
        filename = '/emission-spot-primary-market-auction-report-{}-data.xlsx'.format(i)
        df1 = pd.read_excel(rootdir+filename)
        df1.columns = df1.iloc[1]
        if i == 2016:
            df1.rename(columns={'Auction Price EUR/tCO2':'Auction Price €/tCO2', 'Minimum Bid EUR/tCO2':'Minimum Bid €/tCO2', 'Maximum Bid EUR/tCO2':'Maximum Bid €/tCO2','Mean Price EUR/tCO2':'Mean €/tCO2','Median Price EUR/tCO2':'Median €/tCO2'}, inplace=True)
        df1 = df1.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df1.index[0:5])
        df_Cprice = df_Cprice.append(df1)

    df_Cprice = df_Cprice.reset_index(drop=True)
    df_Cprice.loc[:,'Date'] = pd.to_datetime(df_Cprice['Date'])
    df_Cprice.rename(columns={'Auction Price €/tCO2':'C_Price', 'Minimum Bid €/tCO2':'C_Price_min','Maximum Bid €/tCO2':'C_Price_max','Mean €/tCO2':'C_Price_mean','Median €/tCO2':'C_Price_median'}, inplace=True)

    # prepare electricity price data, taking the average of countries
    filename = r'/european_wholesale_electricity_price_data_daily-5.csv'
    df_Eprice = pd.read_csv(rootdir+filename)
    df_Eprice = df_Eprice.groupby(by=['Date']).mean().reset_index()
    df_Eprice.loc[:,'Date'] = pd.to_datetime(df_Eprice['Date'])
    df_Eprice.rename(columns={'Price (EUR/MWhe)':'E_Price'}, inplace=True)

    # merge CC price data and E price data
    df = pd.merge(left=df_Cprice, right=df_Eprice, how='left', on='Date', sort=False)
    df.fillna(method='pad',inplace=True,limit=1)
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)

    # export and plot
    df.to_excel( rootdir + r'/CCprice.xlsx')

    return df


if __name__ == '__main__':

    rootdir = 'data/source'

    df = preprocess_source(rootdir)

    print(df.info())
    print(df.head())

    date_col = df['Date']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(date_col, df['C_Price'], 'r', label='C_price')
    ax1.legend(loc=2)
    ax1.set_ylabel('C_Price (EUR/tCO2)')
    ax1.set_xlabel('Time')
    ax2 = ax1.twinx() # this is the important function
    ax2.plot(date_col, df['E_Price'], 'g', label='E_price')
    ax2.legend(loc=1)
    ax2.set_ylabel('E_price (EUR/MWhe)')
    plt.show()