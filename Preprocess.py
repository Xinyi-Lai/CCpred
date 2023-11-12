'''
Generate dataset for carbon credit price prediction
    1. EU-ETS
    2. CHN-ETS
'''

import pandas as pd
import matplotlib.pyplot as plt



def prepare_eu(save=False, vis=False):
    ''' EU-ETS 
        Generate dataset for carbon credit price prediction from various sources
        1. carbon credit price data, 20170109-20230707, from EEX.com
        2. electricity price data, 20150101-20230531, from ember-climate.org
        3. other explanatory data, 20180709-20230707, from Yahoo Finance
    '''

    rootdir='data/source/EU/'

    ### prepare EU Cprice data from several source files
    df_Cprice = pd.DataFrame()
    for i in range(2023,2016,-1):
        filename = 'emission-spot-primary-market-auction-report-{}-data.xlsx'.format(i)
        df = pd.read_excel(rootdir+'EEX/'+filename)
        df.columns = df.iloc[4]
        df = df.loc[:,['Date','Auction Price €/tCO2','Minimum Bid €/tCO2','Maximum Bid €/tCO2','Mean €/tCO2','Median €/tCO2']].drop(index=df.index[0:5])
        df_Cprice = df_Cprice.append(df)
    df_Cprice.loc[:,'Date'] = pd.to_datetime(df_Cprice['Date'])
    df_Cprice.rename(columns={'Auction Price €/tCO2':'Cprice', 'Minimum Bid €/tCO2':'Cprice_min','Maximum Bid €/tCO2':'Cprice_max','Mean €/tCO2':'Cprice_mean','Median €/tCO2':'Cprice_median'}, inplace=True)
    df_Cprice = df_Cprice.loc[:,['Date', 'Cprice']] # take the "auction price", FIXME or take the "mean price"?
    df_Cprice = df_Cprice.sort_values(by='Date').reset_index(drop=True)
    
    ### prepare EU Eprice data from source file, take the average of different countries
    filename = 'european_wholesale_electricity_price_data_daily-5.csv'
    df_Eprice = pd.read_csv(rootdir+filename)
    df_Eprice = df_Eprice.groupby(by=['Date']).mean().reset_index()
    df_Eprice.loc[:,'Date'] = pd.to_datetime(df_Eprice['Date'])
    df_Eprice.rename(columns={'Price (EUR/MWhe)':'Eprice'}, inplace=True)
    df_Eprice = df_Eprice.sort_values(by='Date').reset_index(drop=True)

    ### prepare explanatory data from source file
    Xvars = {
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
    for filename,colname in Xvars.items():
        df = pd.read_excel(rootdir+'Xvar/'+filename)
        df = df.loc[:,['Date','Close_']]
        df.rename(columns={'Close_':colname}, inplace=True)
        dfX = pd.merge(dfX, df, on='Date', how='outer')
    dfX['Date'] = pd.to_datetime(dfX['Date'])
    dfX = dfX.sort_values(by='Date').reset_index(drop=True)

    ### merge EU data: Cprice(20170109-20230707), Eprice(20150101-20230531), Xvar(20180709-20230707)
    # outer join to observe data missing
    df_eu_outer = pd.merge(left=df_Cprice, right=df_Eprice, how='outer', on='Date', sort=False)
    df_eu_outer = pd.merge(left=df_eu_outer, right=dfX, how='outer', on='Date', sort=False)
    df_eu_outer = df_eu_outer.sort_values(by='Date').reset_index(drop=True)
    # merge to df_eu(20180709-20230707) NOTE: df_eu is not continuous in time (only weekdays have transaction)
    df_eu = pd.merge(left=df_Cprice, right=dfX, how='right', on='Date', sort=False)
    df_eu = pd.merge(left=df_eu, right=df_Eprice, how='inner', on='Date', sort=False)
    for col in df_eu.columns[1:]:
        if  df_eu[col].dtype != 'float64': # if dtype=object, mainly because there is ',' as thousand separator
            df_eu[col] = df_eu[col].astype(str).str.replace(',','').astype(float)
        df_eu[col] = df_eu[col].interpolate(method='linear', axis=0)
    df_eu = df_eu.sort_values(by='Date').reset_index(drop=True)
    

    ### save to excel
    if save:
        writer = pd.ExcelWriter('data/df_eu.xlsx')
        df_eu.to_excel(writer,'df-eu')
        df_eu_outer.to_excel(writer,'df-eu-outer')
        df_Cprice.to_excel(writer,'Cprice')
        df_Eprice.to_excel(writer,'Eprice')
        dfX.to_excel(writer,'Xvars')
        writer.save()

    ### visualize
    if vis:
        print(df_eu.info())
        print(df_eu.head())
        # normalize to make the plot more readable
        df_norm = df_eu.set_index('Date')
        df_norm = (df_norm - df_norm.mean()) / df_norm.std()
        df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
        df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
        plt.show()

    return df_eu



def prepare_chn(save=False, vis=False):
    ''' CHN-ETS 
        Generate dataset for carbon credit price prediction from various sources
        1. carbon emissions trading market data, 20160701-20230724, from ets.sceex.com.cn
        https://ets.sceex.com.cn/internal.htm?k=guo_nei_xing_qing&url=mrhq_gn&orderby=tradeTime%20desc&pageSize=14
        ['北京绿色交易所' '广州碳排放权交易所' '上海环境能源交易所' '湖北碳排放权交易中心' '天津排放权交易所' '重庆碳排放权交易中心' '福建海峡交易中心' '深圳排放权交易所']
        2. explanatory data, 20160701-20230816, from investing.com
    '''

    rootdir = 'data/source/chn/'
    
    ### CHN markets
    cities = {
        'Guangzhou': '广州',    # 数据相对完整
        'Hubei': '湖北',        # 数据相对完整
        'Shanghai': '上海',     # 数据还算完整
        'Beijing': '北京',      # 数据不完整
        'Fujian': '福建',       # 数据不完整
        'Chongqing': '重庆',    # 数据不完整
        # 'Tianjin': '天津',      # 同一天交易品种太多，不建议使用
        # 'Shenzhen': '深圳'      # 同一天交易品种太多，不建议使用
    }
    df_chn = pd.DataFrame(columns=['Date'])
    for city in cities.keys():
        df0 = pd.read_excel(rootdir+'碳排放权交易-每日行情.xlsx', sheet_name=cities[city], na_values='-')
        df0 = df0[['交易日期', '成交均价']]
        df0.dropna(inplace=True)
        df0.rename(columns={'交易日期':'Date', '成交均价':city}, inplace=True)
        df0['Date'] = pd.to_datetime(df0['Date'])
        df_chn = pd.merge(df_chn, df0, how='outer', on='Date', sort=True)
    df_chn = df_chn.sort_values(by='Date').reset_index(drop=True)
    # print(df_chn.info())
    # print(df_chn)
    # print(df_chn[df_chn['Date'].duplicated()]) # 天津、深圳市场同一天有多个交易品种

    ### X Vars
    Xvars = {
        'EU-CC': '欧洲碳排放期货历史数据.csv',
        'WTI-Oil': 'WTI原油期货历史数据.csv',
        'Brent-Oil': '伦敦布伦特原油期货历史数据.csv',
        'Zhengzhou-Coal': '动力煤期货历史数据.csv',
        'Dalian-Coal': '焦煤期货历史数据.csv',
        'Rtd-Coal': 'Rotterdam Coal Futures历史数据.csv',
        'US-NatGas': '美国天然气期货历史数据.csv',
        # 'TTF-NatGas': 'Dutch TTF Natural Gas Futures历史数据.csv',
        'SH-FOil': '上海燃料油期货历史数据.csv',
        'US-FOil': '美国燃料油期货历史数据.csv',
        'CSI300': '沪深300指数历史数据.csv',
        'US-DJI': '道琼斯工业平均指数历史数据.csv',
        'USD-CNY': 'USD_CNY历史数据.csv',
    }
    dfX = pd.DataFrame(columns=['Date'])
    for xvar in Xvars.keys():
        df0 = pd.read_csv(rootdir + 'Xvar/' + Xvars[xvar])
        df0 = df0[['日期', '收盘']]
        df0.dropna(inplace=True)
        if df0['收盘'].dtype != 'float64': # if dtype=object, mainly because there is ',' as thousand separator# if dtype=object, mainly because there is ',' as thousand separator
            df0['收盘'] = df0['收盘'].str.replace(',','').astype(float)
        df0.rename(columns={'日期':'Date', '收盘':xvar}, inplace=True)
        df0['Date'] = pd.to_datetime(df0['Date'])
        dfX = pd.merge(dfX, df0, how='outer', on='Date', sort=True)
    dfX = dfX.sort_values(by='Date').reset_index(drop=True)

    ### df_all: join CCprices and Xvars
    df_all = pd.merge(df_chn, dfX, how='left', on='Date', sort=True)
    df_all = df_all.sort_values(by='Date').reset_index(drop=True)

    ### df_itpl: interpolate missing values
    df_itpl = df_all.copy()
    for col in df_itpl.columns[1:]:
        df_itpl[col] = df_itpl[col].interpolate(method='linear', axis=0)
    df_itpl.dropna(how='any', inplace=True)
    df_itpl = df_itpl.sort_values(by='Date').reset_index(drop=True)

    ### save to excel
    if save:
        writer = pd.ExcelWriter('data/df_chn.xlsx')
        df_chn.to_excel(writer,'chn-markets')
        dfX.to_excel(writer,'Xvars')
        df_all.to_excel(writer,'df-chn')
        df_itpl.to_excel(writer,'df-chn-itpl')
        writer.save()

    ### visualize
    if vis:
        print(df_itpl.info())
        print(df_itpl.head())
        # normalize to make the plot more readable
        df_norm = df_itpl.set_index('Date')
        df_norm = (df_norm - df_norm.mean()) / df_norm.std()
        df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
        # df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
        plt.show()

    return df_itpl


def vis(df):
    print(df.info())
    print(df.head())
    # normalize to make the plot more readable
    df_norm = df.set_index('Date')
    df_norm = (df_norm - df_norm.mean()) / df_norm.std()
    df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
    df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
    plt.show()
    return


if __name__ == '__main__':
    
    # read, clean, merge, interpolate, plot
    
    # df_eu = prepare_eu(save=True, vis=True)

    df_chn = prepare_chn(save=True, vis=True)

