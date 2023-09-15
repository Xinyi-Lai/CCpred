'''
Generate dataset for carbon credit price prediction
    1. EU-ETS
    2. CHN-ETS
'''

import pandas as pd
import matplotlib.pyplot as plt



''' EU-ETS 
    Generate dataset for carbon credit price prediction from various sources
    1. carbon credit price data, 2017-2023, from EEX.com
    2. electricity price data, 2015-2023, from ember-climate.org
    3. other explanatory data, 2018-2023, from Yahoo Finance
'''

# prepare EU Cprice data from several source files
def prepare_Cprice_EU():
    rootdir='data/source/EU/EEX/'
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
    df_Cprice.sort_values(by='Date', ascending=False, inplace=True)
    # df_Cprice.to_excel(rootdir+'0Cprice.xlsx', index=False)
    return df_Cprice


# prepare EU Eprice data from source file, take the average of different countries
def prepare_Eprice_EU():
    rootdir='data/source/EU/'
    filename = 'european_wholesale_electricity_price_data_daily-5.csv'
    df_Eprice = pd.read_csv(rootdir+filename)
    df_Eprice = df_Eprice.groupby(by=['Date']).mean().reset_index()
    df_Eprice.loc[:,'Date'] = pd.to_datetime(df_Eprice['Date'])
    df_Eprice.rename(columns={'Price (EUR/MWhe)':'Eprice'}, inplace=True)
    df_Eprice.sort_values(by='Date', ascending=False, inplace=True)
    df_Eprice.to_excel(rootdir+'0Eprice.xlsx', index=False)
    return df_Eprice


# prepare explanatory data from source file
def prepare_Xvar_EU():
    rootdir='data/source/EU/Xvar/'
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
    dfX.sort_values(by='Date', ascending=False, inplace=True)
    dfX.to_excel(rootdir+'0Xvar.xlsx', index=False)
    return dfX


# merge EU data: Cprice, Eprice, Xvar
def prepare_all_EU():
    df_Cprice = prepare_Cprice_EU()
    df_Eprice = prepare_Eprice_EU()
    dfX = prepare_Xvar_EU()

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


'''
Generate dataset for carbon markets in China
    ['北京绿色交易所' '广州碳排放权交易所' '上海环境能源交易所' '湖北碳排放权交易中心' '天津排放权交易所' '重庆碳排放权交易中心' '福建海峡交易中心' '深圳排放权交易所']
    https://ets.sceex.com.cn/internal.htm?k=guo_nei_xing_qing&url=mrhq_gn&orderby=tradeTime%20desc&pageSize=14
'''
def prepare_chn_market(rootdir='data/source/'):
    rootdir = rootdir + 'chn/'
    
    # read data of domestic markets
    filename = '碳排放权交易-每日行情.xlsx'
    df = pd.read_excel(rootdir+filename, na_values='-')
    df['交易日期'] = pd.to_datetime(df['交易日期'])

    # prelimary cleaning
    df = df[ (df['交易机构'] != '欧洲气候交易所') & (df['交易机构'] != '欧洲能源交易所') ] # 欧洲的数据不全且有其他来源, 暂时不考虑
    df.replace('北京环境交易所', '北京绿色交易所', inplace=True) # 北京绿色交易所的前身是2008年8月5日成立的北京环境交易所, 2020年更名为北京绿色交易所
    df.replace('海峡股权交易中心', '福建海峡交易中心', inplace=True) # 海峡股权交易中心和福建海峡交易中心都是福建省的排放权交易所
    orgs = df['交易机构'].unique()
    # print(orgs) # ['北京绿色交易所' '广州碳排放权交易所' '上海环境能源交易所' '湖北碳排放权交易中心' '天津排放权交易所' '重庆碳排放权交易中心' '福建海峡交易中心' '深圳排放权交易所']

    # group, join
    df = df.groupby(by=['交易日期','交易机构']).mean().reset_index() # there are multiple rows (products) for the same date and org
    df_chn = pd.DataFrame(columns=['交易日期'])
    for i in orgs:
        df0 = df.loc[df['交易机构']==i, ['交易日期','成交均价']].rename(columns={'成交均价':i})
        df_chn = pd.merge(df_chn, df0, on='交易日期', how='outer')
        
    # read Xvar data of domestic markets
    filename = '欧盟碳价-原油-煤炭-上证-广州碳价-USDCNY-EURCNY.xlsx'
    dfX = pd.read_excel(rootdir+filename)
    dfX['日期'] = pd.to_datetime(dfX['日期'])
    dfX.rename(columns={ '日期':'交易日期'}, inplace=True)
    df_chn = pd.merge(df_chn, dfX, how='inner', on='交易日期', sort=True)

    df_chn.rename(columns={ '交易日期':'Date', 
                            '北京绿色交易所':'Beijing', '广州碳排放权交易所':'Guangzhou', '上海环境能源交易所':'Shanghai', 
                            '湖北碳排放权交易中心':'Hubei', '天津排放权交易所':'Tianjin', '重庆碳排放权交易中心':'Chongqing', 
                            '福建海峡交易中心':'Fujian', '深圳排放权交易所':'Shenzhen',
                            '原油':'Oil', '煤炭':'Coal', '上证':'SSEIndex', '广州碳价':'Guangzhou Cprice'
                            }, inplace=True)
    df_chn = df_chn.sort_values(by='Date').reset_index(drop=True)
    
    # for col in df_chn.columns:
    #     df_chn[col] = df_chn[col].interpolate(method='linear', axis=0)
    
    return df_chn


def prepare_chn(save=True):
    rootdir = 'data/source/chn/'
    
    ### CHN markets
    cities = {
        'Guangzhou': '广州',
        'Hubei': '湖北',
        'Shanghai': '上海',
        'Beijing': '北京',
        'Fujian': '福建',
        'Chongqing': '重庆',
        # 'Tianjin': '天津', # 太多交易品种
        # 'Shenzhen': '深圳' # 太多交易品种
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
    vis(df_chn)
    # print(df_chn.info())
    # print(df_chn)
    # print(df_chn[df_chn['Date'].duplicated()])

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
    vis(df_chn)
    # print(dfX.info())
    # print(dfX)

    ### df_all: join CCprices and Xvars
    df_all = pd.merge(df_chn, dfX, how='left', on='Date', sort=True)
    df_all = df_all.sort_values(by='Date').reset_index(drop=True)
    vis(df_all)
    # print(df_all.info())
    # print(df_all)

    ### df_itpl: interpolate missing values
    df_itpl = df_all.copy()
    for col in df_itpl.columns[1:]:
        df_itpl[col] = df_itpl[col].interpolate(method='linear', axis=0)
    vis(df_itpl)
    # print(df_itpl.info())
    # print(df_itpl)

    ### save to excel
    if save:
        writer = pd.ExcelWriter('data/df_chn_new.xlsx')
        df_chn.to_excel(writer,'chn-markets')
        dfX.to_excel(writer,'Xvars')
        df_all.to_excel(writer,'df-chn')
        df_itpl.to_excel(writer,'df-chn-itpl')
        writer.save()
    
    return df_itpl


def vis(df):
    df_norm = df.set_index('Date')
    df_norm = (df_norm - df_norm.mean()) / df_norm.std()
    df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
    # df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
    plt.show()
    return


if __name__ == '__main__':
    
    # # df_Cprice = prepare_Cprice_EU()
    # # df_Cprice.to_excel('data/source/EU/EEX.xlsx', index=False)
    # # dfX = prepare_Xvar_EU()
    # dfX.to_excel('data/source/EU/dfX.xlsx', index=False)
    
    # ### df_eu
    # # read, clean, merge, interpolate, plot
    # df = prepare_all_EU()
    # df.to_excel('data/df_eu.xlsx', index=False)
    # print(df.info())
    # print(df.head())
    # # normalize to make the plot more readable
    # df_norm = df.set_index('Date')
    # df_norm = (df_norm - df_norm.mean()) / df_norm.std()
    # df_norm.iloc[:,1:].plot(figsize=(12,8), x_compat=True, alpha=0.6)
    # df_norm.Cprice.plot(color='black', label='Cprice', legend=True)
    # plt.show()
    
    # # TODO:
    # rootdir='data/source/'
        
    # ### df_chn
    # df_chn = prepare_chn_market(rootdir)
    # df_chn.to_excel('data/df_chn1.xlsx', index=False)
    # print(df_chn.info())
    # print(df_chn.head())


    # df_chn: read, clean, merge, interpolate
    df_chn = prepare_chn(save=False)
    vis(df_chn)
