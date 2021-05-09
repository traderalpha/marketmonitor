import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
import investpy
import warnings
warnings.filterwarnings("ignore")
from datetime import date, timedelta
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
import plotly

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.core.display import display, HTML
import nsepy


col1, mid, col2 = st.beta_columns([1,1,20])
with col1:
    st.image('Logo.jpg', width=80)
with col2:
    st.write('# ALPHA TRADER')
#st.image("", width=60, height=60)
st.write("### Sector Returns Heatmap")
st.write('(MCap Weighted)')


@st.cache()
def import_data():
	"""
	Imports and formats data from Dashboard.xlsx
	
	"""
	df = pd.read_excel("Dashboard.xlsx", header=1, engine='openpyxl')[1:]
	df = df[['Stock code', 'Stock Name', 'Sector', 'Industry','MCAP Cr',  'Current Price Rs', 'Day Change %', 'Week Change %',
	       'Month Change %', 'Qtr Change %', 'Half Yr Change %', '1Yr change %',
	       '2Yr price change %', 'Spread 52Week High Low %',
	       '% Distance from 52week low', '% Distance from 52week high', 'Beta 1Year',
	       'Consolidated previous end of day volume',
	       'Consolidated 5day average end of day volume',
	       'Consolidated 30day average end of day volume', 'Operating Revenue growth TTM %',
	       'Revenue Growth Qtr YoY %', 'Operating Profit Margin Growth YoY %',
	       'Net Profit TTM Growth %', 'Price To Book Value Annual', 'EPS TTM Growth %',]]

	df.iloc[:,4:] = df.iloc[:,4:].round(2)
	df.columns = ['Ticker', 'Name','Sector', 'Industry', 'MCap(Cr)',
	       'Price', '1D', '1W', '1M', '3M', '6M', '1Y', '2Y',
	       '52W HL(%)', '%52W Low','%52W High', '1Y Beta',
	       '1D Vol','5D Vol','1M Vol','OpRev Gr TTM(%)', 'Rev Gr YoY(%)',
	       'OPM Gr YoY(%)', 'Net Profit TTM Gr(%)','PB Ratio', 'EPS TTM Gr(%)']
	df.set_index('Ticker', inplace=True)
	return df



#MCap Weighted Returns Function
@st.cache(allow_output_mutation=True)
def mcap_weighted(df, groupby, rets_cols=['1D', '1W', '1M', '3M', '6M', '1Y', '2Y'], style=True):
    """
    MCap Weighted Returns:
    
    Groupby: Sector or Sub-Industry (Trendlyne)
    """    
    old_mcap = (1/(1+df[rets_cols]/100)).multiply(df['MCap(Cr)'], axis='rows')
    old = old_mcap.join(df['MCap(Cr)'])
    old.iloc[:,:-1] = -old.iloc[:,:-1].subtract(old.iloc[:,-1], axis='rows')

    change_sum = old.join(df[groupby], on=df.index.name).groupby(groupby).sum().iloc[:,:-1].round(2)
    old_sum = old_mcap.join(df[groupby], on=df.index.name).groupby(groupby).sum().round(2)
    mcap_weight = pd.DataFrame(df.groupby(groupby).sum()['MCap(Cr)']).merge(change_sum.divide(old_sum, axis='rows'), on=groupby)
    df1 = mcap_weight
    df1[rets_cols] = df1[rets_cols]*100
    stocks_count = df[['Name', groupby]].groupby(groupby).count()
    stocks_count.columns = ['Count']
    df1 = df1.join(stocks_count, on=groupby)
    subs = ['1D', '1W', '1M', '3M', '6M', '1Y', '2Y']

    if style==True:
        df1 = df1.fillna(0).sort_values(by='1D', ascending=False).style.format('{0:,.2f}%', subset=subs)\
                       .format('{0:,.0f}', subset=["MCap(Cr)"])\
                       .background_gradient(cmap='RdYlGn', subset=subs)
    else:
        df1 = df1.sort_values(by='1D', ascending=False)
    return df1


def sort_display(df):
	return df.sort_values(by='1D', ascending=False).style.format('{: .2f}', subset='Price')\
             .format('{: .2f}%', subset=df.columns[5:15])\
             .format('{: .2f}', subset='1Y Beta')\
             .format('{: .0f}', subset=df.columns[16:19])\
             .format('{: .2f}', subset=['OpRev Gr TTM(%)', 'Rev Gr YoY(%)', 'OPM Gr YoY(%)',
       'Net Profit TTM Gr(%)', 'PB Ratio', 'EPS TTM Gr(%)','MCap(Cr)'])\
             .background_gradient(cmap='RdYlGn', subset=df.columns[5:11])


#Importing Dataframe
df = import_data()


groupby = st.selectbox('Groupby: ', ['Sector', 'Industry'])
sector = st.multiselect('Select Sector: ', ['All'] + list(df['Sector'].unique()), ['All'])

if sector == ["All"]:
	industry = st.multiselect('Select Industry: ', ['All'] + list(df['Industry'].unique()), ['All'])
	if industry==["All"]:	
		if st.checkbox("Display All Stocks"):
			st.write(sort_display(df.fillna(0)))
		else:
			st.write(mcap_weighted(df, groupby))
	else:
		st.write(sort_display(df[df['Industry'].isin(industry)]))
elif sector !=["All"]:
	industry = st.multiselect('Select Industry: ', ['All'] + list(df[df['Sector'].isin(sector)]['Industry'].unique()), ['All'])
	if industry==["All"]:
		if st.checkbox("Display Sector's Underlying Constituents"):
			st.write(sort_display(df[df['Sector'].isin(sector)]))
		else:
			st.write(mcap_weighted(df[df['Sector'].isin(sector)], 'Industry'))
	else:
		st.write(sort_display(df[df['Industry'].isin(industry)]))	



st.write("### Plot Quick Stock Charts")
stock_name = st.selectbox('Select Ticker', list(df.index))
from nsepy import get_history
#prices = get_history(symbol=str(stock_name),
#                   start=date(date.today().year-5,date.today().month,date.today().day),
#                   end=date.today())
prices = pd.DataFrame(yf.download(str(stock_name)+".NS"))

def plot_chart(data):
    """
    Returns a Plotly Interactive Chart for the given timeseries data (price)
    data = price data for the ETFs (dataframe)
    """
    #df = ((((1+data.dropna().pct_change().fillna(0.00))).cumprod()-1)).round(4)
    df = pd.DataFrame(data['Close'])
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Price', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                      legend_title_text='Securities', yaxis_tickformat = '.0f', width=950, height=600)
    fig.update_traces(hovertemplate='Date: %{x} <br>Price: %{y:.2f}')
    fig.update_yaxes(automargin=True)
    return fig

def plot_ohlc(df):
	fig = go.Figure(data=go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
	fig.update_layout(xaxis_title='Date',
                      yaxis_title='Price', font=dict(family="Segoe UI, monospace", size=13, color="#7f7f7f"),
                      legend_title_text='Securities', title_text=str(stock_name)+" OHLC Chart", yaxis_tickformat = '.0f', width=950, height=600)
	#fig.update_traces(hovertemplate='Date: %{x} <br>Price: %{y:.2f}')
	fig.update_yaxes(automargin=True)
	return fig

if st.checkbox("Show OHLC Chart"):
	st.plotly_chart(plot_ohlc(prices))

