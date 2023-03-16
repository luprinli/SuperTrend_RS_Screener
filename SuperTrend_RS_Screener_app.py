import time
import pandas as pd
import numpy as np
import yfinance as yf
from yahoofinancials import YahooFinancials
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
import seaborn as sns
import streamlit as st


st.set_page_config(layout="wide")

def zero_crossing(data):
	return np.where(np.diff(np.sign(np.array(data))))[0]

def Supertrend(df, atr_period, multiplier):
    
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # calculate ATR
    price_diffs = [high - low, 
                   high - close.shift(), 
                   close.shift() - low]
    true_range = pd.concat(price_diffs, axis=1)
    true_range = true_range.abs().max(axis=1)
    # default ATR calculation in supertrend indicator
    atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean() 
    # df['atr'] = df['tr'].rolling(atr_period).mean()
    
    # HL2 is simply the average of high and low prices
    hl2 = (high + low) / 2
    # upperband and lowerband calculation
    # notice that final bands are set to be equal to the respective bands
    final_upperband = upperband = hl2 + (multiplier * atr)
    final_lowerband = lowerband = hl2 - (multiplier * atr)
    
    # initialize Supertrend column to True
    supertrend = [True] * len(df)
    
    for i in range(1, len(df.index)):
        curr, prev = i, i-1
        
        # if current close price crosses above upperband
        if close[curr] > final_upperband[prev]:
            supertrend[curr] = True
        # if current close price crosses below lowerband
        elif close[curr] < final_lowerband[prev]:
            supertrend[curr] = False
        # else, the trend continues
        else:
            supertrend[curr] = supertrend[prev]
            
            # adjustment to the final bands
            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                final_lowerband[curr] = final_lowerband[prev]
            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                final_upperband[curr] = final_upperband[prev]

        # to remove bands according to the trend direction
        if supertrend[curr] == True:
            final_upperband[curr] = np.nan
        else:
            final_lowerband[curr] = np.nan
    
    return pd.DataFrame({
        'Supertrend': supertrend,
        'Final_Lowerband': final_lowerband,
        'Final_Upperband': final_upperband
    }, index=df.index)
    
RS_period = 55
atr_period = 14
atr_multiplier = 2.7

fetch_period = '1y'

# ======================================================================================= # 
# List of Stokes from the Exchange
nasdaq = pd.read_csv("nasdaq_stock_list.csv")
nasdaq=nasdaq.sort_values(by=['Market Cap'], ascending=False)

nifty = pd.read_excel("nifty1000.xlsx")

st.title("SuperTrend & RS Screener")
st.write("Enter the your ticker symbols separated by commas.")
st.write("(Note: If you want to enter custom NSE stocks then add .NS after the ticker)")
custom_tickers = st.text_input("**:blue[Tickers of Your Choice:]**", "AAPL,MSFT,GOOG")

# ======================================================================================= # 
# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False

col1, col2, col3 = st.columns(3)

with col1:
    custom_scan = st.checkbox("Scan my custom stokes list only", key="disabled")
    US_scan = st.checkbox("Scan through NASDAQ ticker  list", key="US")
    IND_scan = st.checkbox("Scan through NSE ticker list", key="IND")

with col2:
	selected_scan_number = st.radio(
		"Comprehensive Scan (By Market Cap) 👇",
		["Top 100", "Top 500", "Top 1000", "Custom Range"],
		key="Scan_Number",
		label_visibility=st.session_state.visibility,
		horizontal=st.session_state.horizontal,
	)

with col3:
	if selected_scan_number== "Custom Range":
		with st.form(key="custom_scan_range_form"):
			scan_start, scan_end = st.slider("Custom Scan Range", 0, 7000, (0, 7000), 1)
			submit_button = st.form_submit_button(label="Submit")

# ======================================================================================= # 
if US_scan:
	st.markdown("<span style='color:red'>Great! You have selected the option to scan the US stocks.</span>",unsafe_allow_html=True)
	Ticker_List = nasdaq['Symbol'].tolist()
elif IND_scan:
	st.markdown("<span style='color:red'>Great! You have selected the option to scan the NSE stocks.</span>",unsafe_allow_html=True)
	Ticker_List = nifty['Symbol'].tolist()
else:
    st.markdown("<span style='color:red'>Great! You have selected your preferred stocks.</span>",unsafe_allow_html=True)
     # Split the ticker symbols and check each one
    Ticker_List = [custom_tickers.strip().upper() for custom_tickers in custom_tickers.split(",")]

		
if custom_scan:
	scan_start = 0
	scan_number = len(Ticker_List)
elif selected_scan_number:
	if (selected_scan_number == 'Top 100'):
		scan_start = 0
		scan_number = 100
	elif (selected_scan_number == 'Top 500'):
		scan_start = 0
		scan_number = 500
	elif (selected_scan_number == 'Top 1000'):
		scan_start = 0
		scan_number = 1000
	else:
		scan_start = scan_start
		scan_number = scan_end
else:
	scan_start = scan_start
	scan_number = scan_end

# ======================================================================================= # 
if US_scan:
	if (fetch_period == '2y'):
		df_desired_length = 504 
	elif (fetch_period == '1y'):
		df_desired_length = 251
	else:
		df_desired_length = 'none'
elif IND_scan:
	if (fetch_period == '2y'):
		df_desired_length = 498 
	elif (fetch_period == '1y'):
		df_desired_length = 250
	else:
		df_desired_length = 'none'
else:
	if (fetch_period == '2y'):
		df_desired_length = 504 
	elif (fetch_period == '1y'):
		df_desired_length = 251
	else:
		df_desired_length = 'none'

print ("[Scan_Start: Scan_End]", scan_start, scan_number)


# ======================================================================================= # 
base_index = 'DJI'
base_index = '^NSEI'
df_base_index = yf.download(base_index, period=fetch_period)


if st.button('Start the scan now!'):
	st.write(f'We have started the scan ranging [{scan_start+1} - {scan_number}]')
	for tick in range(scan_start, scan_number):
		ticker = Ticker_List[tick]

		if IND_scan:
			ticker = ticker+".NS"
		
		time.sleep(0.2) # Sleep for 0.2 seconds

		print ("Now Accessing the Symbol:", tick, ticker)
		st.write("Now Accessing the Symbol:", tick+1, ticker)
		
		df = yf.download(ticker, period='1y')
		
		if (len(df) != df_desired_length):
			print("Warning! Change the length to:", len(df))
			st.write("Warning! Change the length to:", len(df))

		else:
			df['Vwap'] = (df['Volume']*(df['High']+df['Low']+df['Close'])/3).cumsum() / df['Volume'].cumsum()
			df['sma'] = df['Close'].rolling(window=20).mean()
			date_index=[]
			hist_RS=[]
			for i in range (1,90):
				base_index_RS = df_base_index['Close'][-i]/df_base_index['Close'][-RS_period-i]
				ticker_RS = df['Close'][-i]/df['Close'][-RS_period-i]
				RS = (ticker_RS/base_index_RS)-1
				RS = round(RS, 3)
				date_index.append(-i)
				hist_RS.append(RS)
			
			df_RS = pd.DataFrame({'Day':date_index, 'RS':hist_RS})
			df_RS['sma_RS'] = df_RS['RS'].rolling(window=7).mean()
			df_RS['sg_RS'] = savgol_filter(df_RS['RS'], window_length = 9, polyorder = 1)
			
			cross_points = df_RS.index[zero_crossing(df_RS['sg_RS'])]
			cross_vals = df_RS.loc[cross_points]
			print(cross_vals)
			
			supertrend = Supertrend(df, atr_period, atr_multiplier)
			df = df.join(supertrend)
		
			if (len(cross_vals) > 0):	
				last_crossx = int(cross_vals['Day'].head(1))
				last_crossy = float(cross_vals['sg_RS'].head(1))

				print (last_crossx, round(last_crossy,2))
				
			if not supertrend['Supertrend'][-2] and supertrend['Supertrend'][-1]:
				if (float(df_RS['sg_RS'].head(1)) > 0):
					
					fig = plt.figure(figsize=(16, 9))
					grid = plt.GridSpec(9, 5, wspace = 0.2, hspace = 1.5)
					
					ax1 = fig.add_subplot(grid[0:3, 0:4])
					ax1.plot(df['Close'].tail(90), color='green', lw=1.0, label='Close')
					ax1.plot(df['Vwap'].tail(90), color='red', lw=0.8, label='VWAP')
					ax1.plot(df['sma'].tail(90), color='black', lw=0.8, label='SMA20')
					ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
					plt.xticks(rotation=30)
					ax1.legend(frameon=False)
					plt.title(str(tick)+". "+ticker+" ("+str(last_crossx)+"d): "+str(round(df['Close'][last_crossx],2))+"/"+str(round(df['Close'][-1],2)), color='red', fontsize=16)

					ax2 = fig.add_subplot(grid[0:3, 4], sharey=ax1)
					sns.distplot(df['Vwap'], kde=True, vertical=True, bins=15, ax=ax2, label='VWAP', color='red')
					sns.distplot(df['Close'], kde=True, vertical=True, bins=20, ax=ax2, label='Close', color='green')
					ax2.tick_params(axis='y', which='both', labelleft=False, labelright=True)
					ax2.axes.xaxis.set_visible(False)
					ax2.legend(frameon=False)
					
					ax3 = fig.add_subplot(grid[3:6, 0:4])
					ax3.plot (df_RS['Day'], df_RS['RS'], color='blue', lw=0.8, label='RS')
					ax3.plot (df_RS['Day'], df_RS['sg_RS'], color='black', lw=2)
					ax3.axhline(0, color='red')
					ax3.axvline(x=last_crossx, ls='--', color='cyan')
					ax3.legend(frameon=False)
									
					ax7 = fig.add_subplot(grid[6:9, 0:4])
					ax7.plot(df['Close'].tail(90), label='Close')
					ax7.plot(df['Final_Lowerband'].tail(90), 'g')
					ax7.plot(df['Final_Upperband'].tail(90), 'r',)
					ax7.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
					ax7.legend(frameon=False)
					plt.xticks(rotation=30)
									
					ax8 = fig.add_subplot(grid[6:9, 4])
					ax8.set_axis_off()
					ax8.text(0.13, 0.05, 'SL:%s'%(round(float(df['Final_Lowerband'].tail(1)),2)), fontsize=12, bbox={'facecolor': 'lightcoral', 'pad': 12, 'alpha': 0.5})
					ax8.text(0.13, 0.45, '5%%-TP:%s '%(round(1.05*float(df['Close'].tail(1)),2)), fontsize=12, bbox={'facecolor': 'lime', 'pad': 12, 'alpha': 0.5})
					ax8.text(0.13, 0.85, '10%%-TP:%s '%(round(1.1*float(df['Close'].tail(1)),2)), fontsize=12, bbox={'facecolor': 'green', 'pad': 12, 'alpha': 0.5})
									
					st.pyplot(fig)
									
