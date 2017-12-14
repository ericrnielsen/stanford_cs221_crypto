import pandas as pd
from tqdm import tqdm
import sys
from datetime import datetime
from datetime import timedelta
import numpy as np
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

################################################################################################
################################################################################################
# Setup constant values to be used later 
################################################################################################
################################################################################################

# Time period
START_DATE = '07/08/2015'
END_DATE = '30/11/2017'

# Raw coin price data files
RAW_BTC = "coin_prices_raw/bitcoin_price.csv"
RAW_ETH = "coin_prices_raw/ethereum_price.csv"
RAW_LTC = "coin_prices_raw/litecoin_price.csv"

# Split coin price data files
FILE_PATH = "coin_prices_split/"
INPUT_FILE = "coin_data"
TRAIN = "_train.csv"
DEV = "_dev.csv"
TEST = "_test.csv"

# Train, dev, test splits
PERCENT_TRAIN = .70
PERCENT_DEV = .15
PERCENT_TEST = .15


################################################################################################
################################################################################################
# Create train, dev, and test files
################################################################################################
################################################################################################
def split_data(btc_df, eth_df, ltc_df, start_date, stop_date):

    # Setup
    data = []

    # Loop through price data
    for index, btc_row in btc_df.iterrows():

        # Get rows in ETH and LTC file
        eth_row = eth_df.loc[index]
        ltc_row = ltc_df.loc[index]

        # Get date
        date = datetime.strptime(btc_row['Date'], '%b %d, %Y')

        # Continue until stop date reached, then break once start date reached
        if (date > stop_date):
            continue
        elif (date <= start_date):
            break
        else:

            # Will store data in temp dict, then append to master list
            temp_dict = {}

            # Get date
            temp_dict['Date'] = btc_row['Date']

            # Get price
            temp_dict['BTC_Price'] = btc_row['Close']
            temp_dict['ETH_Price'] = eth_row['Close']
            temp_dict['LTC_Price'] = ltc_row['Close']

            # Get price change since previous day
            temp_dict['BTC_Price_Change'] = btc_row['Close'] - btc_df.loc[index + 1]['Close']
            temp_dict['ETH_Price_Change'] = eth_row['Close'] - eth_df.loc[index + 1]['Close']
            temp_dict['LTC_Price_Change'] = ltc_row['Close'] - ltc_df.loc[index + 1]['Close']

            # Get price volatility 
            temp_dict['BTC_Price_Volatility'] = btc_row['High'] - btc_row['Low']
            temp_dict['ETH_Price_Volatility'] = eth_row['High'] - eth_row['Low']
            temp_dict['LTC_Price_Volatility'] = ltc_row['High'] - ltc_row['Low']

            # Isolate volume
            temp_dict['BTC_Volume'] = locale.atoi(btc_row['Volume'])
            temp_dict['ETH_Volume'] = locale.atoi(eth_row['Volume'])
            temp_dict['LTC_Volume'] = locale.atoi(ltc_row['Volume'])

            # Get sharpe ration (if far enough away from start_date)
            if (date >= start_date + timedelta(days=6)):
                # Build price lists
                i = 0
                b_price_list, e_price_list, l_price_list = [], [], []
                for i in range(6):
                    b_price_list.append(btc_df.loc[index + i]['Close'])
                    e_price_list.append(eth_df.loc[index + i]['Close'])
                    l_price_list.append(ltc_df.loc[index + i]['Close'])
                # Use support function to calculate
                temp_dict['BTC_Sharpe'] = calcSharpeRatio(b_price_list)
                temp_dict['ETH_Sharpe'] = calcSharpeRatio(e_price_list)
                temp_dict['LTC_Sharpe'] = calcSharpeRatio(l_price_list)
            else:
                # Add 0s
                temp_dict['BTC_Sharpe'] = 0.
                temp_dict['ETH_Sharpe'] = 0.
                temp_dict['LTC_Sharpe'] = 0.   

            # Append to master list
            data.append(temp_dict)

    # Store the list of data in a Pandas dataframe, and determine train, dev, and test set indices
    results_df = pd.DataFrame(data)
    results_df = results_df.reindex(columns=["Date", "BTC_Price", "ETH_Price", "LTC_Price", \
                                    "BTC_Price_Change", "ETH_Price_Change", "LTC_Price_Change", \
                                    "BTC_Price_Volatility", "ETH_Price_Volatility", "LTC_Price_Volatility", \
                                    "BTC_Volume", "ETH_Volume", "LTC_Volume", "BTC_Sharpe", "ETH_Sharpe", "LTC_Sharpe"])
    num_rows = results_df.shape[0]
    end_test = int(num_rows * PERCENT_TEST)
    end_dev = int(num_rows * (PERCENT_TEST + PERCENT_DEV))
    end_train = num_rows - 1

    # Split dataframe
    test = results_df[results_df.index.isin(range(0,end_test))]
    dev = results_df[results_df.index.isin(range(end_test,end_dev+1))]
    train = results_df[results_df.index.isin(range(end_dev + 1,num_rows))]

    # Save to csv files
    train.to_csv(FILE_PATH + INPUT_FILE + TRAIN, index=False)
    dev.to_csv(FILE_PATH + INPUT_FILE + DEV, index=False)
    test.to_csv(FILE_PATH + INPUT_FILE + TEST, index=False)

################################################################################################
################################################################################################
# Calculate sharpe ratio (going back 6 days)
################################################################################################
################################################################################################
def calcSharpeRatio(price_list):
    daily_returns = np.zeros(len(price_list)-1)
    for i in range(len(price_list)-1):
        daily_returns[i] = price_list[i+1]/float(price_list[i])-1.0
    expected_return = np.mean(daily_returns)
    risk_free_rate = 0.0
    std_deviation = np.std(daily_returns)
    return (expected_return - risk_free_rate) / float(std_deviation)

################################################################################################
################################################################################################
# Calculate sharpe ratio (going back 6 days)
################################################################################################
################################################################################################
def main():

    # Define start and stop dates
    start_date = datetime.strptime(START_DATE, '%d/%m/%Y')
    stop_date = datetime.strptime(END_DATE, '%d/%m/%Y')

    # Load coin data
    bitcoin_df = pd.read_csv(RAW_BTC)
    ethereum_df = pd.read_csv(RAW_ETH)
    litecoin_df = pd.read_csv(RAW_LTC)
	
    # Split / save
    split_data(bitcoin_df, ethereum_df, litecoin_df, start_date, stop_date)


if __name__ == '__main__':
	main()