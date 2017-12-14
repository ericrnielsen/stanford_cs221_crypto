from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import sys
from datetime import datetime
from datetime import timedelta
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

################################################################################################
################################################################################################
# Setup variables to be used later 
################################################################################################
################################################################################################

# Overall time period
OLDEST_DATE = datetime.strptime('08/07/2015', '%m/%d/%Y')

# Train, dev, and test splits
TRAIN_START_DATE = OLDEST_DATE
TRAIN_END_DATE = datetime.strptime('03/21/2017', '%m/%d/%Y')
TRAIN_DATES = [TRAIN_START_DATE, TRAIN_END_DATE]

DEV_START_DATE = datetime.strptime('03/22/2017', '%m/%d/%Y') + timedelta(days=6) # Since MDP starts +6 days
DEV_END_DATE = datetime.strptime('07/27/2017', '%m/%d/%Y')
DEV_DATES = [DEV_START_DATE, DEV_END_DATE]

TEST_START_DATE = datetime.strptime('07/28/2017', '%m/%d/%Y') + timedelta(days=6) # Since MDP starts +6 days
TEST_END_DATE = datetime.strptime('11/30/2017', '%m/%d/%Y')
TEST_DATES = [TEST_START_DATE, TEST_END_DATE]

# Raw coin price data files
RAW_BTC = "data/coin_prices_raw/bitcoin_price.csv"
RAW_ETH = "data/coin_prices_raw/ethereum_price.csv"
RAW_LTC = "data/coin_prices_raw/litecoin_price.csv"
BITCOIN_DF = pd.read_csv(RAW_BTC)
ETHEREUM_DF = pd.read_csv(RAW_ETH)
LITECOIN_DF = pd.read_csv(RAW_LTC)
COIN_DFS = [BITCOIN_DF, ETHEREUM_DF, LITECOIN_DF]

# Investment
INVESTMENT = 10000      # Original investment amount
MIN_PERCENT = 0.05      # Min investment in any 1 coin
MAX_PERCENT = 0.90      # Max investment in any 1 coin
BITCOIN_PERCENT = 0.6   # Default portfolio % allocated to BTC
ETHEREUM_PERCENT = 0.2  # Default portfolio % allocated to ETH
LITECOIN_PERCENT = 0.2  # Default portfolio % allocated to LTC
COIN_PERCENTS = [BITCOIN_PERCENT, ETHEREUM_PERCENT, LITECOIN_PERCENT]
ALL_BITCOIN = [1., 0., 0.]      # All in on BTC
ALL_ETHEREUM = [0., 1., 0.]     # All in on ETH
ALL_LITECOIN = [0., 0., 1.]     # All in on LTC

# For plotting 
plt.rc('font', size=16)             # controls default text sizes
plt.rc('axes', titlesize=20)        # fontsize of the axes title
plt.rc('axes', labelsize=16)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=18)      # fontsize of the figure title

################################################################################################
################################################################################################
# Main baseline / oracle calculation functions
################################################################################################
################################################################################################

############################################################################
# Calculate final baseline portfolio profit (buy and hold)
############################################################################
def calc_baseline_profit(dates, investment, coin_percents, coin_dfs):

    # Split input
    start_date = dates[0]
    stop_date = dates[1]
    btc_percent = coin_percents[0]
    eth_percent = coin_percents[1]
    ltc_percent = coin_percents[2]
    btc_data = coin_dfs[0]
    eth_data = coin_dfs[1]
    ltc_data = coin_dfs[2]

    # Create date strings
    start_date_str = start_date.strftime('%b %d, %Y')
    stop_date_str = stop_date.strftime('%b %d, %Y')

    # Get start_date usd amount of each coin
    start_usd = [investment * btc_percent, investment * eth_percent, investment * ltc_percent]

    # Get start_date coin prices
    start_prices = get_prices(start_date, coin_dfs)

    # Get start_date coin amounts
    start_coins = usd_to_coins(start_usd, start_prices)

    # Get stop_date coin prices
    stop_prices = get_prices(stop_date, coin_dfs)

    # Get stop_date usd value
    stop_usd = coins_to_usd(start_coins, stop_prices)

    # Get ending total portfolio value and percent change
    end_value = sum(stop_usd)
    percent_change = 100 * ((end_value - investment) / investment)
    percent_sign = '+' if percent_change > 0 else '-'

    # Print results and return
    print "\n[BASELINE]"
    print "Initial investment on %s: $%s\nPortfolio value on %s: $%s\nPercent change: %s%s%%" % \
        (start_date_str, format(investment, ",.2f"), stop_date_str, format(end_value, ",.2f"), percent_sign, format(percent_change, ",.2f"))
    return end_value

############################################################################
# Calculate daily baseline portfolio values (called by model for plotting)
############################################################################
def get_baseline_values(which_data, baseline_type='normal'):

    # List storing daily portfolio values
    daily_values = []

    # Determine which data we're working with
    if which_data == 'Train':
        dates = TRAIN_DATES
    elif which_data == 'Dev':
        dates = DEV_DATES
    elif which_data == 'TrainDev':
        dates = [TRAIN_START_DATE, DEV_END_DATE]
    else:
        dates = TEST_DATES    

    # Split input
    start_date = dates[0]
    stop_date = dates[1]
    if baseline_type == 'normal':
        btc_percent, eth_percent, ltc_percent = COIN_PERCENTS
    elif baseline_type == 'bitcoin':
        btc_percent, eth_percent, ltc_percent = ALL_BITCOIN
    elif baseline_type == 'ethereum':
        btc_percent, eth_percent, ltc_percent = ALL_ETHEREUM
    else:
        btc_percent, eth_percent, ltc_percent = ALL_LITECOIN
    btc_data = COIN_DFS[0]
    eth_data = COIN_DFS[1]
    ltc_data = COIN_DFS[2]
    one_day = timedelta(days=1)

    # Get start_date coin amounts
    start_usd = [INVESTMENT * btc_percent, INVESTMENT * eth_percent, INVESTMENT * ltc_percent]
    start_prices = get_prices(start_date, COIN_DFS)
    num_coins = usd_to_coins(start_usd, start_prices)

    # Loop through all days between start_date and stop_date
    current_date = start_date - one_day
    while current_date < stop_date:
        current_date += one_day
        
        # Calculate current value of portfolio
        current_prices = get_prices(current_date, COIN_DFS)
        current_values = coins_to_usd(num_coins, current_prices)
        daily_values.append(sum(current_values))

    # Return
    return daily_values

############################################################################
# Calculate final oracle portfolio profit for 221 (all in on best coin each day)
############################################################################
def calc_oracle_profit(dates, investment, coin_percents, min_per, max_per, coin_dfs):

    # Split input
    start_date = dates[0]
    stop_date = dates[1]
    btc_percent = coin_percents[0]
    eth_percent = coin_percents[1]
    ltc_percent = coin_percents[2]
    btc_data = coin_dfs[0]
    eth_data = coin_dfs[1]
    ltc_data = coin_dfs[2]

    # Create date strings
    start_date_str = start_date.strftime('%b %d, %Y')
    stop_date_str = stop_date.strftime('%b %d, %Y')

    # Get start_date usd amount of each coin
    start_usd = [investment * btc_percent, investment * eth_percent, investment * ltc_percent]

    # Get start_date coin prices
    start_prices = get_prices(start_date, coin_dfs)

    # Get start_date coin amounts
    portfolio_coins = usd_to_coins(start_usd, start_prices)

    # Loop through all days
    current_date = start_date
    while current_date <= stop_date:

        # Get current and next day coin prices
        today_prices = get_prices(current_date, coin_dfs)
        tomorrow_prices = get_prices(current_date + timedelta(days=1), coin_dfs)

        # Get current usd value of portfolio
        today_usd = sum(coins_to_usd(portfolio_coins, today_prices))

        # Get price change (usd value and percent of coin price)
        change = map(operator.sub, tomorrow_prices, today_prices)
        percent = map(operator.div, change, today_prices)

        # Update portfolio distribution (90% going to winner, 5% going to others)
        if percent[0] >= percent[1] and percent[0] >= percent[2]:
            winner = 'btc'
        elif percent[1] >= percent[0] and percent[1] >= percent[2]:
            winner = 'eth'
        else:
            winner = 'ltc'
        portfolio_coins = oracle_update(today_usd, today_prices, winner, min_per, max_per)

        # Move days
        current_date = current_date + timedelta(days=1)

    # Get stop_date coin prices
    stop_prices = get_prices(stop_date, coin_dfs)

    # Get stop_date usd value
    stop_usd = coins_to_usd(portfolio_coins, stop_prices)

    # Get ending total portfolio value and percent change
    end_value = sum(stop_usd)
    percent_change = 100 * ((end_value - investment) / investment)
    percent_sign = '+' if percent_change > 0 else '-'

    # Print results and return
    print "\n[ORACLE]"
    print "Initial investment on %s: $%s\nPortfolio value on %s: $%s\nPercent change: %s%s%%" % \
        (start_date_str, format(investment, ",.2f"), stop_date_str, format(end_value, ",.2f"), percent_sign, format(percent_change, ",.2f"))
    return end_value

################################################################################################
################################################################################################
# Helper baseline / oracle calculation functions
################################################################################################
################################################################################################

############################################################################
# Convert dollar amount to coins
############################################################################
def usd_to_coins(usd_amounts, coin_prices):
    return [usd_amounts[0] / coin_prices[0], usd_amounts[1] / coin_prices[1], usd_amounts[2] / coin_prices[2]]

############################################################################
# Convert coin amount to usd
############################################################################
def coins_to_usd(coin_amounts, coin_prices):
    return [coin_amounts[0] * coin_prices[0], coin_amounts[1] * coin_prices[1], coin_amounts[2] * coin_prices[2]]

############################################################################
# Get coin prices on a specified day
############################################################################
def get_prices(query_date, coin_dfs):
    date_string = query_date.strftime('%b %d, %Y')
    btc_data = coin_dfs[0]
    eth_data = coin_dfs[1]
    ltc_data = coin_dfs[2]
    btc_row = btc_data[btc_data['Date'].str.match(date_string)].index[0]
    eth_row = eth_data[eth_data['Date'].str.match(date_string)].index[0]
    ltc_row = ltc_data[ltc_data['Date'].str.match(date_string)].index[0]
    return [float(btc_data.loc[btc_row]['Close']), float(eth_data.loc[eth_row]['Close']), float(ltc_data.loc[ltc_row]['Close'])]

############################################################################
# Update portfolio (oracle)
############################################################################
def oracle_update(usd_amount, coin_prices, winner, min_per, max_per):
    if winner == 'btc':
        btc_per = max_per
        eth_per = min_per
        ltc_per = min_per
    elif winner == 'eth':
        btc_per = min_per
        eth_per = max_per
        ltc_per = min_per
    else:
        btc_per = min_per
        eth_per = min_per
        ltc_per = max_per
    return [(usd_amount * btc_per) / coin_prices[0], (usd_amount * eth_per) / coin_prices[1], (usd_amount * ltc_per) / coin_prices[2]]

################################################################################################
################################################################################################
# Plotting model results
################################################################################################
################################################################################################

############################################################
# Plot portfolio value over time (1 trial)
############################################################
def plot_portfolio(which_data, values, values_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Baseline
    baseline = get_baseline_values(which_data)
    plt.plot(baseline, label='Baseline', c='red')
    # Our values
    plt.plot(values, label=values_label, c='darkgreen')
    # Save
    plt.legend(loc=0)
    plt.savefig("%s-SingleTrial-PortfolioValues.png" % (which_data))

############################################################
# Plot portfolio value over time, including daily coin choices (1 trial)
############################################################
def plot_portfolio_choice(which_data, values, choices, values_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    bitcoin = mpatches.Patch(color='gold', label='Bitcoin')
    ethereum = mpatches.Patch(color='green', label='Ethereum')
    litecoin = mpatches.Patch(color='grey', label='Litecoin')
    days = [ i for i in range(len(values)) ]
    # Plot
    value_plt, = plt.plot(values, label=values_label, c='lightgreen', zorder=1)
    colors = ['gold' if coin == 'Bitcoin' else 'green' if coin == 'Ethereum' else 'grey' for coin in choices]
    plt.scatter(days, values, c=colors, zorder=2)
    # Save
    plt.legend(handles=[value_plt, bitcoin, ethereum, litecoin], framealpha=0.3, loc=0)
    plt.savefig("%s-SingleTrial-PortfolioValues-CoinChoices.png" % (which_data))

############################################################
# Plot portfolio value over time, including daily coin choices, against baselines (1 trial)
############################################################
def plot_portfolio_choice_baseline(which_data, values, choices, values_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    bitcoin = mpatches.Patch(color='gold', label='Bitcoin')
    ethereum = mpatches.Patch(color='green', label='Ethereum')
    litecoin = mpatches.Patch(color='grey', label='Litecoin')
    days = [ i for i in range(len(values)) ]
    # Baselines
    baseline = get_baseline_values(which_data, 'normal')
    all_bitcoin = get_baseline_values(which_data, 'bitcoin')
    all_ethereum = get_baseline_values(which_data, 'ethereum')
    all_litecoin = get_baseline_values(which_data, 'litecoin')
    baseline_plt, = plt.plot(baseline, label='Baseline (buy and hold, 60% BTC, 20%% ETH, 20% LTC)', c='red', zorder=3, linewidth=2)
    bitcoin_plt, = plt.plot(all_bitcoin, label='Baseline (buy and hold, 100% BTC)', c='gold', zorder=3, linewidth=2)
    ethereum_plt, = plt.plot(all_ethereum, label='Baseline (buy and hold, 100%% ETH)', c='green', zorder=3, linewidth=2)
    litecoin_plt, = plt.plot(all_litecoin, label='Baseline (buy and hold, 100% LTC)', c='grey', zorder=3, linewidth=2)
    # Our values
    value_plt, = plt.plot(values, label=values_label, c='lightgreen', zorder=1)
    colors = ['gold' if coin == 'Bitcoin' else 'green' if coin == 'Ethereum' else 'grey' for coin in choices]
    plt.scatter(days, values, c=colors, zorder=2)
    # Save
    plt.legend(handles=[baseline_plt, bitcoin_plt, ethereum_plt, litecoin_plt, value_plt, bitcoin, ethereum, litecoin],
               framealpha=0.3, loc=0)
    plt.savefig("%s-SingleTrial-PortfolioValues-CoinChoices-Baselines.png" % (which_data))

############################################################
# Plot portfolio value over time (2+ trials)
############################################################
def plot_portfolio_multi(which_data, values, avg, values_label, avg_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    num_trials = len(values)
    # Our values
    i = 0
    for value in values:
        i += 1
        if i == 1: plt.plot(value, label=values_label, c='lightgreen', zorder=1)
        else: plt.plot(value, c='lightgreen', zorder=1)
    plt.plot(avg, label=avg_label, c='darkgreen', zorder=3)
    # Save
    plt.legend(loc=0)
    plt.savefig("%s-MultiTrial-PortfolioValues.png" % (which_data))

############################################################
# Plot portfolio value over time (2+ trials) and daily coin choices
############################################################
def plot_portfolio_multi_choice(which_data, values, choices, avg, values_label, avg_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    bitcoin = mpatches.Patch(color='gold', label='Bitcoin')
    ethereum = mpatches.Patch(color='green', label='Ethereum')
    litecoin = mpatches.Patch(color='grey', label='Litecoin')
    days = [ i for i in range(len(values[0])) ]
    # Plot
    i = 0
    for value in values:
        i += 1
        if i == 1: 
            value_plt, = plt.plot(value, label=values_label, c='lightgreen', zorder=1)
        else: 
            plt.plot(value, c='lightgreen', zorder=1)
        colors = ['gold' if coin == 'Bitcoin' else 'green' if coin == 'Ethereum' else 'grey' for coin in choices[i-1]]
        plt.scatter(days, value, c=colors, zorder=2)
    avg_plt, = plt.plot(avg, label=avg_label, c='blue', zorder=3, linewidth=3)
    # Save
    plt.legend(handles=[avg_plt, value_plt, bitcoin, ethereum, litecoin], framealpha=0.3, loc=0)
    plt.savefig("%s-MultiTrial-PortfolioValues-CoinChoices.png" % (which_data))

############################################################
# Plot expected MDP rewards
############################################################
def plot_rewards(which_data, values, expected, data_label, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    num_iterations = len(values)
    # Baseline
    baseline = get_baseline_values(which_data, 'normal')
    baseline_reward = baseline[-1] - baseline[0]
    plt.hlines(baseline_reward, xmin=0, xmax=num_iterations, colors='r', linestyles='dashed', label='Baseline', linewidth=2)
    # Our values
    iterations = [ i for i in range(num_iterations) ]
    plt.bar(iterations, values, align='center', color='darkgreen')
    plt.hlines(expected, xmin=0, xmax=num_iterations, colors='b', label='Expected Profit', linewidth=2)
    # Show
    plt.legend(loc=0)
    plt.savefig("%s-ExpectedRewards.png" % (which_data))

############################################################
# Plot only daily coin choice
############################################################
def plot_only_choice(which_data, daily_choices, portfolio_values, xlabel, ylabel, title):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    colors=['gold' if value=='Bitcoin' else 'green' if value=='Ethereum' else 'grey' for value in daily_choices]
    bitcoin = mpatches.Patch(color='gold', label='Bitcoin')
    ethereum = mpatches.Patch(color='green', label='Ethereum')
    litecoin = mpatches.Patch(color='grey', label='Litecoin')
    plt.legend(handles=[bitcoin, ethereum, litecoin])
    days = [ i for i in range(len(portfolio_values)) ]
    # Our values
    plt.plot(portfolio_values, c='lightblue', zorder=1)
    plt.scatter(days, portfolio_values, c=colors, zorder=2)
    # Show
    plt.savefig("%s-DailyChoice.png" % (which_data))

############################################################
# Plot baseline portfolio values
############################################################
def plot_baseline(time_period):
    # Setup
    plt.figure(figsize=(30, 15))
    plt.style.use('bmh')
    plt.xlabel('Day')
    plt.ylabel('Portfolio Value ($US)')
    plt.title('%s - Baseline Portfolio Value Over Time' % time_period)
    # Plot
    baseline = get_baseline_values(time_period, 'normal')
    all_bitcoin = get_baseline_values(time_period, 'bitcoin')
    all_ethereum = get_baseline_values(time_period, 'ethereum')
    all_litecoin = get_baseline_values(time_period, 'litecoin')
    baseline_plt, = plt.plot(baseline, label='Baseline (buy and hold, 60% BTC, 20% ETH, 20% LTC)', c='red', linewidth=2)
    bitcoin_plt, = plt.plot(all_bitcoin, label='Baseline (buy and hold, 100% BTC)', c='gold', linewidth=2)
    ethereum_plt, = plt.plot(all_ethereum, label='Baseline (buy and hold, 100% ETH)', c='green', linewidth=2)
    litecoin_plt, = plt.plot(all_litecoin, label='Baseline (buy and hold, 100% LTC)', c='grey', linewidth=2)
    # Show, then reset
    plt.legend(handles=[baseline_plt, bitcoin_plt, ethereum_plt, litecoin_plt], framealpha=0.3, loc=0)
    plt.savefig('%s - Baseline Portfolio Value Over Time' % time_period)

################################################################################################
################################################################################################
# Other
################################################################################################
################################################################################################

############################################################
# Update portfolio distribution
############################################################
def portfolio_update(usd_amount, coin_prices, coins, high_per, medium_per, low_per):

    btc_per = high_per if coins[0] == 'Bitcoin' else medium_per if coins[1] == 'Bitcoin' else low_per
    eth_per = high_per if coins[0] == 'Ethereum' else medium_per if coins[1] == 'Ethereum' else low_per
    ltc_per = high_per if coins[0] == 'Litecoin' else medium_per if coins[1] == 'Litecoin' else low_per

    # If using different states / actions
    #btc_per = action.count('Bitcoin') * 0.25
    #eth_per = action.count('Ethereum') * 0.25
    #ltc_per = action.count('Litecoin') * 0.25

    return ((usd_amount * btc_per) / coin_prices[0], (usd_amount * eth_per) / coin_prices[1], (usd_amount * ltc_per) / coin_prices[2])

############################################################
# Load crypto coin price data (from single file)
############################################################
def loadCsvData(fileName):
    # Open a file
    with open(fileName, 'rU') as f:
        reader = csv.reader(f)	
        # Set up dict and keys
        coinData = {}
        header = next(reader)
        for item in header:
            if not item == 'Date':
                coinData[item] = []
        # Add values
        for row in reader:
            for j in range(1, len(row)):
                coinData[header[j]] = [float(row[j])] + coinData[header[j]]	
    return coinData

############################################################
# Load crypto coin price data (from single file)
############################################################
def loadCsvData_multi(train_file, dev_file):
    # Initialize dict
    coinData = {}
    # Open dev file first (adding values newest to oldest)
    with open(dev_file, 'rU') as f1:
        reader = csv.reader(f1)	
        # Set up keys
        header = next(reader)
        for item in header:
            if not item == 'Date':
                coinData[item] = []
        # Add values
        for row in reader:
            for j in range(1, len(row)):
                coinData[header[j]] = [float(row[j])] + coinData[header[j]]	
    # Open train file second
    with open(train_file, 'rU') as f2:
        reader = csv.reader(f2)	
        # Add values
        header = next(reader)
        for row in reader:
            for j in range(1, len(row)):
                coinData[header[j]] = [float(row[j])] + coinData[header[j]]	         
    return coinData

################################################################################################
################################################################################################
# Main (only for testing purposes)
################################################################################################
################################################################################################
def main():

    # Calculate final baseline portfolio profit
    baseline_train_profit = calc_baseline_profit(TRAIN_DATES, INVESTMENT, COIN_PERCENTS, COIN_DFS)
    baseline_dev_profit = calc_baseline_profit(DEV_DATES, INVESTMENT, COIN_PERCENTS, COIN_DFS)
    baseline_test_profit = calc_baseline_profit(TEST_DATES, INVESTMENT, COIN_PERCENTS, COIN_DFS)

    # Calculate final oracle portfolio profit
    oracle_train_profit = calc_oracle_profit(TRAIN_DATES, INVESTMENT, COIN_PERCENTS, MIN_PERCENT, MAX_PERCENT, COIN_DFS)
    oracle_dev_profit = calc_oracle_profit(DEV_DATES, INVESTMENT, COIN_PERCENTS, MIN_PERCENT, MAX_PERCENT, COIN_DFS)
    oracle_test_profit = calc_oracle_profit(TEST_DATES, INVESTMENT, COIN_PERCENTS, MIN_PERCENT, MAX_PERCENT, COIN_DFS)

    # Build lists of baseline daily portfolio values
    baseline_train_values = get_baseline_values('Train')
    baseline_dev_values = get_baseline_values('Dev')
    baseline_test_values = get_baseline_values('Test')

    print baseline_train_profit
    print baseline_dev_profit
    print baseline_test_profit

    print oracle_train_profit
    print oracle_dev_profit
    print oracle_test_profit

    for item in baseline_train_values:
        print item
    for item in baseline_dev_values:
        print item
    for item in baseline_test_values:
        print item

if __name__ == '__main__':
	main()