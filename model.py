import util_221hw, math, random, sys
from collections import defaultdict
from util_221hw import ValueIteration
import time
import numpy as np
from util_custom import *

################################################################################################
################################################################################################
# Setup constant values to be used later 
################################################################################################
################################################################################################

# Time period
START_DATE = '07/08/2015'
END_DATE = '30/11/2017'

# Raw coin price data files
RAW_BTC = "data/coin_prices_raw/bitcoin_price.csv"
RAW_ETH = "data/coin_prices_raw/ethereum_price.csv"
RAW_LTC = "data/coin_prices_raw/litecoin_price.csv"

# Split coin price data files
TRAIN_DATA = "data/coin_prices_split/coin_data_train.csv"
DEV_DATA = "data/coin_prices_split/coin_data_dev.csv"
TEST_DATA = "data/coin_prices_split/coin_data_test.csv"

# Portfolio info
START_VALUE = 10000     # Initial investment
HIGH = 0.6      # High portfolio % assigned to 1 coin
MEDIUM = 0.3    # Medium portfolio % assigned to 1 coin
LOW = 0.1       # Low portfolio % assigned to 1 coin

# MDP training exploration
SINGLE_EXPLORE = 'Single'           # Constant exploration probability
SINGLE_PROBS = [0.5]          
MULTI_EXPLORE = 'Multi'             # Decreasing exploration probability
MULTI_PROBS = [1, 0.5, 0.1]   

####################################################################################
####################################################################################
# MDP
####################################################################################
####################################################################################
class CryptoTrading(util_221hw.MDP):
    def __init__(self, coinData, startValue, high_per, medium_per, low_per):
        """
        coinData: dictionary with all coin price data needed for training
        startValue: starting value of the portfolio
        startDay: Starting day (default to 6 to allow enough history)
        """
        self.coinData = coinData
        self.startValue = startValue
        self.startDay = 6
        self.portfolio_vals = []
        self.portfolio_vals_single_iter = []
        self.daily_choice = []
        self.high_per = high_per
        self.medium_per = medium_per
        self.low_per = low_per

        # For 81 possible actions; Tried but worse results and long run time
        # Each appearance of a coin name in the action equates to 25% of the portfolio being invested in the coin
        #action_list = []
        #for first in range (3):
        #    for second in range (3):
        #        for third in range (3):
        #            for fourth in range (3):
        #                coin1 = 'Bitcoin' if first == 0 else 'Ethereum' if first == 1 else 'Litecoin'
        #                coin2 = 'Bitcoin' if second == 0 else 'Ethereum' if second == 1 else 'Litecoin'
        #                coin3 = 'Bitcoin' if third == 0 else 'Ethereum' if third == 1 else 'Litecoin'
        #                coin4 = 'Bitcoin' if fourth == 0 else 'Ethereum' if fourth == 1 else 'Litecoin'
        #                action = (coin1, coin2, coin3, coin4)
        #                action_list.append(action)
        #self.actions = action_list

    def startState(self):

        # Set portfolio_vals_single_iter to None
        self.portfolio_vals_single_iter = []
        self.daily_choice = []

        # Buy enough of each coin so that 1/3 of the portfolio is spent on each
        startBitcoin = float(self.startValue/self.coinData['BTC_Price'][self.startDay]) / 3
        startEthereum = float(self.startValue/self.coinData['ETH_Price'][self.startDay]) / 3
        startLitecoin = float(self.startValue/self.coinData['LTC_Price'][self.startDay]) / 3

        # Create a tuple of the past 7 days history for each to pass in state
        b_price = tuple(self.coinData['BTC_Price'][:self.startDay+1])
        e_price = tuple(self.coinData['ETH_Price'][:self.startDay+1])
        l_price = tuple(self.coinData['LTC_Price'][:self.startDay+1])

        # Adding new features
        b_change = tuple(self.coinData['BTC_Price_Change'][:self.startDay+1])
        e_change = tuple(self.coinData['ETH_Price_Change'][:self.startDay+1])
        l_change = tuple(self.coinData['LTC_Price_Change'][:self.startDay+1])

        b_volatility = tuple(self.coinData['BTC_Price_Volatility'][:self.startDay+1])
        e_volatility = tuple(self.coinData['ETH_Price_Volatility'][:self.startDay+1])
        l_volatility = tuple(self.coinData['LTC_Price_Volatility'][:self.startDay+1])

        b_volume = tuple(self.coinData['BTC_Volume'][:self.startDay+1])
        e_volume = tuple(self.coinData['ETH_Volume'][:self.startDay+1])
        l_volume = tuple(self.coinData['LTC_Volume'][:self.startDay+1])

        b_sharpe = tuple(self.coinData['BTC_Sharpe'][:self.startDay+1])
        e_sharpe = tuple(self.coinData['ETH_Sharpe'][:self.startDay+1])
        l_sharpe = tuple(self.coinData['LTC_Sharpe'][:self.startDay+1])   

        b_229 = tuple(self.coinData['229_BTC'][:self.startDay+1])
        e_229 = tuple(self.coinData['229_ETH'][:self.startDay+1])
        l_229 = tuple(self.coinData['229_LTC'][:self.startDay+1])  		

        state_data = {  'b_price':b_price, 'e_price':e_price, 'l_price':l_price,
                        'b_change':b_change, 'e_change':b_change, 'l_change':b_change,
                        'b_volatility':b_volatility, 'e_volatility':e_volatility, 'l_volatility':l_volatility,
                        'b_volume':b_volume, 'e_volume':e_volume, 'l_volume':l_volume,
                        'b_sharpe':b_sharpe, 'e_sharpe':e_sharpe, 'l_sharpe':l_sharpe, 
                        'b_229':b_229, 'e_229':e_229, 'l_229':l_229  }

        return (self.startDay, state_data,
                (startBitcoin, startEthereum, startLitecoin), self.startValue)

    # Now there are 6 possible actions, corresponding to 6 different portfolio distributions
    # In each action (tuple), first = high investment %, second = medium, third = low
    def actions(self, state):
        return [('Bitcoin', 'Ethereum', 'Litecoin'),
                ('Bitcoin', 'Litecoin', 'Ethereum'),
                ('Ethereum', 'Bitcoin', 'Litecoin'),
                ('Ethereum', 'Litecoin', 'Bitcoin'),
                ('Litecoin', 'Bitcoin', 'Ethereum'),
                ('Litecoin', 'Ethereum', 'Bitcoin')]

    # Defines movements between states
    def succAndProbReward(self, state, action):

        # Setup high, medium, and low investment percentages
        high_per = self.high_per
        medium_per = self.medium_per
        low_per = self.low_per

        # Unpack the state for easy handling
        current_day, state_data, num_coins, portfolio_value = state

        # Unpack the num_coins tuple and price_history tuple
        b_price, e_price, l_price = state_data['b_price'], state_data['e_price'], state_data['l_price']
        b_change, e_change, l_change = state_data['b_change'], state_data['e_change'], state_data['l_change']
        b_volatility, e_volatility, l_volatility = state_data['b_volatility'], state_data['e_volatility'], state_data['l_volatility']
        b_volume, e_volume, l_volume = state_data['b_volume'], state_data['e_volume'], state_data['l_volume']
        b_sharpe, e_sharpe, l_sharpe = state_data['b_sharpe'], state_data['e_sharpe'], state_data['l_sharpe']
        b_229, e_229, l_229 = state_data['b_229'], state_data['e_229'], state_data['l_229']
        numBitcoin, numEthereum, numLitecoin = num_coins

        # Initialize results array, return immediately if end of sim has been reached
        results = []
        if current_day == len(self.coinData['LTC_Price']): return results

        # Set the next_day value:
        next_day = current_day + 1

        # "Sell" all coins (will be purchased again below), calculate new portfolio value
        LitecoinProfit = float(numLitecoin * self.coinData['LTC_Price'][current_day])
        BitcoinProfit = float(numBitcoin * self.coinData['BTC_Price'][current_day])
        EthereumProfit = float(numEthereum * self.coinData['ETH_Price'][current_day])
        nextValue = LitecoinProfit + BitcoinProfit + EthereumProfit

        # Get current price for each coin
        coin_prices =  (self.coinData['BTC_Price'][current_day],
                        self.coinData['ETH_Price'][current_day],
                        self.coinData['LTC_Price'][current_day])

        # "Buy" back coins according to action
        newBitcoin, newEthereum, newLitecoin = portfolio_update(nextValue, coin_prices, action, high_per, medium_per, low_per)

        # Unpack the coin data (stored in the MDP state)
        b_price, e_price, l_price = list(b_price), list(e_price), list(l_price)
        b_change, e_change, l_change = list(b_change), list(e_change), list(l_change)
        b_volatility, e_volatility, l_volatility = list(b_volatility), list(e_volatility), list(l_volatility)
        b_volume, e_volume, l_volume = list(b_volume), list(e_volume), list(l_volume)
        b_sharpe, e_sharpe, l_sharpe = list(b_sharpe), list(e_sharpe), list(l_sharpe)
        b_229, e_229, l_229 = list(b_229), list(e_229), list(l_229)

        # Remove the oldest data for each data type for each coin, add new data
        b_price.pop(0); e_price.pop(0); l_price.pop(0)
        b_change.pop(0); e_change.pop(0); l_change.pop(0)
        b_volatility.pop(0); e_volatility.pop(0); l_volatility.pop(0)
        b_volume.pop(0); e_volume.pop(0); l_volume.pop(0)
        b_sharpe.pop(0); e_sharpe.pop(0); l_sharpe.pop(0)
        b_229.pop(0); e_229.pop(0), l_229.pop(0)
        if current_day + 1 != len(self.coinData['LTC_Price']):
            # Price
            b_price += [self.coinData['BTC_Price'][next_day]]
            e_price += [self.coinData['ETH_Price'][next_day]]
            l_price += [self.coinData['LTC_Price'][next_day]]
            # Price Change
            b_change += [self.coinData['BTC_Price_Change'][next_day]]
            e_change += [self.coinData['ETH_Price_Change'][next_day]]
            l_change += [self.coinData['LTC_Price_Change'][next_day]]
            # Price Volatility
            b_volatility += [self.coinData['BTC_Price_Volatility'][next_day]]
            e_volatility += [self.coinData['ETH_Price_Volatility'][next_day]]
            l_volatility += [self.coinData['LTC_Price_Volatility'][next_day]]
            # Volume
            b_volume += [self.coinData['BTC_Volume'][next_day]]
            e_volume += [self.coinData['ETH_Volume'][next_day]]
            l_volume += [self.coinData['LTC_Volume'][next_day]]
            # Sharpe ratio
            b_sharpe += [self.coinData['BTC_Sharpe'][next_day]]
            e_sharpe += [self.coinData['ETH_Sharpe'][next_day]]
            l_sharpe += [self.coinData['LTC_Sharpe'][next_day]]
            # 229 Output
            b_229 += [self.coinData['229_BTC'][next_day]]
            e_229 += [self.coinData['229_ETH'][next_day]]
            l_229 += [self.coinData['229_LTC'][next_day]]

        # Convert price data back to tuple format
        state_data = {  'b_price':tuple(b_price), 'e_price':tuple(e_price), 'l_price':tuple(l_price),
                        'b_change':tuple(b_change), 'e_change':tuple(b_change), 'l_change':tuple(b_change),
                        'b_volatility':tuple(b_volatility), 'e_volatility':tuple(e_volatility), 'l_volatility':tuple(l_volatility),
                        'b_volume':tuple(b_volume), 'e_volume':tuple(e_volume), 'l_volume':tuple(l_volume),
                        'b_sharpe':tuple(b_sharpe), 'e_sharpe':tuple(e_sharpe), 'l_sharpe':tuple(l_sharpe),
                        'b_229':tuple(b_229), 'e_229':tuple(e_229), 'l_229':tuple(l_229)  }

        # The first element of action tuple is the highest percentage choice for the day
        # Unpack the action for ease of understanding and use
        highest_per_coin, medium_per_coin, low_per_coin = action

        # Set the reward and factor in transaction cost
        if (True): # incorporate transaction cost
            fee = 0.002 # 0.2%

            # Make sure we have already made a past choice to compare too
            if len(self.daily_choice) >= 1:
                if highest_per_coin == self.daily_choice[-1]:
                    #print("Hold option")
                    reward = nextValue - portfolio_value
                else:
                    transaction_cost = nextValue * fee
                    #print("Transaction Cost: {}".format(transaction_cost))
                    reward = nextValue - portfolio_value - transaction_cost
            else:
                transaction_cost = nextValue * fee
                reward = nextValue - portfolio_value - transaction_cost

        else: # Don't incorporate transaction cost
            reward = nextValue - portfolio_value

        # Save daily choice for highest investment percentage
        self.daily_choice.append(highest_per_coin)

        # Add the new state, prob and reward to results
        results.append(((next_day, state_data, (newBitcoin, newEthereum, newLitecoin), nextValue), 1, reward))

        # Save portfolio values to plot later
        self.portfolio_vals.append(nextValue)
        self.portfolio_vals_single_iter.append(nextValue)

        return results
        
    def discount(self):
        return 1

####################################################################################
####################################################################################
# Q-LEARNING
####################################################################################
####################################################################################

############################################################
# Q-learning class
############################################################
class QLearningAlgorithm(util_221hw.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for feature, value in self.featureExtractor(state, action):
            score += self.weights[feature] * value
        return score

    # Determine an action given a state.
    def getAction(self, state):
        self.numIters += 1
        # Will explore randomly during training
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        # Will choose action based on Q-values during testing
        else:
            best = max((self.getQ(state, action), action) for action in self.actions(state))[1]
            #print best
            return best

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        #return 0.000001 / math.sqrt(self.numIters)
        return 1.0 / math.sqrt(self.numIters)

    # Update weights
    def incorporateFeedback(self, state, action, reward, newState):
	
        # Check if s is a terminal state and return if it is
        if newState == None: return

        eta = self.getStepSize()
        Q_opt = self.getQ(state, action)
        gamma = self.discount
        V_opt = max((self.getQ(newState, next_action), next_action) for next_action in self.actions(newState))[0]

        for feature, val in self.featureExtractor(state, action):
            self.weights[feature] -= eta * (Q_opt - (reward + (gamma * V_opt))) * val

############################################################
# Return a list of (feature key, feature value) pairs.
############################################################
def FeatureExtractor(state, action):

    # Setup
    day, state_data, numCoins, portfolioValue = state
    features = []

    # Get lists from coinData dictionary
    b_price, e_price, l_price = list(state_data['b_price']), list(state_data['e_price']), list(state_data['l_price'])
    b_change, e_change, l_change = list(state_data['b_change']), list(state_data['e_change']), list(state_data['l_change'])
    b_volatility, e_volatility, l_volatility = list(state_data['b_volatility']), list(state_data['e_volatility']), list(state_data['l_volatility'])
    b_volume, e_volume, l_volume = list(state_data['b_volume']), list(state_data['e_volume']), list(state_data['l_volume'])
    b_sharpe, e_sharpe, l_sharpe = list(state_data['b_sharpe']), list(state_data['e_sharpe']), list(state_data['l_sharpe'])
    b_229, e_229, l_229 = list(state_data['b_229']), list(state_data['e_229']), list(state_data['l_229'])
    num_days = len(b_price)
	
    # Add PRICE features (for past 6 days)
    if (True):
        #feature_range = range(1)
        feature_range = range(num_days)
        for i in feature_range:
            # Normalize
            price_sum = abs(b_price[i]) + abs(e_price[i]) + abs(l_price[i])
            if price_sum == 0: price_sum = 1
            features.append(((action, 'b_price' + str(i)), b_price[i]/price_sum))
            features.append(((action, 'e_price' + str(i)), e_price[i]/price_sum))
            features.append(((action, 'l_price' + str(i)), l_price[i]/price_sum))

    # Add PRICE CHANGE features (for past 6 days)
    if (True):
        #feature_range = range(1)
        feature_range = range(num_days)
        for i in feature_range:
            # Normalize
            change_sum = abs(b_change[i]) + abs(e_change[i]) + abs(l_change[i])
            if change_sum == 0: change_sum = 1
            # Add feature
            features.append(((action, 'b_change' + str(i)), b_change[i]/change_sum))
            features.append(((action, 'e_change' + str(i)), e_change[i]/change_sum))
            features.append(((action, 'l_change' + str(i)), l_change[i]/change_sum))

    # Add PRICE VOLATILITY features (for past 6 days)
    if (False):
        #feature_range = range(1)
        feature_range = range(num_days)
        for i in feature_range:
            # Normalize            
            volatility_sum = abs(b_volatility[i]) + abs(e_volatility[i]) + abs(l_volatility[i])
            if volatility_sum == 0: volatility_sum = 1
            # Add feature            
            features.append(((action, 'b_volatility' + str(i)), b_volatility[i]/volatility_sum))
            features.append(((action, 'e_volatility' + str(i)), e_volatility[i]/volatility_sum))
            features.append(((action, 'l_volatility' + str(i)), l_volatility[i]/volatility_sum))

    # Add VOLUME features (for past 6 days)
    if (False):
        #feature_range = range(1)
        feature_range = range(num_days)
        for i in feature_range:
            # Normalize            
            volume_sum = abs(b_volume[i]) + abs(e_volume[i]) + abs(l_volume[i])
            if volume_sum == 0: volume_sum = 1
            # Add feature            
            features.append(((action, 'b_volume' + str(i)), b_volume[i]/volume_sum))
            features.append(((action, 'e_volume' + str(i)), e_volume[i]/volume_sum))
            features.append(((action, 'l_volume' + str(i)), l_volume[i]/volume_sum))
			
    # Add SHARPE RATIO features (calculated based on past num_days days)
    if (True):
        feature_range = range(1)
        #feature_range = range(num_days)
        for i in feature_range:
            # Normalize
            sharpe_sum = abs(b_sharpe[i]) + abs(e_sharpe[i]) + abs(l_sharpe[i])
            if sharpe_sum == 0: sharpe_sum = 1
            # Add feature            
            features.append(((action, 'b_sharpe'), b_sharpe[i]/sharpe_sum))
            features.append(((action, 'e_sharpe'), e_sharpe[i]/sharpe_sum))
            features.append(((action, 'l_sharpe'), l_sharpe[i]/sharpe_sum))
			
    # Add 229 OUTPUT features (prediction of future price change)
    if (False):
        feature_range = range(1)
        for i in feature_range:
            features.append(((action, 'b_229'), b_229[i]))
            features.append(((action, 'e_229'), e_229[i]))
            features.append(((action, 'l_229'), l_229[i]))

    return features
	
####################################################################################
####################################################################################
# TRAINING / TESTING
####################################################################################
####################################################################################

############################################################
# Training the MDP
############################################################
def trainModel(MDP, featureExtractor, startValue, iterations, explore_type, single_explore, multi_explore, verbose=True):
	
    # Setup
    start = time.time()
    QLAlgo = QLearningAlgorithm(MDP.actions, MDP.discount(), featureExtractor)

    # Choose between single / multiple explore probabilities
    if explore_type == 'Single': probs = single_explore
    else: probs = multi_explore
	
    # Train
    QL_rewards = []
    for prob in probs:
        QLAlgo.explorationProb = prob
        if verbose: print "Training with exploration probability: %.2f" % prob
        QL_rewards += util_221hw.simulate(MDP, QLAlgo, numTrials = iterations)
		
    # Get expected reward
    QL_expReward = sum(QL_rewards)/float(len(QL_rewards))
    checkpoint_2 = time.time()
    if verbose:
        print 'QL training took ', str(checkpoint_2-start), 'seconds'
        print 'QL_expTrainReward: ', QL_expReward

    # Return
    return {'QLAlgo': QLAlgo,
            'Final Reward': QL_expReward,
            'All Rewards': QL_rewards,
            'Daily Choices': MDP.daily_choice,
            'Portfolio Value Single': MDP.portfolio_vals_single_iter,
            'Portfolio Value All': MDP.portfolio_vals}

############################################################
# Running the MDP on dev/test data
############################################################
def runModel(MDP, QLAlgo, featureExtractor, startValue, iterations, verbose=True):
	
    # Setup and test
    start = time.time()
    QLAlgo.explorationProb = 0
    QL_rewards = util_221hw.simulate(MDP, QLAlgo, numTrials = iterations)
    QL_expReward = sum(QL_rewards)/float(len(QL_rewards))
    end = time.time()
    if verbose:
        print 'QL testing took ', str(end-start), 'seconds'
        print 'QL_expTestReward: ', QL_expReward

    # Return
    return {'QLAlgo': QLAlgo,
            'Final Reward': QL_expReward,
            'All Rewards': QL_rewards,
            'Daily Choices': MDP.daily_choice,
            'Portfolio Value Single': MDP.portfolio_vals_single_iter,
            'Portfolio Value All': MDP.portfolio_vals}

####################################################################################
####################################################################################
# MAIN FUNCTION
####################################################################################
####################################################################################	
def main():

    ############################################################
    # Setup
    ############################################################

    # Load coin data
    coinData_train = loadCsvData(TRAIN_DATA)
    coinData_dev = loadCsvData(DEV_DATA)
    coinData_train_combined = loadCsvData_multi(TRAIN_DATA, DEV_DATA)
    coinData_test = loadCsvData(TEST_DATA)

    # Choose if running model for validation (dev set) or final testing (test set)
    #run_type = 'Validation'
    run_type = 'Test'

    # If validating, train on train, run on dev
    if (run_type == 'Validation'):
        training_data = coinData_train
        running_data = coinData_dev
        training_data_name = 'Train'
        running_data_name = 'Dev'
    # Else testing, train on train+dev, run on test
    else:
        training_data = coinData_train_combined
        running_data = coinData_test
        training_data_name = 'TrainDev'
        running_data_name = 'Test'

    # Choose training exploration type (constant vs decaying)
    #explore_type = SINGLE_EXPLORE
    explore_type = MULTI_EXPLORE

    # Choose number of training iterations (is 3x when using MULTI_EXPLORE, so adjust if necessary)
    train_iterations = 3000
    if (explore_type == MULTI_EXPLORE):
        train_iterations = int(train_iterations / 3.)

    ########################################################################################
    # [1] DEFAULT MODEL PERFORMANCE: Train MDP, then test on either dev or test sets
    ########################################################################################
    if (True):

        # Create train MDP, then train
        print 'Creating Crypto Trading MDP...'
        cryptoMDP = CryptoTrading(training_data, START_VALUE, HIGH, MEDIUM, LOW)
        print 'Running Q-Learning...'
        train_results = trainModel(cryptoMDP, FeatureExtractor, START_VALUE, train_iterations, explore_type, SINGLE_PROBS, MULTI_PROBS)
        QLAlgo = train_results['QLAlgo']

        # Swap MDP data to dev/test data, then run
        print '\nSwapping data in Crypto Trading MDP...'
        cryptoMDP.coinData = running_data
        print 'Testing on %s set...' % run_type
        runModel(cryptoMDP, QLAlgo, FeatureExtractor, START_VALUE, 1)

    ########################################################################################
    # [2] Plot baseline portfolio values for all time periods
    ########################################################################################
    if (False):

        # Loop through all time periods
        time_periods = ['Train', 'Dev', 'TrainDev', 'Test']
        for period in time_periods:
            plot_baseline(period)

    ########################################################################################
    # [3] Train MDP, test, print final rewards, (selectively) plot
    ########################################################################################
    if (False):

        # Train
        cryptoMDP = CryptoTrading(training_data, START_VALUE, HIGH, MEDIUM, LOW)
        train_results = trainModel(cryptoMDP, FeatureExtractor, START_VALUE, train_iterations, explore_type, SINGLE_PROBS, MULTI_PROBS)
        QLAlgo = train_results['QLAlgo']

        # Run
        cryptoMDP.coinData = running_data
        run_results = runModel(cryptoMDP, QLAlgo, FeatureExtractor, START_VALUE, 1)

        # Plot train results
        if (True):
            plot_portfolio(which_data=training_data_name, values=train_results['Portfolio Value Single'], 
                        values_label="Training Set", xlabel="Day", ylabel="Portfolio Value ($US)",
                        title="[Training] Portfolio Value vs. Day")
        if (True):
            plot_portfolio(which_data=training_data_name, values=train_results['Portfolio Value All'],
                        values_label="Training Set", xlabel="Day", ylabel="Portfolio Value ($US)",
                        title="[Training] Portfolio Value (All Iterations) vs. Day")
        if (True):
            plot_rewards(which_data=training_data_name, values=train_results['All Rewards'], expected=train_results['Final Reward'],
                        values_label="Final Profit", xlabel="Iterations", ylabel="Final Profit ($US)",
                        title="[Training] Final Profit vs. Training Iterations")
        if (True):
            plot_portfolio_choice_baseline(which_data=training_data_name, values=train_results['Portfolio Value Single'], 
                        choices=train_results['Daily Choices'], values_label="Final Train Results", 
                        xlabel="Day", ylabel="Portfolio Value ($US)", 
                        title="[Training] Portfolio Value and Largest Coin Investment vs. Day")

        # Plot run results
        if (True):
            plot_portfolio(which_data=running_data_name, values=run_results['Portfolio Value Single'], 
                        values_label="Dev Set", xlabel="Day", ylabel="Portfolio Value ($US)",
                        title=("[%s] Portfolio Value (of Last Iteration) vs. Day" % run_type))
        if (True):
            plot_portfolio(which_data=running_data_name, values=run_results['Portfolio Value All'],
                        values_label="Dev Set", xlabel="Day", ylabel="Portfolio Value ($US)",
                        title=("[%s] Portfolio Value (All Iterations) vs. Day" % run_type))
        if (True):
            plot_only_choice(which_data=running_data_name, daily_choices=run_results['Daily Choices'], 
                        portfolio_values=run_results['Portfolio Value Single'], 
                        xlabel="Day", ylabel="Portfolio Value ($US)", 
                        title=("[%s] Portfolio Value and Largest Coin Investment vs. Day" % run_type))

    ########################################################################################
    # [4] Run multiple trials, plot each result and average results
    ########################################################################################
    if (False):

        # Choose how many trials to run
        num_trials = 5

        # Lists to keep track of results
        rewards = []
        values = []
        choices = []

        # Loop 5 times
        for i in range(num_trials):

            print 'Beginning trial %d...' % i

            # Train
            cryptoMDP = CryptoTrading(training_data, START_VALUE, HIGH, MEDIUM, LOW)
            train_results = trainModel(cryptoMDP, FeatureExtractor, START_VALUE, train_iterations, explore_type, SINGLE_PROBS, MULTI_PROBS, verbose=False)
            QLAlgo = train_results['QLAlgo']
            train_values = train_results['Portfolio Value Single']
            train_choices = train_results['Daily Choices']

            # Run
            cryptoMDP.coinData = running_data
            run_results = runModel(cryptoMDP, QLAlgo, FeatureExtractor, START_VALUE, 1, verbose=False)

            # Save results / print
            rewards.append(run_results['Final Reward']) 
            values.append(run_results['Portfolio Value Single'])
            choices.append(run_results['Daily Choices'])
            print "[%d] Reward: %.2f" % (i + 1, run_results['Final Reward'])

        # Print average final reward
        print "Average reward: %.2f" % (sum(rewards)/len(rewards))

        # Determine daily average portfolio value
        avg_value = []
        num_days = len(values[0])
        for day in range(num_days):
            total = 0
            for trial in range(num_trials):
                total += values[trial][day]
            avg = total / num_trials
            avg_value.append(avg)

        # Plot portfolio values for training
        if (True):
            plot_portfolio_choice(which_data=training_data_name, values=train_values, choices=train_choices,
                        values_label="Train Results (final iteration)",
                        xlabel="Day", ylabel="Portfolio Value ($US)",
                        title="[Training] Portfolio Value and Largest Coin Investment vs. Day") 

        # Plot portfolio values for each run trial (with coin choices)
        if (True):
            plot_portfolio_multi_choice(which_data=running_data_name, values=values, choices=choices, avg=avg_value,
                        values_label=("Results (%d trials)" % num_trials), avg_label="Avg. Result",
                        xlabel="Day", ylabel="Portfolio Value ($US)", 
                        title=("[%s] Portfolio Value and Largest Coin Investment vs. Day" % run_type))                  

    ########################################################################################
    # [5] Trying different portfolio allocation values
    ########################################################################################
    if (False):

        # Allocation options to try
        allocations = [ (0.5, 0.4, 0.1),
                        (0.5, 0.3, 0.2),
                        (0.5, 0.25, 0.25),
                        (0.6, 0.3, 0.1),
                        (0.6, 0.2, 0.2),
                        (0.7, 0.2, 0.1),
                        (0.7, 0.15, 0.15),
                        (0.8, 0.15, 0.05),
                        (0.8, 0.1, 0.1),
                        (0.9, 0.05, 0.05)]

        # Loop through all allocation options
        for allocation in allocations:

            # Isolate each percentage
            high_per = allocation[0]
            medium_per = allocation[1]
            low_per = allocation[2]

            # Run 10 trials for each
            run_rewards = []
            print "\nHIGH: %.2f, MEDIUM: %.2f, LOW: %.2f" % (allocation[0], allocation[1], allocation[2])
            for i in range(20):

                # Train
                cryptoMDP = CryptoTrading(training_data, START_VALUE, high_per, medium_per, low_per)
                train_results = trainModel(cryptoMDP, FeatureExtractor, START_VALUE, train_iterations, 
                    explore_type, SINGLE_PROBS, MULTI_PROBS, verbose=False)
                QLAlgo = train_results['QLAlgo']

                # Run
                cryptoMDP.coinData = running_data
                run_results = runModel(cryptoMDP, QLAlgo, FeatureExtractor, START_VALUE, 1, verbose=False)

                # Save results / print
                run_rewards.append(run_results['Final Reward'])
                print "[%d] Reward: %.2f" % (i + 1, run_results['Final Reward'])

            # Print average final reward
            print "Average reward: %.2f" % (sum(run_rewards)/len(run_rewards))

    ########################################################################################
    # [6] Checking for convergence
    ########################################################################################
    if (False):

        # Setup
        cryptoMDP = CryptoTrading(training_data, START_VALUE, HIGH, MEDIUM, LOW)
        num_iterations = [100, 1000, 5000, 10000, 15000, 20000]
        train_rewards = []
        run_rewards = []
        print 'Beginning training...'

        # Loop through each of the values in num_iterations
        for iterations in num_iterations:

            print '\nTraining for %d iterations...' % iterations

            # Train (divide iterations by 3 because doing decaying exploration probabilities)
            train_results = trainModel(cryptoMDP, FeatureExtractor, START_VALUE, int(iterations / 3.), 
                MULTI_EXPLORE, SINGLE_PROBS, MULTI_PROBS, verbose=False)
            QLAlgo = train_results['QLAlgo']

            # Save final reward and difference between last few rewards
            final_reward = train_results['Final Reward']
            num_rewards = len(train_results['All Rewards'])
            prior_reward1 = train_results['All Rewards'][num_rewards - 1]
            prior_reward2 = train_results['All Rewards'][num_rewards - 2]
            prior_reward3 = train_results['All Rewards'][num_rewards - 3]
            difference1 = final_reward - prior_reward1
            difference2 = prior_reward1 - prior_reward2
            difference3 = prior_reward2 - prior_reward3
            train_rewards.append((final_reward, difference1, difference2, difference3))
            print 'Final Train Reward: %.2f, Last Few Changes: %.2f, %.2f, %.2f' % (final_reward, difference1, difference2, difference3)

            # Run
            cryptoMDP.coinData = running_data
            run_results = runModel(cryptoMDP, QLAlgo, FeatureExtractor, START_VALUE, 1, verbose=False)

            # Save final reward
            final_reward = run_results['Final Reward']
            run_rewards.append(final_reward)
            print 'Final Run Reward: %.2f' % (final_reward)

if __name__ == '__main__':
    main()
