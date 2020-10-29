import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates 

sns.set()

import csv 
import os
import sys

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from datetime import datetime
from matplotlib.dates import DateFormatter

from tqdm import tqdm_notebook

from itertools import product

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

import warnings
warnings.filterwarnings('ignore')

subreddit = sys.argv[1]
path = sys.argv[2]

os.mkdir(subreddit)



def get_csv(filename, output):
  

    if os.path.exists(output):
        os.remove(output)
    else:
        print("The file does not exist")  

    fields = [] 
    rows = [] 

    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 

        fields = next(csvreader) 
        for row in csvreader: 
            rows.append(row) 


    columns = ["liwc_anger", "liwc_anxiety", "liwc_death", "liwc_negative_emotion", "liwc_positive_emotion", "liwc_sadness", "liwc_swear_words"]
    indices = []
    for column in columns:
        indices.append(fields.index(column))

    rows = sorted(rows,key=lambda l:l[2])

    date = rows[0][2]
    values = []
    for i in indices:
        values.append(float(rows[0][i]))
    count = 1

    sample = open(output, 'w') 
    
    col = "date,"+(",").join(columns)
    print(col, file = sample)

    for i in range(1,len(rows)):

        if(rows[i][2] == date):
            for x,index in enumerate(indices):
                values[x] += float(rows[i][index])
            count += 1

        else:    
            if(date[5:] != "02/29"):
            
                for x in range(len(values)):
                    values[x] /= count 
                print(date[5:]+","+str(values)[1:-1],file = sample)
            
            date = rows[i][2]
            
            for x,index in enumerate(indices):
                values[x] = float(rows[i][index])
            
            count = 1

    for x in range(len(values)):
        values[x] /= count 
    print(date[5:]+","+str(values)[1:-1],file = sample)
    
    sample.close() 



get_csv(path + "/" + subreddit +"_2019_features_tfidf_256.csv","sample2019.csv")
get_csv(path + "/" + subreddit +"_post_features_tfidf_256.csv","sample2020.csv")



DATAPATH2019 =  'sample2019.csv'
DATAPATH2020 =  'sample2020.csv'



parser = lambda date: pd.datetime.strptime(date, '%m/%d')




data2019 = pd.read_csv(DATAPATH2019, index_col=['date'], parse_dates=['date'], date_parser=parser)
data2020 = pd.read_csv(DATAPATH2020, index_col=['date'], parse_dates=['date'], date_parser=parser)


def plot_moving_average(series, series2, window, metric, plot_intervals=False, scale=1.96):

    rolling_mean = series.rolling(window=window).mean()
    rolling_mean2 = series2.rolling(window=window).mean()
    
    plt.figure(figsize=(17,8))
    plt.title('Moving average\n window size = {}'.format(window))
    
    plt.plot(rolling_mean2,  label='Rolling mean trend for 2019')
    plt.plot(rolling_mean,  label='Rolling mean trend for 2020')
    
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bound = rolling_mean - (mae + scale * deviation)
        upper_bound = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
            
    # plt.plot(series[window:], label='Actual values for 2020')
    plt.legend(loc='best')
    plt.grid(True)
    # plt.show()
    plt.savefig(subreddit + "/" + metric + "/MovingAverage.png" )



def exponential_smoothing(series, alpha):

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result



def plot_exponential_smoothing(series, alphas, metric):
 
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
    plt.plot(series.values, "c", label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Exponential Smoothing")
    plt.grid(True);
    # plt.show()
    plt.savefig(subreddit + "/" + metric + "/ExponentialSmoothing.png" )


def double_exponential_smoothing(series, alpha, beta):

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result




def plot_double_exponential_smoothing(series, metric, alphas, betas):
     
    plt.figure(figsize=(17, 8))
    for alpha in alphas:
        for beta in betas:
            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
    plt.plot(series.values, label = "Actual")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.title("Double Exponential Smoothing")
    plt.grid(True)
    # plt.show()
    plt.savefig(subreddit + "/" + metric + "/DoubleExponentialSmoothing.png" )

def tsplot(y, metric, lags=None, figsize=(12, 7), syle='bmh'):
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()

    # plt.show()
    plt.savefig(subreddit + "/" + metric + "/TSPlot.png" )
        


for metric in data2020.columns:

    os.mkdir(subreddit + "/" + metric)
    figure, axes = plt.subplots(figsize=(25, 8)) 
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d')) 
    
    axes.plot(data2019[metric])
    axes.plot(data2020[metric])  
    
    axes.legend(["2019", "2020"], loc ="upper right") 
    axes.set(xlabel="Date", ylabel=metric,title="2019 "+ metric +" vs 2020 " + metric)
    
    plt.savefig(subreddit + "/" + metric + "/Comparison.png" )
    # plt.show()


    plot_moving_average(data2020[metric], data2019[metric], 14, metric, plot_intervals=True)

    plot_exponential_smoothing(data2020[metric], [0.05, 0.65], metric)
    plot_double_exponential_smoothing(data2020[metric], metric, alphas=[0.6, 0.2], betas=[0.6, 0.2])

    tsplot(data2020[metric], metric, lags=30)
    data_diff = data2020[metric] - data2020[metric].shift(1)
    tsplot(data_diff[1:], metric, lags=30)


os.remove("sample2020.csv")
os.remove("sample2019.csv")