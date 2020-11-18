## Contents
This directory contains 2 sub-directories, GraphsAverage and GraphsSum, which in turn contain graphs analysing the data from 9 subreddits.

The graphs have been saved in an organised structure to ensure efficiency of search. GraphsAverage and GraphsSum have a common directory structure

```
Directory
└── Subreddit
    └── Quantity
        ├──Comparison.png
        ├──MovingAverage.png
        ├──ExponentialSmoothing.png
        ├──DoubleExponentialSmoothing.png
        └──TSPlot.png
```

Here the Dirctories refer to GraphAverage and GraphSum. The difference between these two subreddits is that the values plotted are the average and sum of the quantity measured in all the posts on that day respectively in that particular subreddit.

Subreddits refer to any of the 9 subreddits which have been analysed, namely, r/addiction, r/alcoholism, r/anxiety, r/conspiracy, r/depression, r/healthanxiety, r/mentalhealth, r/socialanxiety and r/suicidewatch.

The Quantities that have been measured are count(of the posts), anger, anxiety, death, negative emotion, positive emotion, sadness and swear words. Each of these quantities have been measured for all of the subreddits.

5 graphs have been plotted for each of the quantities:
- Comparison.png : Comparison of the values measured during Jan-Apr 2019 and 2020
- MovingAverage.png : Moving Average of the values over a 14-day period
- ExponentialSmoothing.png : Similar to Moving Average but here no explicit time frame is mentioned and dates further away the actual date are given less weightage.
- DoubleExponentialSmoothing.png : Recursive use of Exponential Smoothing
- TSPlot.png : Dickey-Fuller Test Results

## Scripts
- GraphGenrator.py is a python script which generates all the graphs corresponding to a particular subreddit for every quantity except Count. The syntax is:
```bash
    python3 GraphGenerator.py subreddit path_to_dataset
```

- CountGenerator.py is a python script which generates all the graphs corresponding to a particular subreddit for the Count quantity.  The syntax is:
```bash
    python3 CountGenerator.py subreddit path_to_dataset
```

- script.sh is a shell script which runs the GraphGenrator.py and CountGenerator.py.  The syntax is:
```bash
    ./script.sh
```