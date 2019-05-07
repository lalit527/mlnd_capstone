# Machine Learning Engineer Nanodegree
## Capstone Proposal
Lalit Yadav
May 5th, 2019

## Proposal

### Domain Background

Time Series is one of the most challenging probelms in the field of Machine Learning.In this project, we will use machine learning algorithm to forecast the future web traffic for approximately 145,000 Wikipedia articles.
The field of time series encapsulates many different problems, ranging from analysis and inference to classification and forecast. Predicting this type of future traffic for an web page will 



### Problem Statement

Sequential or temporal observations emerge in many key real-world problems, ranging from biological data, financial markets, weather forecasting, to audio and video processing. The field of time series encapsulates many different problems, ranging from analysis and inference to classification and forecast.

This challenge is about predicting the future behaviour of time series’ that describe the web traffic for Wikipedia articles. The data contains about 145k time series and comes in two separate files: train_1.csv holds the traffic data, where each column is a date and each row is an article, and key_1.csv contains a mapping between page names and a unique ID column (to be used in the submission file).

### Datasets and Inputs
The datasets are provided by Google on Kaggle competition website.

The training dataset consists of approximately 145k time series. Each of these time series represent a number of daily views of a different Wikipedia article, starting from July, 1st, 2015 up until December 31st, 2016. The leaderboard during the training stage is based on traffic from January, 1st, 2017 up until March 1st, 2017. For each time series, you are provided the name of the article as well as the type of traffic that this time series represent (all, mobile, desktop, spider). You may use this metadata and any other publicly available data to make predictions.

### Solution Statement
A deep learning algorithm with RNN GRU/ LSTM architecture.
Recurrent Neural Networks — Long short-term memory (LSTM), Gated Recurrent Unit (GRU). An LSTM model architecture for time series forecasting comprised of separate autoencoder and forecasting sub-models. The skill of the proposed LSTM architecture at rare event demand forecasting and the ability to reuse the trained model on unrelated forecasting problems.

### Benchmark Model

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
The Evaluation Metrics used for this Kaggle competition is SMAPE(Symmetric mean absolute percentage error (SMAPE or sMAPE))
SAMPE is an accuracy based on percentage (or relative) errors. It is defined as:-

       {\displaystyle {\text{SMAPE}}={\frac {1}{n}}\sum _{t=1}^{n}{\frac {\left|F_{t}-A_{t}\right|}{(A_{t}+F_{t})/2}}}
       
where At is the actual value and Ft is the forecast value.

The absolute difference between At and Ft is divided by half the sum of absolute values of the actual value At and the forecast value Ft. The value of this calculation is summed for every fitted point t and divided again by the number of fitted points n.

### Project Design

1. We will first import the data required for the project.
2. We will have to perform the required data cleaning. We also have to add some features that we will be using got the solution. Days, Months, Years are interesting to forecast with a Machine Learning Approach or to do an analysis.
3. 


-----------

**Before submitting your proposal, ask yourself. . .**

- Kaggle
  - https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion
- Kernels
  - https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795#latest-525730
  - https://www.kaggle.com/headsortails/wiki-traffic-forecast-exploration-wtf-eda
- SMAPE

