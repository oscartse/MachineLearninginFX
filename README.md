# MachineLearninginFX
This is a Final Year Project concerning about Machine learning in Foreign Exchange, combining GA with technical analysis to forcast the FX market 
Hv Fun! :D

# Main Idea Genetic algorithm
To first generate random models then evaluate based on performance, after that sort them in order and breeding next generation based on the good ones that survived in the previous selection

# Core architecture used
- Long-Short Term Model
  - Common in processing financial data, time-series related
  - Ability to recognize similar event that previously encountered
- Data Preprocessing
  - SMA/EMA/BBands combining with raw data to hv a better training idea
  - Reshaping to LSTM input layer
  
# Testing whether it is profitable or not
Using PyAlgoTrade libuary to backtest data generated combining with stragegies
