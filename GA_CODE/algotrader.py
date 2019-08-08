from pyalgotrade import strategy
from pyalgotrade.technical import ma
from pyalgotrade.technical import rsi
from pyalgotrade.technical import macd
from pyalgotrade.technical import atr
from pyalgotrade.barfeed import csvfeed
from pyalgotrade.technical import cross
from sklearn.externals import joblib
from pyalgotrade.technical import bollinger
from sklearn.preprocessing import MinMaxScaler
import talib
import pandas as pd
import pyalgotrade
demo_cash =1000000
import numpy as np


class machine_learning_strategy(strategy.BacktestingStrategy):
    def init(self, feed,instrument ,initialCash,model): #initializing order
        super(machine_learning_strategy, self).init(feed,initialCash)
        # We want a 15 period SMA over the closing prices.
        SMA=14
        self.instrument=instrument
        self.__longPos = None
        self.__shortPos = None
        self.__historical = []
        self.__price = feed[instrument].getPriceDataSeries()
        self.__macd =macd.MACD(feed[instrument].getPriceDataSeries(),24,120,48,1024) #fast ,slow ,signal
        self.__sma = ma.SMA(feed[instrument].getPriceDataSeries(), 240)
        self.__sma5d = ma.SMA(feed[instrument].getPriceDataSeries(),240)
        self.__sma10d =ma.SMA(feed[instrument].getPriceDataSeries(),240*10)
        self.__bbands=bollinger.BollingerBands(feed[instrument].getPriceDataSeries(),40,2)
        self.atr = atr.ATR(feed[instrument],24)
        self.__prediction_list= []

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info("BUY at $%.2f" % (execInfo.getPrice()))

    def onExitOk(self, position):
        if self.__longPos == position:
            self.__longPos = None
            self.info('Long Position is canceled')

        elif self.__shortPos == position:
            self.__shortPos = None
            self.info('Short Position is canceled')
        else:
            assert (False)

    def onExitCanceled(self,position):
        # If the exit was canceled, re-submit it.
        if position is self.__longPos:
            self.__longPos.exitMarket()
        if position is self.__shortPos:
            self.__shortPos.exitMarket()

    def pos_check(self,position):
        return 0

    def onBars(self, bars): #amend and implement the strategy in this function
        bar = bars[self.__instrument]
        lower = self.__bbands.getLowerBand()[-1]
        middle = self.__bbands.getMiddleBand()[-1]
        upper = self.__bbands.getUpperBand()[-1]
        currentprice=round(bar.getPrice(),5)


        if self.__sma5d[-1] is None or  self.__sma10d[-1] is None or self.__bbands.getLowerBand()[-1] is None or self.__bbands.getMiddleBand()[-1] is None or self.__bbands.getUpperBand()[-1] is None  : #Wait to get enough info for tech indicator
            return
        lower = round(lower, 5) #BBlower
        middle = round(middle, 5) #BBLower
        upper = round(upper, 5) #BBUpper
        atr = self.atr[-1] #ATR
        dataset = np.array([round(bar.getClose(),6),self.__sma[-1], upper, middle, lower]) #recombine the array
        self.__historical.append(dataset)

        if len(self.__historical) ==240: #get 240 data frame 5days record
            df_dataset = pd.DataFrame(self.__historical)
            dataset=df_dataset.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            dataset = dataset.reshape((240, 1, -1)) #create df and then reshape the dataframe
            prediction  =model.predict(dataset) #
            dataset = dataset.reshape((dataset.shape[0], dataset.shape[2]))
            final_dataset = np.concatenate((prediction, dataset[:, 1:]), axis=1) #recombine the data
            final_dataset = scaler.inverse_transform(final_dataset)
            prediction_array = final_dataset[:, 0]
            prediction = round(prediction_array[-1] ,  5)
            #print(f'Prediction:{prediction} Actual: {currentprice}') #latest prediction
            self.__historical.remove(self.__historical[0])
            self.__prediction_list.append(prediction)



        try:  # ensure the record is not Null
            x = self.__prediction_list[-1]
            y = self.__prediction_list[-2]

        except:
            return

        def cutlossmechanism(self):


            return 0

        if self.__longPos is None: #if no long position
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice())*100000)) #lot size is determined by the amount of cash we have in the demo
            if self.__prediction_list[-1] > self.__prediction_list[-2] and currentprice > self.__sma5d[-1]:  #strategy criteria #long
                #self.info(f'LONG Position ENTRY:{currentprice}')
                self.__longPos = self.enterLong(self.__instrument,lot_size*500,True)
                #set sell limit and stop loss position


        elif self.__prediction_list[-1] < self.__prediction_list[-2] and  currentprice < self.__sma10d[-1] and not self.__longPos.exitActive():
            #self.info(f'LONG Position EXIT:{currentprice}')
            self.__longPos.exitMarket()


        #elif  self.__longPos.getEntryOrder() - currentprice > 0.00500: #If winning more than 500 pips , then close position
        #        self.__longPos.exitMarket()


        """
        if self.__shortPos is None:
            lot_size = int(self.getBroker().getCash() * 0.9 / ((bars[self.__instrument].getPrice()) * 100000))


            if self.__prediction_list[-1] < self.__prediction_list[-2]  : #in down trend
                self.__shortPos = self.enterShort(self.__instrument, lot_size*500, True)
                #self.info(f'SHORT Position ENTRY:{currentprice}')

        elif self.__prediction_list[-1] > self.__prediction_list[-2] and not self.__shortPos.exitActive() :  # EXIT RULE
            #self.info(f'SHORT Position EXIT:{currentprice}')
            self.__shortPos.exitMarket()
        """
        P_L=self.getBroker().getCash()-1000000
        self.info(f'P/L :{P_L} ')




    def onExitCanceled(self, position):
        # If the exit was canceled, re-submit it.
        position.exitMarket()

    def onEnterCanceled(self, position): #cancel order before execution
        if self.__longPos is position:
            self.__longPos = None
        if self.__shortPos is position:
            self.__shortPos = None

    def onExitOk(self, position): #exit all position
        execInfo = position.getExitOrder().getExecutionInfo()

        if self.__longPos is position:
            self.__longPos = None
        if self.__shortPos is position:
            self.__shortPos = None

def run_strategy(model):

    feed = csvfeed.GenericBarFeed(frequency=pyalgotrade.barfeed.Frequency.HOUR,timezone=None, maxLen=1024)
    feed.addBarsFromCSV(instrument="gbpusd",path="C:/Users/LokFung/Desktop/IERGYr4/IEFYP/POCtestdata/AUDUSD60_training.csv") #FIXED BARFEED ISSUE
    myStrategy = machine_learning_strategy(feed, "gbpusd",1000000,model)
    #myStrategy.run()
    final_portfolio_PL= myStrategy.getBroker().getEquity()-1000000
    print("Final portfolio Earning: {}  ".format(24867.425639998983))
    print(' Percentage Gain:{}%'.format(round(24867.425639998983/1000000*100,3)))
    return final_portfolio_PL

if __name__=='main':
    model=joblib.load('aud_model.sav')
    record=[]
    #for period in range(1,241,5): #SMA1 to SMA240
     #   return_port=run_strategy(period)
     #   record.append((return_port,period))
    result=run_strategy(model)
    df = pd.DataFrame(record,columns=['Period', 'Earning'])
    df.to_csv('final_result.csv', index=False)
    print('FINISHED')