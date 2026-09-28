from freqtrade.strategy import IStrategy
from technical import enrich

class HybridTechnicalV2(IStrategy):
 INTERFACE_VERSION=3
 timeframe='5m'
 can_short=True
 startup_candle_count=250
 stoploss=-0.015
 minimal_roi={'0':0.03}
 process_only_new_candles=True
 def populate_indicators(self,dataframe,metadata):return enrich(dataframe)
 def populate_entry_trend(self,dataframe,metadata):return dataframe
 def populate_exit_trend(self,dataframe,metadata):return dataframe
