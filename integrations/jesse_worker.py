"""On-demand native Jesse baseline backtest. It does not test LAYA decisions."""
import sys,json,time,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
with contextlib.redirect_stdout(sys.stderr):
 import numpy as np
 from jesse.strategies import Strategy
 from jesse.research import backtest
 import jesse.helpers as jh
 from technical import enrich,STRATEGY_ID
 import pandas as pd
 payload=json.load(sys.stdin)
 rows=payload['rows'];symbol=payload['symbol']
 if len(rows)<2000:raise ValueError('At least 2000 closed 1m candles required')
 # Jesse order: timestamp, open, CLOSE, high, low, volume.
 candles=np.array([[r[0],r[1],r[4],r[2],r[3],r[5]] for r in rows],dtype=float)
 if not np.all(np.diff(candles[:,0])==60000):raise ValueError('Missing/duplicate 1m candles')
 class HybridBaseline(Strategy):
  def signal(self):
   c=self.candles
   d=pd.DataFrame({'open':c[:,1],'close':c[:,2],'high':c[:,3],'low':c[:,4],'volume':c[:,5]})
   return enrich(d).iloc[-1]
  def should_long(self):return len(self.candles)>=250 and bool(self.signal()['enter_long'])
  def should_short(self):return len(self.candles)>=250 and bool(self.signal()['enter_short'])
  def should_cancel_entry(self):return True
  def go_long(self):
   qty=self.balance*.07/self.price
   self.buy=qty,self.price
   self.stop_loss=qty,self.price*.985
   self.take_profit=qty,self.price*1.03
  def go_short(self):
   qty=self.balance*.07/self.price
   self.sell=qty,self.price
   self.stop_loss=qty,self.price*1.015
   self.take_profit=qty,self.price*.97
  def update_position(self):
   if (self.is_long and self.signal()['exit_long']) or (self.is_short and self.signal()['exit_short']):self.liquidate()
 exchange='Binance Perpetual Futures';pair=symbol[:-4]+'-USDT';key=jh.key(exchange,pair)
 config={'starting_balance':2000,'fee':.0007,'type':'futures','futures_leverage':1,'futures_leverage_mode':'isolated','exchange':exchange,'warm_up_candles':250}
 routes=[{'exchange':exchange,'strategy':HybridBaseline,'symbol':pair,'timeframe':'5m'}]
 warm={key:{'exchange':exchange,'symbol':pair,'candles':candles[:1250]}}
 trading={key:{'exchange':exchange,'symbol':pair,'candles':candles[1250:]}}
 result=backtest(config,routes,[],trading,warmup_candles=warm,generate_equity_curve=True,fast_mode=False)
 m=result['metrics'];curve=result.get('equity_curve') or []
 # Drawdown is derived from the actual returned equity curve, avoiding unit ambiguity.
 values=[]
 for point in curve:
  if isinstance(point,dict):value=point.get('value',point.get('balance'))
  else:value=point[1]
  if value is not None:values.append(float(value))
 if not values:values=[2000,2000*(1+float(m.get('net_profit_percentage',0))/100)]
 peak=2000.;dd=0.
 for v in values:peak=max(peak,v);dd=max(dd,(peak-v)/peak*100)
 report={'schema':1,'market':'USDT_PERPETUAL','source':'jesse','kind':'research','symbol':symbol,'strategy_id':STRATEGY_ID,
  'timeframe':'5m','data_start':rows[1250][0]/1000,'data_end':rows[-1][6]/1000,
  'generated_at':time.time(),'metrics':{'return_pct':float(m.get('net_profit_percentage',0)),
  'max_drawdown_pct':dd,'trades':int(m.get('total',0))}}
print(json.dumps(report,allow_nan=False))
