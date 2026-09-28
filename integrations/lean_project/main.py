"""LEAN-native futures baseline validation, not LAYA backtesting or live trading."""
from AlgorithmImports import *
from collections import deque
from datetime import datetime,timezone,timedelta
import json,time
import pandas as pd
from technical import enrich,STRATEGY_ID

class PercentFee(FeeModel):
 def get_order_fee(self,parameters):
  value=abs(float(parameters.order.quantity))*float(parameters.security.price)*.0005
  return OrderFee(CashAmount(value,'USDT'))

class LayaBaselineLab(QCAlgorithm):
 def initialize(self):
  if self.live_mode:raise RuntimeError('This integration is research-only')
  start=datetime.fromisoformat(self.get_parameter('start') or '2026-09-01')
  end=datetime.fromisoformat(self.get_parameter('end') or '2026-09-27')
  self.set_start_date(start.year,start.month,start.day);self.set_end_date(end.year,end.month,end.day)
  self.set_time_zone(TimeZones.UTC);self.set_account_currency('USDT');self.set_cash(2000)
  self.ticker=self.get_parameter('symbol') or 'BTCUSDT'
  security=self.add_crypto_future(self.ticker,Resolution.MINUTE,Market.BINANCE,leverage=1)
  security.set_fee_model(PercentFee());security.set_slippage_model(ConstantSlippageModel(.0002))
  self.symbol=security.symbol;self.consolidate(self.symbol,timedelta(minutes=5),self.on_five_minute);self.rows=deque(maxlen=300);self.pending=None
  self.entry=0.;self.peak=2000.;self.drawdown=0.;self.closed=0;self.start_seen=None;self.end_seen=None
 def on_data(self,data):
  pass
 def on_five_minute(self,bar):
  now=bar.end_time.replace(tzinfo=timezone.utc).timestamp()
  if self.start_seen is None:self.start_seen=now
  self.end_seen=now
  # Act on the prior completed bar's signal, avoiding same-bar signal/fill assumptions.
  if self.pending in ('LONG','SHORT') and not self.portfolio[self.symbol].invested:
   self.market_order(self.symbol,(1 if self.pending=='LONG' else -1)*float(self.portfolio.total_portfolio_value)*.07/float(bar.close))
  elif self.pending=='EXIT' and self.portfolio[self.symbol].invested:
   self.liquidate(self.symbol)
  self.pending=None
  self.rows.append({'open':float(bar.open),'high':float(bar.high),'low':float(bar.low),'close':float(bar.close),'volume':float(bar.volume)})
  if len(self.rows)<250:return
  row=enrich(pd.DataFrame(self.rows)).iloc[-1]
  holding=self.portfolio[self.symbol]
  if holding.invested:
   e=float(holding.average_price);p=float(bar.close)
   d=1 if holding.is_long else -1
   if (row.exit_long if d==1 else row.exit_short) or d*(p/e-1)<=-.015 or d*(p/e-1)>=.03:self.pending='EXIT'
  elif row.enter_long:self.pending='LONG'
  elif row.enter_short:self.pending='SHORT'
  equity=float(self.portfolio.total_portfolio_value)
  self.peak=max(self.peak,equity);self.drawdown=max(self.drawdown,(self.peak-equity)/self.peak*100)
 def on_order_event(self,event):
  if event.status==OrderStatus.FILLED and not self.portfolio[self.symbol].invested:self.closed+=1
 def on_end_of_algorithm(self):
  if self.start_seen is None:return
  report={'schema':1,'market':'USDT_PERPETUAL','source':'lean','kind':'research','symbol':self.ticker,'strategy_id':STRATEGY_ID,
   'timeframe':'5m','data_start':self.start_seen,'data_end':self.end_seen,'generated_at':time.time(),
   'metrics':{'return_pct':(float(self.portfolio.total_portfolio_value)/2000-1)*100,
              'max_drawdown_pct':self.drawdown,'trades':self.closed}}
  key='laya_lean_report.json';self.object_store.save(key,json.dumps(report,allow_nan=False))
  self.debug('Research report saved to Object Store: '+key)
