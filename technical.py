"""Causal baseline shared by native frameworks. EMA200 is 200 five-minute bars."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

STRATEGY_ID='HybridTechnicalV2-'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

def frame(rows):
 d=pd.DataFrame(rows)
 out=pd.DataFrame({'date':pd.to_datetime(d[0],unit='ms',utc=True)})
 for i,k in [(1,'open'),(2,'high'),(3,'low'),(4,'close'),(5,'volume')]:out[k]=d[i].astype(float)
 return out

def enrich(df):
 d=df.copy();c=d['close'];delta=c.diff()
 gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 d['rsi']=100-100/(1+gain/loss.replace(0,np.nan))
 d.loc[(loss==0)&(gain>0),'rsi']=100;d.loc[(loss==0)&(gain==0),'rsi']=50
 for n in [9,21,200]:d[f'ema{n}']=c.ewm(span=n,adjust=False,min_periods=n).mean()
 tr=pd.concat([d.high-d.low,(d.high-c.shift()).abs(),(d.low-c.shift()).abs()],axis=1).max(axis=1)
 d['atr']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
 mid=c.rolling(20).mean();sd=c.rolling(20).std(ddof=0)
 d['bb_lower']=mid-2*sd;d['bb_upper']=mid+2*sd
 d['macd']=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
 d['macd_hist']=d.macd-d.macd.ewm(span=9,adjust=False).mean()
 valid=(d.volume>0)&d.ema200.notna()
 d['enter_long']=(valid&(c>d.ema200)&(d.ema9>d.ema21)&d.rsi.between(45,65)&(d.macd_hist>0)).astype(int)
 d['enter_short']=(valid&(c<d.ema200)&(d.ema9<d.ema21)&d.rsi.between(35,55)&(d.macd_hist<0)).astype(int)
 d['exit_long']=((d.ema9<d.ema21)|(d.rsi>75)).astype(int)
 d['exit_short']=((d.ema9>d.ema21)|(d.rsi<25)).astype(int)
 return d

def summarize(d):
 r=d.iloc[-1]
 values={k:round(float(r[k]),6) for k in ['rsi','ema9','ema21','ema200','atr','macd_hist','bb_lower','bb_upper']}
 if not all(np.isfinite(v) for v in values.values()):raise ValueError('Warmup incomplete')
 return {**values,'signal':'LONG' if r.enter_long else ('SHORT' if r.enter_short else 'WAIT'),
         'exit_long':bool(r.exit_long),'exit_short':bool(r.exit_short),'strategy_id':STRATEGY_ID}
