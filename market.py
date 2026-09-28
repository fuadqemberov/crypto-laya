"""Pooled read-only market API, bounded rate usage, memory + SQLite candle cache."""
import copy,json,sqlite3,threading,time
from collections import deque
from pathlib import Path
import httpx

class TTLCache:
 def __init__(self):
  self.data={};self.locks={};self.guard=threading.Lock();self.hits=0;self.misses=0
 def get(self,key,ttl,loader,version=None):
  with self.guard:lock=self.locks.setdefault(key,threading.Lock())
  with lock:
   now=time.monotonic()
   if key in self.data and now-self.data[key][0]<ttl and self.data[key][2]==version:
    with self.guard:self.hits+=1
    return self.data[key][1]
   value=loader()
   self.data[key]=(time.monotonic(),value,version)
   with self.guard:self.misses+=1
   return value
 def stats(self):
  with self.guard:return {'hits':self.hits,'misses':self.misses,'entries':len(self.data)}

class WeightLimiter:
 def __init__(self,budget=1200,window=60):
  self.budget=budget;self.window=window;self.events=deque();self.lock=threading.Lock();self.blocked_until=0.
 def acquire(self,weight):
  if weight>self.budget:raise ValueError('Request exceeds configured weight budget')
  while True:
   with self.lock:
    now=time.monotonic()
    while self.events and self.events[0][0]<=now-self.window:self.events.popleft()
    delay=max(0,self.blocked_until-now)
    if not delay and sum(w for _,w in self.events)+weight<=self.budget:
     self.events.append((now,weight));return
    if not delay:delay=max(.05,self.events[0][0]+self.window-now)
   time.sleep(min(delay,1))
 def block(self,seconds):
  with self.lock:self.blocked_until=max(self.blocked_until,time.monotonic()+seconds)

class MarketFeed:
 def __init__(self,directory,workers=8,budget=1200):
  self.base='https://fapi.binance.com/fapi/v1/'
  self.client=httpx.Client(timeout=10,limits=httpx.Limits(max_connections=workers+4,max_keepalive_connections=workers+4))
  self.cache=TTLCache();self.limiter=WeightLimiter(budget);self.funding_limiter=WeightLimiter(450,300);self.db_lock=threading.Lock();self.requests=0
  Path(directory).mkdir(parents=True,exist_ok=True)
  self.db=sqlite3.connect(str(Path(directory)/'candles.sqlite'),check_same_thread=False)
  self.db.execute('CREATE TABLE IF NOT EXISTS candles (symbol TEXT PRIMARY KEY,payload TEXT NOT NULL)');self.db.commit()
  self.rows={}
 def get(self,path,**params):
  weights={'exchangeInfo':1,'ticker/24hr':40,'ticker/bookTicker':5 if not params.get('symbol') else 2,'depth':2,'klines':2,'premiumIndex':10,'fundingRate':1}
  if path=='fundingRate':self.funding_limiter.acquire(1)
  if path=='klines':weights[path]=1 if params.get('limit',500)<100 else (2 if params.get('limit',500)<500 else 5)
  self.limiter.acquire(weights.get(path,2))
  with self.db_lock:self.requests+=1
  r=self.client.get(self.base+path,params=params)
  if r.status_code in (418,429):
   try:seconds=max(5,float(r.headers.get('Retry-After','60')))
   except ValueError:seconds=60
   self.limiter.block(seconds);raise RuntimeError(f'Exchange rate limit {r.status_code}; cooling down {seconds}s')
  r.raise_for_status();value=r.json()
  if isinstance(value,dict) and 'code' in value:raise RuntimeError(str(value))
  return value
 def universe(self):
  def load():
   d=self.get('exchangeInfo')
   return sorted(x['symbol'] for x in d['symbols'] if x['status']=='TRADING' and x['quoteAsset']=='USDT' and x.get('contractType')=='PERPETUAL')
  return self.cache.get('universe',3600,load)
 def tickers(self):
  return self.cache.get('tickers',30,lambda:{x['symbol']:x for x in self.get('ticker/24hr')})
 def books(self):
  def load():
   start=time.monotonic();data=self.get('ticker/bookTicker')
   if time.monotonic()-start>5:raise ValueError('Slow bulk quote response')
   at=time.time();out={}
   for x in data:
    b,a=float(x['bidPrice']),float(x['askPrice'])
    if b>0 and a>=b:out[x['symbol']]={'bid':b,'ask':a,'mid':(a+b)/2,'at':at}
   return out
  return self.cache.get('books',2,load)
 def quote(self,symbol):
  q=self.books().get(symbol)
  if not q or not 0<=time.time()-q['at']<=5:raise ValueError('Missing/stale quote '+symbol)
  mark=self.marks().get(symbol)
  if not mark or not 0<=time.time()-float(mark['time'])/1000<=10:raise ValueError('Stale mark price '+symbol)
  return {**q,'mark':float(mark['markPrice']),'funding_rate':float(mark['lastFundingRate']),'next_funding_time':int(mark['nextFundingTime'])}
 def candles(self,symbol,interval='5m'):
  seconds={'1m':60,'5m':300,'15m':900,'1h':3600}[interval]
  minute=int(time.time()//seconds)
  cache_symbol=symbol+':'+interval
  # Cache by symbol, with expiry at the next boundary; single-flight per symbol.
  def load():
   old=self.rows.get(cache_symbol)
   if old is None:
    with self.db_lock:r=self.db.execute('SELECT payload FROM candles WHERE symbol=?',(cache_symbol,)).fetchone()
    old=json.loads(r[0]) if r else []
   now=time.time()*1000
   if old and old[-1][6]>=minute*seconds*1000-1 and old[-1][6]<now:
    self.rows[cache_symbol]=old;return old
   gap=now-old[-1][6] if old else float('inf')
   new=self.get('klines',symbol=symbol,interval=interval,limit=3 if gap<seconds*2000 else 300)
   merged={r[0]:r for r in old}
   merged.update({r[0]:r for r in new if r[6]<now})
   rows=[merged[k] for k in sorted(merged)][-300:]
   if len(rows)<250 or now-rows[-1][6]>seconds*2000:raise ValueError('Insufficient/stale history '+symbol)
   if any(b[0]-a[0]!=seconds*1000 for a,b in zip(rows,rows[1:])):raise ValueError('Gapped candle history '+symbol)
   self.rows[cache_symbol]=rows
   with self.db_lock:
    self.db.execute('INSERT OR REPLACE INTO candles VALUES (?,?)',(cache_symbol,json.dumps(rows)));self.db.commit()
   return rows
  ttl=max(.25,(minute+1)*seconds+1-time.time())
  rows=self.cache.get('candles:'+cache_symbol,seconds,load,version=minute)
  if time.time()*1000-rows[-1][6]>seconds*2000:raise ValueError('Cached history too old')
  return rows
 def marks(self):
  return self.cache.get('marks',2,lambda:{x['symbol']:x for x in self.get('premiumIndex')})
 def funding(self,symbol,start_ms):
  def load():
   end=int(time.time()*1000);cursor=int(start_ms)+1;result=[]
   while cursor<=end:
    chunk=self.get('fundingRate',symbol=symbol,startTime=cursor,endTime=end,limit=1000)
    result.extend(chunk)
    if len(chunk)<1000:break
    cursor=int(chunk[-1]['fundingTime'])+1
   return result
  return self.cache.get('funding:'+symbol+':'+str(start_ms),30,load)
 def stats(self):return {**self.cache.stats(),'http_requests':self.requests,'weight_budget_per_minute':self.limiter.budget}
 def close(self):self.client.close();self.db.close()
