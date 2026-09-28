"""Every active USDT perpetual futures pair is analyzed; bounded AI queue is fair, not top-N only."""
import threading,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from technical import frame,enrich,summarize

class Scanner:
 def __init__(self,feed,workers=8,min_volume=1000000):
  self.feed=feed;self.workers=workers;self.min_volume=min_volume;self.lock=threading.RLock()
  self.rows={};self.features_cache={};self.stop_event=threading.Event();self.thread=None
  self.pool=ThreadPoolExecutor(workers);self.cycle=0;self.duration=0.;self.error='';self.total=0;self.done=0
 def analyze(self,s,tickers,books):
  data=self.feed.candles(s);key=data[-1][0];old=self.features_cache.get(s)
  if old and old[0]==key:tech=old[1]
  else:tech=summarize(enrich(frame(data)));self.features_cache[s]=(key,tech)
  q=books.get(s);volume=float(tickers.get(s,{}).get('quoteVolume',0))
  spread=(q['ask']-q['bid'])/q['mid']*10000 if q else None
  eligible=volume>=self.min_volume and spread is not None and spread<=15
  return {'symbol':s,'at':time.time(),'candle_at':data[-1][6]/1000,'signal':tech['signal'],
   'rsi':tech['rsi'],'volume24h':volume,'spread_bps':spread,'eligible':eligible,
   'reason':'AI queue' if eligible else 'Low liquidity / wide or absent spread','error':''}
 def run_once(self):
  started=time.monotonic();symbols=self.feed.universe();tickers=self.feed.tickers();books=self.feed.books()
  with self.lock:
   self.total=len(symbols);self.done=0;self.error=''
   self.rows={s:v for s,v in self.rows.items() if s in symbols}
  futures={self.pool.submit(self.analyze,s,tickers,books):s for s in symbols}
  for future in as_completed(futures):
   if self.stop_event.is_set():
    for f in futures:f.cancel()
    break
   s=futures[future]
   try:r=future.result()
   except Exception as exc:r={'symbol':s,'at':time.time(),'signal':'ERROR','eligible':False,'error':str(exc)}
   with self.lock:self.rows[s]=r;self.done+=1
  with self.lock:self.cycle+=1;self.duration=round(time.monotonic()-started,2)
 def loop(self):
  while not self.stop_event.is_set():
   started=time.monotonic()
   try:self.run_once()
   except Exception as exc:
    with self.lock:self.error=str(exc)
   self.stop_event.wait(max(1,60-(time.monotonic()-started)))
 def start(self):
  if self.thread and self.thread.is_alive():return
  self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
 def candidates(self,reviewed,limit,positions):
  with self.lock:rows=[dict(r) for r in self.rows.values() if r['eligible'] and time.time()-r['at']<150]
  # Unreviewed/oldest first ensures eventual AI coverage; liquidity breaks ties.
  rows.sort(key=lambda r:(reviewed.get(r['symbol'],0),-r.get('volume24h',0)))
  return list(dict.fromkeys(list(positions)+[r['symbol'] for r in rows[:limit]]))
 def snapshot(self):
  with self.lock:
   return {'total':self.total,'done':self.done,'cycles':self.cycle,'cycle_seconds':self.duration,'error':self.error},list(self.rows.values())
 def stop(self):self.stop_event.set();self.pool.shutdown(wait=False,cancel_futures=True)
