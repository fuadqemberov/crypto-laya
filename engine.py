"""Single-owner paper USDT perpetual account. Exchange credentials/orders are absent."""
import copy,json,math,os,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
from risk import FEE,SLIP,MAINTENANCE,plan,liquidation

def stamp():return datetime.now(timezone.utc).isoformat(timespec='seconds')

class Engine:
 def __init__(self,path,brain=None,feed=None,hub=None,scanner=None,batch_size=4,max_leverage=20):
  self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
  self.brain=brain;self.feed=feed;self.hub=hub;self.scanner=scanner;self.batch_size=batch_size;self.max_leverage=max_leverage
  self.lock=threading.RLock();self.shutdown=threading.Event();self.active=False;self.generation=0;self.threads=[]
  self.quotes={};self.signals={};self.errors={};self.reviewed={};self.last_decision='—'
  self.pool=ThreadPoolExecutor(min(batch_size,4));self.state={'schema':2,'balance':2000.,'fees':0.,'funding':0.,
   'positions':{},'trades':[],'curve':[],'allocation':7.,'last_entry_bar':{}}
  if self.path.exists():
   loaded=json.loads(self.path.read_text())
   if loaded.get('schema')!=2 or not math.isfinite(loaded['balance']):raise ValueError('Not a futures-v2 account. Use a new data directory; legacy spot state is not imported.')
   self.state=loaded
  self.save()
 def save(self):
  temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.state,allow_nan=False),encoding='utf8');os.replace(temp,self.path)
 @staticmethod
 def fresh(q):
  if not q or not 0<=time.time()-q['at']<=8:raise ValueError('Stale quote: execution blocked')
  if any(not math.isfinite(q[k]) or q[k]<=0 for k in ['bid','ask','mark']) or q['ask']<q['bid']:raise ValueError('Invalid quote')
 @staticmethod
 def pnl(p,price):return p['side']*p['qty']*(price-p['entry'])
 def equity(self):
  return self.state['balance']+sum(self.pnl(p,self.quotes.get(s,{}).get('mark',p['entry'])) for s,p in self.state['positions'].items())
 def free(self):return self.equity()-sum(p['margin'] for p in self.state['positions'].values())
 def open(self,s,side,q,atr,proposed,candle):
  self.fresh(q)
  if s in self.state['positions']:return 'Existing position → WAIT'
  if self.state['last_entry_bar'].get(s)==candle:return 'Already entered this 5m candle → WAIT'
  if (q['ask']-q['bid'])/q['mid']>.0015:return 'Spread >15bps → WAIT'
  equity=self.equity()
  entry=(q['ask'] if side==1 else q['bid'])*(1+side*SLIP)
  risk=plan(equity,self.free(),entry,atr,proposed,self.max_leverage,self.state['allocation']/100)
  existing=sum(max(0,-self.pnl(p,p['sl']))+p['qty']*p['entry']*2*(FEE+SLIP) for p in self.state['positions'].values())
  if existing+risk['risk_usd']>equity*.05:return 'Portfolio stop-risk budget 5% → WAIT'
  qty=risk['notional']/entry;fee=risk['notional']*FEE;distance=entry*risk['stop_fraction']
  self.state['balance']-=fee;self.state['fees']+=fee
  self.state['positions'][s]={'id':uuid.uuid4().hex,'side':side,'entry':entry,'qty':qty,'margin':risk['margin'],
   'leverage':risk['leverage'],'proposed_leverage':proposed,'risk_reason':risk['reason'],
   'sl':entry-side*distance,'tp':entry+side*distance*2,'liquidation_est':liquidation(entry,side,risk['leverage']),
   'entry_fee':fee,'funding':0.,'opened':stamp(),'opened_ms':int(time.time()*1000),'funding_cursor':int(time.time()*1000)}
  self.state['last_entry_bar'][s]=candle
  return f'OPEN {"LONG" if side==1 else "SHORT"} {risk["leverage"]}x • '+risk['reason']
 def apply_funding(self,s,events):
  p=self.state['positions'].get(s)
  if not p:return
  for event in sorted(events,key=lambda x:int(x['fundingTime'])):
   ts=int(event['fundingTime'])
   if not p['funding_cursor']<ts<=time.time()*1000:continue
   mark=float(event['markPrice']);rate=float(event['fundingRate'])
   if not math.isfinite(rate) or not math.isfinite(mark) or mark<=0:raise ValueError('Invalid funding event')
   cash=-p['side']*p['qty']*mark*rate
   p['funding']+=cash;self.state['funding']+=cash;self.state['balance']+=cash;p['funding_cursor']=ts
 def close(self,s,q,reason):
  self.fresh(q);p=self.state['positions'].pop(s)
  exit=(q['bid'] if p['side']==1 else q['ask'])*(1-p['side']*SLIP)
  gross=self.pnl(p,exit);fee=p['qty']*exit*FEE
  self.state['balance']+=gross-fee;self.state['fees']+=fee
  self.state['trades'].append({'symbol':s,'direction':'LONG' if p['side']==1 else 'SHORT','leverage':p['leverage'],
   'opened':p['opened'],'closed':stamp(),'entry':p['entry'],'exit':exit,'qty':p['qty'],'gross':gross,
   'fees':p['entry_fee']+fee,'funding':p['funding'],'net':gross-fee-p['entry_fee']+p['funding'],'reason':reason,
   'side':p['side'],'funding_cursor':p['funding_cursor'],'closed_ms':int(time.time()*1000),'funding_pending':True})
  return reason
 def breakeven(self,s,q):
  p=self.state['positions'].get(s)
  if not p:return 'Flat → WAIT'
  d=p['side'];e=p['entry'];funding_per_coin=p['funding']/p['qty']
  exit_price=(e*(d+FEE)-funding_per_coin)/(d-FEE)
  trigger=exit_price/(1-d*SLIP);market=q['bid'] if d==1 else q['ask']
  if d*(market-trigger)<=0:return 'Insufficient profit for cost-adjusted breakeven'
  if d*(trigger-p['sl'])>0:p['sl']=trigger
  return 'BREAKEVEN set • later funding/gaps may still cause loss'
 def risk(self,s,q):
  self.fresh(q);self.quotes[s]=q;p=self.state['positions'].get(s)
  if not p:return
  mark=q['mark'];d=p['side']
  # Approximate isolated maintenance check. Gap losses are booked, never hidden/capped.
  if p['margin']+self.pnl(p,mark)<=p['qty']*mark*MAINTENANCE:self.close(s,q,'SIMULATED LIQUIDATION')
  elif d*(mark-p['sl'])<=0:self.close(s,q,'STOP')
  elif d*(mark-p['tp'])>=0:self.close(s,q,'TAKE PROFIT')
 def prepare(self,s):
  rows=self.feed.candles(s);expert=self.hub.collect(s,rows) if self.hub else {}
  atr=float(expert.get('atr',0))
  if atr<=0:
   from technical import enrich,frame
   atr=float(enrich(frame(rows)).iloc[-1].atr)
  trends={}
  for interval in ['15m','1h']:
   r=self.feed.candles(s,interval)
   closes=[float(x[4]) for x in r]
   avg=sum(closes[-50:])/50
   trends[interval]={'last':closes[-1],'sma50':round(avg,6),'direction':'UP' if closes[-1]>avg else 'DOWN'}
  q=self.feed.quote(s)
  with self.lock:
   self.risk(s,q);p=self.state['positions'].get(s);identity=p['id'] if p else None
   position={k:p[k] for k in ['side','entry','leverage','sl','tp','margin']} if p else 'flat'
   state={'symbol':s,'market':'USDT perpetual futures','timeframe':'5m','mark':q['mark'],
    'funding_rate':q.get('funding_rate'),'higher_timeframes':trends,'position':position,
    'experts':expert,'risk_limits':{'max_leverage':self.max_leverage,'per_trade_equity_risk_pct':.75},
    'ohlcv_last4':[[float(x[i]) for i in [1,2,3,4,5]] for x in rows[-4:]]}
   self.save()
  return {'symbol':s,'state':state,'identity':identity,'atr':atr,'candle':rows[-1][0],'prepared_at':time.time()}
 def apply_decision(self,item,result,generation):
  s=item['symbol']
  if time.time()-item['prepared_at']>90:raise ValueError('Analysis >90s old; discarded')
  with self.lock:p=self.state['positions'].get(s);cursor=p['funding_cursor'] if p else None
  events=self.feed.funding(s,cursor) if cursor is not None else []
  q=self.feed.quote(s)
  with self.lock:
   if not self.active or generation!=self.generation:return
   self.apply_funding(s,events);self.risk(s,q);p=self.state['positions'].get(s)
   if (p['id'] if p else None)!=item['identity']:self.save();return
   action=result['action'];proposed=result['proposed_leverage']
   if action in ('LONG','SHORT'):
    try:status=self.open(s,1 if action=='LONG' else -1,q,item['atr'],proposed,item['candle'])
    except ValueError as exc:status='RISK WAIT: '+str(exc)
   elif action=='CLOSE':status=self.close(s,q,'LAYA CLOSE') if p else 'Flat → WAIT'
   elif action=='BREAKEVEN':status=self.breakeven(s,q)
   else:status='WAIT'
   actual=self.state['positions'].get(s,{}).get('leverage')
   self.signals[s]={'time':stamp(),**result,'applied_leverage':actual,'execution':status}
   self.reviewed[s]=time.time();self.last_decision=stamp();self.errors.pop('AI '+s,None);self.save()
   try:
    with (self.path.parent/'decisions.jsonl').open('a',encoding='utf8') as log:
     log.write(json.dumps({'symbol':s,**self.signals[s],'state':item['state']},allow_nan=False)+'\n')
   except OSError as exc:self.errors['Audit']=str(exc)
 def decision_loop(self):
  while not self.shutdown.is_set():
   began=time.monotonic()
   with self.lock:active=self.active;generation=self.generation;positions=list(self.state['positions'])
   if active:
    symbols=self.scanner.candidates(self.reviewed,self.batch_size,positions) if self.scanner else positions
    for offset in range(0,len(symbols),self.batch_size):
     if self.shutdown.is_set():break
     with self.lock:
      if not self.active or generation!=self.generation:break
     futures={self.pool.submit(self.prepare,s):s for s in symbols[offset:offset+self.batch_size]};items=[]
     for future in as_completed(futures):
      try:items.append(future.result())
      except Exception as exc:
       s=futures[future]
       with self.lock:self.errors['AI '+s]=stamp()+' '+str(exc);self.reviewed[s]=time.time()
     if not items:continue
     try:
      results=self.brain.predict_batch([i['state'] for i in items])
      if len(results)!=len(items):raise ValueError('Mismatched LAYA batch')
      for item,result in zip(items,results):
       try:self.apply_decision(item,result,generation)
       except Exception as exc:
        with self.lock:self.errors['AI '+item['symbol']]=stamp()+' '+str(exc)
     except Exception as exc:
      with self.lock:self.errors['LAYA']=stamp()+' '+str(exc)
   self.shutdown.wait(max(1,30-(time.monotonic()-began)) if active else 1)
 def reconcile_funding(self):
  with self.lock:pending=[(i,dict(t)) for i,t in enumerate(self.state['trades']) if t.get('funding_pending')][:20]
  for index,trade in pending:
   try:
    events=self.feed.funding(trade['symbol'],trade['funding_cursor'])
    with self.lock:
     current=self.state['trades'][index]
     for e in sorted(events,key=lambda e:int(e['fundingTime'])):
      ts=int(e['fundingTime'])
      if current['funding_cursor']<ts<=current['closed_ms']:
       rate=float(e['fundingRate']);mark=float(e['markPrice'])
       if not math.isfinite(rate) or not math.isfinite(mark) or mark<=0:raise ValueError('Invalid funding')
       cash=-current['side']*current['qty']*mark*rate
       current['funding']+=cash;current['net']+=cash;self.state['funding']+=cash;self.state['balance']+=cash;current['funding_cursor']=ts
     if time.time()*1000-current['closed_ms']>300000:current['funding_pending']=False
     self.save()
   except Exception as exc:
    with self.lock:self.errors['Funding reconcile']=str(exc)
 def monitor_loop(self):
  while not self.shutdown.is_set():
   with self.lock:positions={s:p['funding_cursor'] for s,p in self.state['positions'].items()}
   for s,cursor in positions.items():
    try:
     events=self.feed.funding(s,cursor);q=self.feed.quote(s)
     with self.lock:self.apply_funding(s,events);self.risk(s,q);self.errors.pop('Feed '+s,None);self.save()
    except Exception as exc:
     # Funding endpoint failure must not disable stop monitoring.
     with self.lock:self.errors['Feed '+s]=stamp()+' '+str(exc)
     try:
      q=self.feed.quote(s)
      with self.lock:self.risk(s,q);self.save()
     except Exception:pass
   self.reconcile_funding()
   with self.lock:
    if positions:
     self.state['curve'].append({'time':stamp(),'equity':self.equity()});self.state['curve']=self.state['curve'][-10000:];self.save()
   self.shutdown.wait(3)
 def launch_workers(self):
  if self.threads:return
  for fn in (self.monitor_loop,self.decision_loop):
   t=threading.Thread(target=fn,daemon=True);self.threads.append(t);t.start()
 def start(self,allocation=7):
  if not self.brain:return 'LAYA not loaded; no trading enabled.'
  with self.lock:
   if self.active:return 'Already running'
   self.state['allocation']=max(1,min(float(allocation),30));self.generation+=1;self.active=True;self.save()
  if self.scanner:self.scanner.start()
  self.launch_workers();return 'RUNNING • all active USDT perpetuals scanned; LAYA decides in batches'
 def pause(self):
  with self.lock:self.active=False;self.generation+=1
  return 'Paused • scanner and position risk monitoring continue'
 def close_all(self):
  self.pause()
  with self.lock:positions={s:p['funding_cursor'] for s,p in self.state['positions'].items()}
  errors=[]
  for s,cursor in positions.items():
   try:
    events=self.feed.funding(s,cursor);q=self.feed.quote(s)
    with self.lock:
     if s in self.state['positions']:self.apply_funding(s,events);self.close(s,q,'MANUAL CLOSE');self.save()
   except Exception as exc:errors.append(s+': '+str(exc))
  return '\n'.join(errors) if errors else 'All positions closed; bot paused'
 def reset(self,confirmed):
  with self.lock:
   if not confirmed:return 'Confirm reset first'
   if self.active or self.state['positions']:return 'Pause and close all positions first'
   self.path.with_name('archive-'+str(time.time_ns())+'.json').write_text(json.dumps(self.state))
   self.state.update(balance=2000.,fees=0.,funding=0.,positions={},trades=[],curve=[],last_entry_bar={});self.signals.clear();self.save()
  return 'New virtual futures account: $2,000'
 def stop(self):
  self.pause();self.shutdown.set();self.pool.shutdown(wait=False,cancel_futures=True)
  if self.scanner:self.scanner.stop()
  if self.hub:self.hub.close()
