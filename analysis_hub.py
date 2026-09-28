"""One analysis bus; no framework here owns or sends an order."""
import copy,json,math,os,subprocess,sys,threading,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from technical import frame,enrich,summarize,STRATEGY_ID

ROOT=Path(__file__).resolve().parent

def finite_tree(v):
 if isinstance(v,float) and not math.isfinite(v):raise ValueError('Non-finite number')
 if isinstance(v,dict):
  for x in v.values():finite_tree(x)
 if isinstance(v,list):
  for x in v:finite_tree(x)

def validate_book(book):
 for side,reverse in [('bids',True),('asks',False)]:
  rows=book[side]
  if not 1<=len(rows)<=100:raise ValueError('Orderbook size')
  prices=[]
  for p,q in rows:
   if not math.isfinite(p) or not math.isfinite(q) or p<=0 or q<=0:raise ValueError('Bad orderbook level')
   prices.append(p)
  if prices!=sorted(prices,reverse=reverse):raise ValueError('Unsorted orderbook')
 if book['bids'][0][0]>=book['asks'][0][0]:raise ValueError('Crossed orderbook')
 if not 0<=time.time()-book['at']<=12:raise ValueError('Stale orderbook')
 return book

def fill(levels,quantity):
 remain=quantity;value=0.
 for price,size in levels:
  take=min(remain,size);value+=take*price;remain-=take
  if remain<=quantity*1e-10:return value
 raise ValueError('Insufficient visible depth')

def book_summary(b):
 b=validate_book(b);bid=b['bids'][0][0];ask=b['asks'][0][0];mid=(bid+ask)/2
 bid_notional=sum(p*q for p,q in b['bids'][:10]);ask_notional=sum(p*q for p,q in b['asks'][:10])
 return {'spread_bps':round((ask-bid)/mid*10000,3),
 'imbalance_top10':round((bid_notional-ask_notional)/(bid_notional+ask_notional),4),
 'bid_depth10_usd':round(bid_notional,2),'ask_depth10_usd':round(ask_notional,2)}

def arbitrage(a,b,budget=140,fee_a=.001,fee_b=.001,slip=.0002):
 validate_book(a);validate_book(b)
 if abs(a['at']-b['at'])>3:raise ValueError('Venue snapshot skew >3s')
 opportunities=[]
 for buy,sell,buy_fee,sell_fee in [(a,b,fee_a,fee_b),(b,a,fee_b,fee_a)]:
  # Pick quantity then shrink to account for full depth cost and buy-side costs.
  qty=budget/(buy['asks'][0][0]*(1+buy_fee)*(1+slip))
  for _ in range(3):
   cost=fill(buy['asks'],qty)*(1+buy_fee)*(1+slip)
   if cost<=budget+1e-8:break
   qty*=budget/cost
  proceeds=fill(sell['bids'],qty)*(1-sell_fee)*(1-slip)
  opportunities.append({'buy':buy['venue'],'sell':sell['venue'],'qty':qty,'buy_cost':cost,
   'net_usd':round(proceeds-cost,5),'net_bps':round((proceeds/cost-1)*10000,3)})
 return max(opportunities,key=lambda x:x['net_usd'])

class Hub:
 def __init__(self,directory):
  self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
  self.lock=threading.RLock();self.latest={};self.native_books={};self.reports={}
  self.worker=os.environ.get('FREQTRADE_PYTHON','');self.feed=None
  self.ft_cache={};self.ft_error='Native worker not configured';self.pool=ThreadPoolExecutor(2)
  self.closed=False
  path=self.directory/'research_reports.json'
  if path.exists():
   for packet in json.loads(path.read_text()):self.ingest(packet,persist=False)
 def close(self):
  self.closed=True;self.pool.shutdown(wait=False,cancel_futures=True)
 def ingest(self,raw,persist=True):
  if isinstance(raw,str):
   if len(raw)>200000:raise ValueError('Packet >200KB')
   raw=json.loads(raw)
  packet=copy.deepcopy(raw);finite_tree(packet)
  if packet.get('schema')!=1:raise ValueError('Schema must be 1')
  source=packet.get('source');kind=packet.get('kind')
  if packet.get('market')!='USDT_PERPETUAL':raise ValueError('Futures-only hub: market must be USDT_PERPETUAL')
  symbol=packet.get('symbol','')
  if not symbol.isalnum() or not symbol.endswith('USDT'):raise ValueError('USDT symbol required')
  if source=='hummingbot' and kind=='orderbook':
   if abs(time.time()-float(packet['generated_at']))>12:raise ValueError('Hummingbot packet expired')
   if not 1<=len(packet['books'])<=3:raise ValueError('Book count')
   for book in packet['books']:
    if not isinstance(book['venue'],str) or len(book['venue'])>40:raise ValueError('Venue')
    validate_book(book)
   with self.lock:self.native_books[symbol]=packet
  elif source in ('jesse','lean') and kind=='research':
   if packet.get('strategy_id')!=STRATEGY_ID:raise ValueError('Different baseline code; report rejected')
   if packet.get('timeframe')!='5m':raise ValueError('Report timeframe must be 5m')
   start=float(packet['data_start']);end=float(packet['data_end']);generated=float(packet['generated_at'])
   if not start<end<=time.time()+5 or not end<=generated<=time.time()+5:raise ValueError('Research dates invalid')
   metrics=packet['metrics']
   for key in ['return_pct','max_drawdown_pct','trades']:
    if not isinstance(metrics.get(key),(int,float)) or isinstance(metrics[key],bool):raise ValueError('Missing numeric metric '+key)
   if not 0<=metrics['max_drawdown_pct']<=100 or metrics['trades']<0 or int(metrics['trades'])!=metrics['trades']:raise ValueError('Invalid metrics')
   packet={k:packet[k] for k in ['schema','market','source','kind','symbol','strategy_id','timeframe','data_start','data_end','generated_at','metrics']}
   packet['metrics']={k:metrics[k] for k in ['return_pct','max_drawdown_pct','trades']}
   with self.lock:
    self.reports[(source,symbol)]=packet
    if persist:
     path=self.directory/'research_reports.json';tmp=path.with_suffix('.tmp')
     tmp.write_text(json.dumps(list(self.reports.values()),allow_nan=False));os.replace(tmp,path)
  else:raise ValueError('Only native Hummingbot books and Jesse/LEAN research reports accepted')
  return source+' '+symbol+' accepted'
 def technical(self,symbol,rows):
  key=(symbol,rows[-1][0])
  if key in self.ft_cache:return copy.deepcopy(self.ft_cache[key])
  baseline=summarize(enrich(frame(rows)))
  result={'provider':'built-in technical baseline (NOT Freqtrade runtime)',**baseline}
  if self.worker:
   try:
    p=subprocess.run([self.worker,str(ROOT/'integrations/freqtrade_worker.py')],
     input=json.dumps({'symbol':symbol,'rows':rows}),text=True,capture_output=True,timeout=20)
    if p.returncode:raise RuntimeError(p.stderr[-1000:])
    native=json.loads(p.stdout);finite_tree(native)
    if native['strategy_id']!=STRATEGY_ID:raise ValueError('Freqtrade code mismatch')
    result={'provider':'Freqtrade native IStrategy',**native};self.ft_error=''
   except Exception as e:self.ft_error=str(e)
  result['native_error']=self.ft_error if result['provider'].startswith('built-in') else ''
  self.ft_cache={k:v for k,v in self.ft_cache.items() if k[0]!=symbol}
  self.ft_cache[key]=result
  return copy.deepcopy(result)
 def public_book(self,venue,symbol):
  if venue=='binance':url='https://fapi.binance.com/fapi/v1/depth?symbol='+symbol+'&limit=50'
  else:url='https://api.bybit.com/v5/market/orderbook?category=linear&symbol='+symbol+'&limit=50'
  before=time.time()
  if venue=='binance' and self.feed:d=self.feed.get('depth',symbol=symbol,limit=50)
  else:
   with urllib.request.urlopen(url,timeout=6) as r:d=json.load(r)
  now=time.time()
  if now-before>5:raise ValueError(venue+' response >5s')
  if venue=='binance':bids=d['bids'];asks=d['asks'];at=float(d.get('E',now*1000))/1000
  else:
   if d.get('retCode')!=0:raise ValueError(str(d.get('retMsg')))
   bids=d['result']['b'];asks=d['result']['a'];at=d['result']['ts']/1000
  return validate_book({'venue':venue,'at':at,'bids':[[float(p),float(q)] for p,q in bids],
                         'asks':[[float(p),float(q)] for p,q in asks]})
 def collect(self,symbol,rows):
  tech=self.technical(symbol,rows)
  with self.lock:
   packet=copy.deepcopy(self.native_books.get(symbol));reports=copy.deepcopy(self.reports)
  books=[];errors=[];provider='Public REST (NOT Hummingbot runtime)'
  if packet:
   try:
    if time.time()-packet['generated_at']>12:raise ValueError('Native Hummingbot stream expired')
    books=[validate_book(x) for x in packet['books']];provider='Hummingbot native connector'
   except Exception as e:errors.append(str(e));books=[]
  if not books:
   futures=[self.pool.submit(self.public_book,v,symbol) for v in ['binance','bybit']]
   for future in futures:
    try:books.append(future.result())
    except Exception as e:errors.append(str(e))
  market={};arb=None
  if books:
   try:market=book_summary(books[0])
   except Exception as e:errors.append(str(e))
  if len(books)>=2:
   try:arb=arbitrage(books[0],books[1])
   except Exception as e:errors.append(str(e))
  research=[]
  for source in ['jesse','lean']:
   report=reports.get((source,symbol))
   if report:
    age=(time.time()-report['data_end'])/86400
    research.append({'source':source,'baseline_only':True,'data_age_days':round(age,1),
                     'eligible_context':age<=30,**report['metrics']})
  value={'at':time.time(),'technical':tech,'microstructure':{'provider':provider,**market},
         'arbitrage':arb,'research':research,'errors':errors}
  with self.lock:self.latest[symbol]=value
  # Keep the model state bounded. Research is baseline context, not proof of LAYA performance.
  return {'technical_signal':tech['signal'],'technical_provider':tech['provider'],
          'rsi':tech['rsi'],'ema200_5m':tech['ema200'],'atr':tech['atr'],
          'orderbook':market,'cross_exchange_net_bps':arb['net_bps'] if arb else None,
          'baseline_research':[x for x in research if x['eligible_context']]}
 def status(self):
  with self.lock:
   live=sum(1 for p in self.native_books.values() if time.time()-p['generated_at']<=12)
   return {'Freqtrade':('native worker configured; '+('last call OK' if not self.ft_error else self.ft_error)) if self.worker else 'NOT CONNECTED • built-in baseline active',
           'Hummingbot':f'{live} fresh native stream(s)' if live else 'NOT CONNECTED • public REST analysis active',
           'Jesse':f'{sum(k[0]=="jesse" for k in self.reports)} imported baseline report(s)',
           'LEAN':f'{sum(k[0]=="lean" for k in self.reports)} imported baseline report(s)'}
