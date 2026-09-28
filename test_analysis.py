import json,math,tempfile,threading,time,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from market import TTLCache
from analysis_hub import Hub,arbitrage,book_summary
from technical import frame,enrich,STRATEGY_ID
from scanner import Scanner

def fixture(n=300):
 return [[i*300000,100+i*.01,101+i*.01,99+i*.01,100+i*.01+math.sin(i/5)*.1,100,(i+1)*300000-1] for i in range(n)]
def book(venue,bid,ask):return {'venue':venue,'at':time.time(),'bids':[[bid,100]],'asks':[[ask,100]]}
class AnalysisTests(unittest.TestCase):
 def test_cache_single_flight(self):
  cache=TTLCache();calls=[]
  def load():time.sleep(.01);calls.append(1);return 4
  with ThreadPoolExecutor(10) as pool:values=list(pool.map(lambda _:cache.get('x',60,load),range(20)))
  self.assertEqual(len(calls),1);self.assertEqual(values,[4]*20)
 def test_candle_boundary_invalidates(self):
  c=TTLCache();self.assertEqual(c.get('x',300,lambda:1,version=1),1)
  self.assertEqual(c.get('x',300,lambda:2,version=2),2)
 def test_no_lookahead(self):
  data=frame(fixture());short=enrich(data.iloc[:270]);full=enrich(data)
  pd.testing.assert_frame_equal(short,full.iloc[:270])
 def test_depth_costs(self):
  result=arbitrage(book('a',100,100.1),book('b',100.2,100.3))
  self.assertLess(result['net_usd'],0)
 def test_stale_rejected(self):
  b=book('a',100,101);b['at']-=30
  with self.assertRaises(ValueError):book_summary(b)
 def test_wrong_research_code_and_market_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   h=Hub(d)
   with self.assertRaises(ValueError):h.ingest({'schema':1,'source':'jesse','kind':'research','symbol':'BTCUSDT'})
   packet={'schema':1,'market':'USDT_PERPETUAL','source':'jesse','kind':'research','symbol':'BTCUSDT','strategy_id':'wrong','timeframe':'5m'}
   with self.assertRaises(ValueError):h.ingest(packet)
   h.close()
 def test_research_roundtrip(self):
  with tempfile.TemporaryDirectory() as d:
   h=Hub(d);now=time.time()
   report={'schema':1,'market':'USDT_PERPETUAL','source':'jesse','kind':'research','symbol':'BTCUSDT','strategy_id':STRATEGY_ID,'timeframe':'5m',
    'data_start':now-1000,'data_end':now-20,'generated_at':now,'metrics':{'return_pct':-2,'max_drawdown_pct':4,'trades':20}}
   h.ingest(report);h.close();h2=Hub(d);self.assertEqual(len(h2.reports),1);h2.close()
 def test_fair_ai_queue_and_stale_exclusion(self):
  s=Scanner(None);now=time.time()
  s.rows={x:{'symbol':x,'eligible':True,'at':now,'volume24h':10} for x in ['AUSDT','BUSDT','CUSDT']}
  self.assertEqual(s.candidates({'AUSDT':now},1,[]),['BUSDT'])
  s.rows['BUSDT']['at']=0
  self.assertEqual(s.candidates({'AUSDT':now},1,[]),['CUSDT']);s.stop()
if __name__=='__main__':unittest.main()
