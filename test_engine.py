import tempfile,time,unittest,math
from pathlib import Path
from engine import Engine
from risk import plan,FEE,SLIP
from brain import Brain,ACTIONS

def quote(price=100):return {'bid':price,'ask':price,'mid':price,'mark':price,'at':time.time()}
class FakeFeed:
 def quote(self,s):return quote(101)
 def funding(self,s,c):return []
class FuturesTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.e=Engine(Path(self.tmp.name)/'account.json',feed=FakeFeed())
 def tearDown(self):self.e.stop();self.tmp.cleanup()
 def open(self,side=1):return self.e.open('BTCUSDT',side,quote(),.5,10,123)
 def test_dynamic_leverage(self):
  low=plan(2000,2000,100,.4,20);high=plan(2000,2000,100,2,20)
  self.assertGreater(low['leverage'],high['leverage']);self.assertLessEqual(low['leverage'],20)
  self.assertLessEqual(low['risk_usd'],15.000001)
 def test_requested_leverage_caps(self):
  p=plan(2000,2000,100,.4,2);self.assertLessEqual(p['leverage'],2)
 def test_long_accounting(self):
  self.open();p=dict(self.e.state['positions']['BTCUSDT'])
  self.assertGreater(p['leverage'],1)
  self.assertAlmostEqual(p['qty']*p['entry'],p['margin']*p['leverage'])
  self.e.close('BTCUSDT',quote(102),'test');t=self.e.state['trades'][-1]
  self.assertGreater(t['net'],0);self.assertAlmostEqual(self.e.state['balance'],2000+t['net'])
 def test_short_accounting(self):
  self.open(-1);self.e.close('BTCUSDT',quote(98),'test');t=self.e.state['trades'][-1]
  self.assertGreater(t['net'],0);self.assertAlmostEqual(self.e.state['balance'],2000+t['net'])
 def test_funding_idempotence(self):
  self.open();p=self.e.state['positions']['BTCUSDT'];p['funding_cursor']=1
  event={'fundingTime':int(time.time()*1000)-1,'fundingRate':'.001','markPrice':'100'}
  self.e.apply_funding('BTCUSDT',[event]);balance=self.e.state['balance']
  self.e.apply_funding('BTCUSDT',[event]);self.assertEqual(balance,self.e.state['balance'])
  self.assertLess(p['funding'],0)
  self.e.close('BTCUSDT',quote(),'test');self.assertAlmostEqual(self.e.state['balance'],2000+self.e.state['trades'][-1]['net'])
 def test_short_receives_positive_funding(self):
  self.open(-1);p=self.e.state['positions']['BTCUSDT'];p['funding_cursor']=1
  self.e.apply_funding('BTCUSDT',[{'fundingTime':int(time.time()*1000)-1,'fundingRate':'.001','markPrice':'100'}]);self.assertGreater(p['funding'],0)
 def test_breakeven_both_sides(self):
  for side in [1,-1]:
   self.e.state['last_entry_bar'].clear();self.open(side)
   self.e.breakeven('BTCUSDT',quote(110 if side==1 else 90))
   stop=self.e.state['positions']['BTCUSDT']['sl'];self.e.close('BTCUSDT',quote(stop),'BE')
   self.assertAlmostEqual(self.e.state['trades'][-1]['net'],0,places=7)
 def test_stop_mark_trigger_and_market_gap_fill(self):
  self.open();q=quote(90);self.e.risk('BTCUSDT',q)
  self.assertNotIn('BTCUSDT',self.e.state['positions'])
  self.assertAlmostEqual(self.e.state['trades'][-1]['exit'],90*(1-SLIP))
 def test_stale_blocks(self):
  q=quote();q['at']-=30
  with self.assertRaises(ValueError):self.e.open('BTCUSDT',1,q,.5,10,1)
  self.assertEqual(self.e.state['balance'],2000)
 def test_duplicate_and_same_candle(self):
  self.open();balance=self.e.state['balance'];self.open();self.assertEqual(balance,self.e.state['balance'])
  self.e.close('BTCUSDT',quote(),'test');self.assertIn('Already',self.open())
 def test_pause_discards_model_result(self):
  self.e.active=True;generation=self.e.generation;self.e.pause()
  self.e.apply_decision({'symbol':'BTCUSDT','prepared_at':time.time(),'identity':None}, {'action':'LONG'},generation)
  self.assertFalse(self.e.state['positions'])
 def test_restore(self):
  self.open();self.e.save();other=Engine(self.e.path)
  self.assertEqual(self.e.state,other.state);self.assertFalse(other.active);other.stop()
 def test_model_shape_validation(self):
  result={'answers':{'action':{'probabilities':{k:1/5 for k in ACTIONS}},'leverage':{'probabilities':{k:1/6 for k in ['1','2','3','5','10','20']}}}}
  parsed=Brain.parse(result,1);self.assertIn(parsed['action'],ACTIONS)
  result['answers']['action']['probabilities']['LONG']=float('nan')
  with self.assertRaises(ValueError):Brain.parse(result,1)
if __name__=='__main__':unittest.main()
