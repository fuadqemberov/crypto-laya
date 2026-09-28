"""Actual local LAYA inference, shared batches, no heuristic decision fallback."""
import json,math,threading,time
ACTIONS={'LONG':'Open long when flat and evidence favors rising futures prices.',
'SHORT':'Open short when flat and evidence favors falling futures prices.',
'WAIT':'Wait or hold the current position when evidence is uncertain.',
'CLOSE':'Close an existing position when its thesis is invalid or risk is excessive.',
'BREAKEVEN':'Move the stop of a sufficiently profitable position to cover costs.'}
QUESTIONS={
'action':{'type':'choice','instructions':'Decide a futures position action using 5m, 15m, 1h trends, funding, current position and experts. Historical baseline results are not a forecast. Arbitrage spread is not price direction. Prefer WAIT if unclear.', 'criteria':ACTIONS},
'leverage':{'type':'choice','instructions':'Choose initial leverage. High volatility, mixed trends, weak evidence or expensive funding calls for lower leverage. Risk engine may lower your choice.',
 'criteria':{'1':'No leverage amplification','2':'Low leverage','3':'Modest leverage','5':'Moderate leverage','10':'High leverage only with strong aligned evidence','20':'Very high leverage only with strong aligned evidence and low volatility'}}}

class Brain:
 def __init__(self,batch_size=4):
  from laya import Router
  import torch
  self.device='cuda' if torch.cuda.is_available() else 'cpu'
  if self.device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
  self.router=Router(device=self.device);self.batch_size=batch_size;self.lock=threading.Lock()
  self.predict_batch([{'market':'warmup','position':'flat'}])
 @staticmethod
 def parse(result,elapsed):
  answers=result['answers'];p=answers['action']['probabilities'];lp=answers['leverage']['probabilities']
  for values,keys in [(p,set(ACTIONS)),(lp,{'1','2','3','5','10','20'})]:
   if set(values)!=keys or any(not isinstance(v,(float,int)) or not math.isfinite(v) or not 0<=v<=1 for v in values.values()) or not .98<=sum(values.values())<=1.02:raise ValueError('Invalid LAYA distribution')
  return {'action':max(p,key=p.get),'probabilities':p,'proposed_leverage':int(max(lp,key=lp.get)),
          'leverage_probabilities':lp,'batch_latency_ms':round(elapsed,1)}
 def predict_batch(self,states):
  requests=[{'state':json.dumps(s,separators=(',',':')),'questions':QUESTIONS,'model':'multilingual','max_len':2048} for s in states]
  start=time.monotonic()
  with self.lock:
   result=self.router.predict_batch(requests,batch_size=self.batch_size)
  elapsed=(time.monotonic()-start)*1000
  if len(result)!=len(states):raise ValueError('LAYA batch size mismatch')
  return [self.parse(r,elapsed) for r in result]
