"""Download real closed candles and call a separate native Jesse process."""
import json,os,subprocess,time
from pathlib import Path
from market import MarketFeed

def download_history(feed,symbol,bars=4000):
 if not symbol.isalnum() or not symbol.endswith('USDT'):raise ValueError('USDT symbol')
 if not 2000<=int(bars)<=10000:raise ValueError('2000–10000 bars')
 end=int(time.time()*1000)-1;found={}
 while len(found)<bars:
  batch=feed.get('klines',symbol=symbol,interval='1m',limit=min(1000,bars-len(found)),endTime=end)
  if not batch:break
  for r in batch:
   if r[6]<time.time()*1000:found[r[0]]=r
  end=batch[0][0]-1
 rows=[found[k] for k in sorted(found)][-bars:]
 if len(rows)<bars or any(b[0]-a[0]!=60000 for a,b in zip(rows,rows[1:])):raise ValueError('Incomplete candle history')
 return rows

def run_jesse(hub,symbol,bars=2000):
 python=os.environ.get('JESSE_PYTHON','')
 if not python:raise RuntimeError('JESSE_PYTHON is not configured. Run the optional Jesse install cell.')
 rows=download_history(hub.feed,symbol,int(bars))
 history=hub.directory/(symbol+'_research_input.json')
 history.write_text(json.dumps({'symbol':symbol,'rows':rows}))
 result=subprocess.run([python,str(Path(__file__).parent/'integrations/jesse_worker.py')],
  input=json.dumps({'symbol':symbol,'rows':rows}),capture_output=True,text=True,timeout=300)
 if result.returncode:raise RuntimeError(result.stderr[-3000:])
 report=json.loads(result.stdout);hub.ingest(report)
 target=hub.directory/(symbol+'_jesse_report.json');target.write_text(json.dumps(report,indent=2))
 return report,str(target)
