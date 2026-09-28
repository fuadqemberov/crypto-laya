import argparse,copy,json,os,secrets
from pathlib import Path
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from engine import Engine
from market import MarketFeed
from scanner import Scanner
from analysis_hub import Hub
from brain import Brain

ROOT=Path(__file__).resolve().parent

def build_ui(bot):
 def snapshot():
  with bot.lock:
   state=copy.deepcopy(bot.state);quotes=copy.deepcopy(bot.quotes);signals=copy.deepcopy(bot.signals)
   errors=dict(bot.errors);eq=bot.equity();free=bot.free();active=bot.active
  blocks=[]
  for label,value in [('Equity',eq),('Wallet',state['balance']),('Free margin',free),('Total PnL',eq-2000),('Fees',state['fees']),('Funding cashflow',state['funding'])]:
   blocks.append(f'<div style="flex:1;min-width:140px;background:#142338;padding:18px;border-radius:12px"><div style="color:#a4b3c8;font-size:13px">{label}</div><b style="font-size:25px;color:{"#34d399" if value>=0 else "#fb7185"}">${value:,.2f}</b></div>')
  import time
  pos=[]
  for s,p in state['positions'].items():
   q=quotes.get(s);mark=q['mark'] if q else p['entry']
   pos.append([s,'LONG' if p['side']==1 else 'SHORT',p['leverage'],p['entry'],mark,p['qty'],p['margin'],bot.pnl(p,mark),p['sl'],p['tp'],p['liquidation_est'],p['funding'],round(time.time()-q['at'],1) if q else None])
  trades=[[t.get(k) for k in ['symbol','direction','leverage','opened','closed','entry','exit','gross','fees','funding','net','reason']] for t in reversed(state['trades'][-500:])]
  decisions=[[s,v['time'],v['action'],round(max(v['probabilities'].values())*100,1),v['proposed_leverage'],v['applied_leverage'],v['batch_latency_ms'],v['execution']] for s,v in list(signals.items())[-200:]]
  curve=pd.DataFrame(state['curve'] or [{'time':pd.Timestamp.now(tz='UTC').isoformat(),'equity':eq}]);curve['time']=pd.to_datetime(curve['time'],utc=True).dt.tz_localize(None)
  scan,rows=bot.scanner.snapshot() if bot.scanner else ({},[])
  universe=[[r['symbol'],r['signal'],r.get('rsi'),r.get('volume24h'),r.get('spread_bps'),r.get('reason',r.get('error')),round(time.time()-r['at'],1)] for r in sorted(rows,key=lambda r:r['symbol'])]
  with bot.hub.lock:experts=copy.deepcopy(bot.hub.latest)
  status=f"{'🟢 RUNNING' if active else '⏸ PAUSED'} · LAYA {getattr(bot.brain,'device','OFF')} · {scan.get('done',0)}/{scan.get('total',0)} perpetuals · Last scan {scan.get('cycle_seconds',0)}s"
  if scan.get('error'):errors['Scanner']=scan['error']
  return '<div style="display:flex;gap:10px;flex-wrap:wrap">'+''.join(blocks)+'</div>',status,pos,trades,decisions,curve,universe,bot.hub.status(),experts,bot.feed.stats(),'\n'.join(f'{k}: {v}' for k,v in errors.items()) or 'No errors'
 def export():
  with bot.lock:
   target=bot.path.parent/'trades.csv';pd.DataFrame(bot.state['trades']).to_csv(target,index=False)
  return str(target)
 def ingest(payload):return bot.hub.ingest(payload)
 def import_file(path):
  if not path:return 'Choose a JSON report'
  if Path(path).stat().st_size>200000:raise ValueError('Report too large')
  return ingest(Path(path).read_text())
 def backtest(symbol,bars):
  from research_lab import run_jesse
  return run_jesse(bot.hub,symbol.strip().upper(),int(bars))
 with gr.Blocks(title='LAYA Futures Lab') as demo:
  gr.Markdown('# LAYA · Futures Lab\n**USDT PERPETUAL · PAPER $2,000 · 5m / 15m / 1h · Dynamic leverage**')
  gr.Markdown('Bütün aktiv Binance USDT perpetual cütlükləri skan edilir. LAYA giriş, çıxış və X təklifini verir. Bütün sifarişlər virtualdır.')
  metrics=gr.HTML();status=gr.Markdown()
  with gr.Row():
   allocation=gr.Slider(1,30,value=bot.state['allocation'],step=1,label='Hər mövqe üçün maksimum margin / equity %')
   start=gr.Button('▶ Başlat',variant='primary');pause=gr.Button('⏸ Pause');close=gr.Button('Bütün mövqeləri bağla',variant='stop')
  message=gr.Textbox(label='Status',interactive=False)
  chart=gr.LinePlot(x='time',y='equity',title='Equity • USDT ≈ USD',height=240)
  with gr.Tab('Futures mövqeləri'):
   positions=gr.Dataframe(headers=['Simvol','Direction','X','Entry','Mark','Qty','Margin','Unrealized PnL','SL','TP','Est. liquidation','Funding','Quote age s'],interactive=False)
   gr.Markdown('Liquidation qiyməti 0.5% maintenance fərziyyəsi ilə təxminidir; real birjanın tier hesablaması deyil. Funding gecikməsi zamanı yekun PnL düzəlişi ola bilər.')
  with gr.Tab('LAYA qərarları'):
   decisions=gr.Dataframe(headers=['Simvol','UTC','Action','Model score %','LAYA X','Applied X','Batch ms','İcra / risk səbəbi'],interactive=False)
   gr.Markdown('Model score win-rate deyil. Qərarın datası və hər iki probability paylanması data/decisions.jsonl faylına yazılır. X yalnız yeni mövqe açılarkən tətbiq edilir.')
  with gr.Tab('Bütün coinlər'):
   universe=gr.Dataframe(headers=['Simvol','Texniki siqnal','RSI','24h USDT volume','Spread bps','Filter / xəta','Age s'],interactive=False)
   gr.Markdown('Texniki skan bütün cütlükləri əhatə edir. Likvidlik və spread şərtlərini keçənlər LAYA növbəsinə düşür; əvvəllər baxılmamış və köhnə analizlər öncə işlənir. Yeni listinqdə 250 candle yoxdursa ayrıca xəta görünür.')
  with gr.Tab('Helper integrations'):
   integration_status=gr.JSON(label='Faktiki qoşulma statusu');experts=gr.JSON(label='Son ekspert datası')
   packet=gr.Textbox(label='Native JSON packet',lines=2);submit=gr.Button('İdxal et')
   # API endpoint is protected by the same local UI credentials.
   submit.click(ingest,inputs=packet,outputs=message,api_name='ingest')
   report=gr.File(file_types=['.json'],type='filepath',label='Jesse / LEAN hesabatı');report.upload(import_file,inputs=report,outputs=message)
  with gr.Tab('Jesse research'):
   gr.Markdown('Əlavə Jesse mühiti tələb olunur. Bu, texniki futures baseline testidir; LAYA strategiyasının backtest-i deyil. 1m datadan 5m strategiya qurulur. Nəticələr yalnız əlavə tarixi kontekstdir.')
   with gr.Row():
    research_symbol=gr.Textbox(value='BTCUSDT',label='Simvol');bars=gr.Slider(2000,10000,value=4000,step=1000,label='1m input bars')
   run_test=gr.Button('Native Jesse backtest');research_result=gr.JSON();research_file=gr.File()
   run_test.click(backtest,inputs=[research_symbol,bars],outputs=[research_result,research_file])
  with gr.Tab('Tarixçə'):
   trades=gr.Dataframe(headers=['Simvol','Direction','X','Opened UTC','Closed UTC','Entry','Exit','Gross','Fees','Funding','Net','Reason'],interactive=False)
   download=gr.Button('CSV');csv=gr.File();download.click(export,outputs=csv)
  with gr.Tab('Performance / Reset'):
   cache=gr.JSON(label='Cache / API');errors=gr.Textbox(label='Monitor errors',lines=5)
   gr.Markdown('Pause yeni AI əməliyyatlarını dayandırır; SL/TP monitoru və skan davam edir. Proqram bağlandıqda monitor dayanır. Gap-lər stop qiymətindən daha pis fill yarada bilər.')
   confirm=gr.Checkbox(label='Tarixçəni arxivlə və $2,000 virtual futures hesabına reset et');reset=gr.Button('Reset');reset.click(bot.reset,inputs=confirm,outputs=message)
  start.click(bot.start,inputs=allocation,outputs=message);pause.click(bot.pause,outputs=message);close.click(bot.close_all,outputs=message)
  outputs=[metrics,status,positions,trades,decisions,chart,universe,integration_status,experts,cache,errors]
  demo.load(snapshot,outputs=outputs);gr.Timer(5).tick(snapshot,outputs=outputs)
 return demo

def run(no_model=False,port=7860):
 load_dotenv(ROOT/'.env')
 directory=Path(os.environ.get('LAYA_DATA',str(ROOT/'data')))
 workers=int(os.environ.get('SCAN_WORKERS','8'));batch=int(os.environ.get('LAYA_BATCH_SIZE','4'))
 if not 1<=workers<=24 or not 1<=batch<=16:raise ValueError('Workers 1–24; batch 1–16')
 feed=MarketFeed(directory,workers=workers);hub=Hub(directory);hub.feed=feed
 scanner=Scanner(feed,workers=workers,min_volume=float(os.environ.get('MIN_VOLUME_USDT','1000000')))
 print('Loading local LAYA checkpoint...' if not no_model else 'ANALYSIS ONLY: LAYA disabled; trading cannot start.')
 brain=None if no_model else Brain(batch)
 bot=Engine(directory/'futures_account.json',brain,feed,hub,scanner,batch,int(os.environ.get('MAX_LEVERAGE','20')))
 bot.launch_workers();scanner.start();demo=build_ui(bot)
 password=os.environ.get('UI_PASSWORD') or secrets.token_urlsafe(12)
 print(f'Local UI: http://127.0.0.1:{port}\nUser: fuad\nPassword: {password}')
 try:demo.queue().launch(server_name='127.0.0.1',server_port=port,share=False,auth=('fuad',password),inbrowser=True,show_error=True)
 finally:bot.stop()
 return bot,demo
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--no-model',action='store_true',help='Explicit analysis-only mode for setup/diagnostics');p.add_argument('--port',type=int,default=7860)
 args=p.parse_args();run(args.no_model,args.port)
