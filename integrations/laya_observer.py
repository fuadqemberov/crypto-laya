"""Run inside an installed Hummingbot environment. Public futures data only."""
import argparse,asyncio,json,os,time
from itertools import islice
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative import BinancePerpetualDerivative

async def main(symbols):
 from gradio_client import Client
 connector=BinancePerpetualDerivative(trading_pairs=symbols,trading_required=False)
 client=None
 await connector.start_network()
 try:
  while True:
   if connector.ready:
    try:
     if client is None:
      client=await asyncio.to_thread(Client,os.environ.get('LAYA_UI_URL','http://127.0.0.1:7860'),auth=('fuad',os.environ['LAYA_UI_PASSWORD']),verbose=False)
     for pair in symbols:
      book=connector.get_order_book(pair);now=time.time()
      packet={'schema':1,'market':'USDT_PERPETUAL','source':'hummingbot','kind':'orderbook',
       'symbol':pair.replace('-',''),'generated_at':now,'books':[{'venue':'binance_perpetual','at':now,
       'bids':[[float(r.price),float(r.amount)] for r in islice(book.bid_entries(),50)],
       'asks':[[float(r.price),float(r.amount)] for r in islice(book.ask_entries(),50)]}]}
      await asyncio.to_thread(client.predict,json.dumps(packet),api_name='/ingest')
    except Exception as exc:
     client=None;print('Bridge error:',type(exc).__name__,flush=True)
   await asyncio.sleep(5)
 finally:await connector.stop_network()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--symbols',default='BTC-USDT,ETH-USDT,SOL-USDT');a=p.parse_args()
 asyncio.run(main([s.strip() for s in a.symbols.split(',')]))
