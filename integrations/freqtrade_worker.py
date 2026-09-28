"""Real IStrategy execution; stdout is a JSON transport, diagnostics go to stderr."""
import sys,json,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
with contextlib.redirect_stdout(sys.stderr):
 from HybridTechnicalV2 import HybridTechnicalV2
 from technical import frame,summarize
 import freqtrade
 payload=json.load(sys.stdin)
 strategy=HybridTechnicalV2(config={})
 df=frame(payload['rows'])
 meta={'pair':payload['symbol'].replace('USDT','/USDT')}
 df=strategy.populate_indicators(df,meta)
 df=strategy.populate_entry_trend(df,meta)
 df=strategy.populate_exit_trend(df,meta)
 result=summarize(df)
 result['native_version']=freqtrade.__version__
print(json.dumps(result,allow_nan=False))
