"""Conservative paper risk budget. Leverage never multiplies an arbitrary fixed bet."""
import math
FEE=.0005
SLIP=.0002
MAINTENANCE=.005 # Assumption, NOT the exchange's account-specific tier table.

def plan(equity,free,price,atr,proposed,max_leverage=20,allocation=.07,risk_fraction=.0075):
 if any(not math.isfinite(float(v)) for v in [equity,free,price,atr,proposed]):raise ValueError('Invalid risk input')
 if equity<=0 or free<=0 or price<=0 or atr<=0:raise ValueError('Nonpositive risk input')
 stop_fraction=max(2*atr/price,.008)
 margin_budget=min(equity*allocation,free*.95)
 risk_budget=equity*risk_fraction
 costs=2*(FEE+SLIP)
 risk_cap=max(1,int(risk_budget/(margin_budget*(stop_fraction+costs))))
 liquidation_cap=max(1,int(1/(2*stop_fraction+MAINTENANCE+costs)))
 leverage=max(1,min(int(proposed),int(max_leverage),risk_cap,liquidation_cap))
 notional=min(margin_budget*leverage,risk_budget/(stop_fraction+costs))
 margin=notional/leverage
 if notional<10 or margin+notional*FEE>free:raise ValueError('Insufficient margin')
 return {'leverage':leverage,'proposed_leverage':int(proposed),'notional':notional,'margin':margin,
 'stop_fraction':stop_fraction,'risk_usd':notional*(stop_fraction+costs),
 'reason':f'LAYA {int(proposed)}x; risk cap {risk_cap}x; liquidation-distance cap {liquidation_cap}x; configured cap {max_leverage}x'}

def liquidation(entry,side,leverage):
 return entry*(1-side/leverage)/(1-side*MAINTENANCE)
