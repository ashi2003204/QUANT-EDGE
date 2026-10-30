"""Quant Edge portfolio risk dashboard for the Serendib University Endowment."""
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import chi2, genpareto, norm
import plotly.express as px

TICKERS = ["SPY", "EFA", "EEM", "AGG", "GLD", "XLE", "XLK"]
st.set_page_config(page_title="Quant Edge | Endowment Risk", layout="wide")
st.title("Quant Edge | Serendib University Endowment")
st.caption("USD 10 million mandate · conservative, long-term capital preservation")
with st.sidebar:
    st.header("Analysis settings")
    conf = st.selectbox("Confidence level", [.90, .95, .99], index=1, format_func=lambda x:f"{x:.0%}")
    method = st.selectbox("Backtest VaR method", ["Historical", "Parametric", "Monte Carlo"])
    start = st.date_input("Start date", date(2015,1,1))
    end = st.date_input("End date", date.today())
    threshold_pct = st.slider("EVT threshold percentile", 90, 97, 95)
    st.caption("Risk-free rate proxy: 3% annualized placeholder. Replace with FRED data for final submission.")

@st.cache_data(ttl=3600)
def get_prices(a,b):
    raw=yf.download(TICKERS,start=a,end=b,auto_adjust=True,progress=False)
    px=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw
    return px.reindex(columns=TICKERS).ffill(limit=3).dropna()

try:
    prices=get_prices(start,end)
    rets=np.log(prices/prices.shift()).dropna()
except Exception as e:
    st.error(f"Price download failed: {e}. Check internet access and date range."); st.stop()
if len(rets)<260 or len(rets.columns)<2:
    st.error("Need at least one year of aligned data and two available assets."); st.stop()
assets=list(rets.columns); n=len(assets); mu=rets.mean()*252; cov=rets.cov()*252
res=minimize(lambda w:w@cov@w,np.repeat(1/n,n),method="SLSQP",bounds=[(0,.35)]*n,
             constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"maxiter":2000})
w=res.x if res.success else np.repeat(1/n,n)
weights=pd.Series(w,index=assets); r=rets@weights; loss=-r; wealth=np.exp(r.cumsum()); draw=wealth/wealth.cummax()-1
annret=float(r.mean()*252); vol=float(r.std(ddof=1)*np.sqrt(252)); rf=.03
hist=float(loss.quantile(conf)); hist_es=float(loss[loss>=hist].mean()); sigma=float(r.std(ddof=1))
param=float(-r.mean()+norm.ppf(conf)*sigma); param_es=float(-r.mean()+norm.pdf(norm.ppf(1-conf))/(1-conf)*sigma)
rng=np.random.default_rng(17); sim=rng.normal(r.mean(),sigma,100000); mc=float(-np.quantile(sim,1-conf)); mc_es=float(-sim[sim<=np.quantile(sim,1-conf)].mean())
vals={"Historical":(hist,hist_es),"Parametric":(param,param_es),"Monte Carlo":(mc,mc_es)}
threshold=float(loss.quantile(threshold_pct/100)); excess=loss[loss>threshold]-threshold; ev=es_evt=np.nan
if len(excess)>=30:
    shape,_,scale=genpareto.fit(excess,floc=0); q=(1-conf)/(len(excess)/len(loss))
    ev=threshold+scale/shape*(q**(-shape)-1) if abs(shape)>1e-6 else threshold-scale*np.log(q)
    if shape<1: es_evt=(ev+scale-shape*threshold)/(1-shape)
st.subheader("Portfolio recommendation: long-only minimum variance (35% asset cap)")
c=st.columns(4); c[0].metric("Annualized return",f"{annret:.1%}"); c[1].metric("Annualized volatility",f"{vol:.1%}"); c[2].metric("Sharpe ratio",f"{(annret-rf)/vol:.2f}"); c[3].metric("Maximum drawdown",f"{draw.min():.1%}")
a,b=st.columns(2); a.plotly_chart(px.bar(weights.sort_values(),orientation="h",title="Recommended weights"),use_container_width=True); b.plotly_chart(px.line(wealth,title="Growth of $1"),use_container_width=True)
st.plotly_chart(px.area(draw,title="Portfolio drawdown"),use_container_width=True)
st.subheader(f"Downside risk · {conf:.0%} confidence · one day")
selected_var,selected_es=vals[method]
st.dataframe(pd.DataFrame({"Measure":["Selected VaR: "+method,"Selected Expected Shortfall","EVT POT-GPD VaR","EVT POT-GPD ES"],"Loss %":[selected_var,selected_es,ev,es_evt],"Loss on $10m":[selected_var*1e7,selected_es*1e7,ev*1e7 if np.isfinite(ev) else np.nan,es_evt*1e7 if np.isfinite(es_evt) else np.nan]}).style.format({"Loss %":"{:.2%}","Loss on $10m":"${:,.0f}"}),use_container_width=True)
with st.expander("Compare all three VaR methods"):
    st.dataframe(pd.DataFrame({k:{"VaR":v[0],"ES":v[1]} for k,v in vals.items()}).T.style.format("{:.2%}"))

st.subheader("Out-of-sample backtest")
split=int(.7*len(r)); train=r.iloc[:split]; test=r.iloc[split:]; alpha=1-conf
if method=="Historical": forecast=np.repeat(-train.quantile(alpha),len(test))
elif method=="Parametric": forecast=np.repeat(-train.mean()+norm.ppf(conf)*train.std(ddof=1),len(test))
else: forecast=np.repeat(-np.quantile(rng.normal(train.mean(),train.std(ddof=1),100000),alpha),len(test))
hit=(-test.to_numpy()>forecast).astype(int); N=len(hit); x=int(hit.sum()); ph=max(x/N,1e-10)
lr=-2*((N-x)*np.log(alpha)+x*np.log(1-alpha)-(N-x)*np.log(1-ph)-x*np.log(ph)); pof=chi2.sf(lr,1)
st.write(f"Chronological 70/30 split · test {test.index.min().date()} to {test.index.max().date()} · {x}/{N} exceptions ({x/N:.2%}) · Kupiec POF p={pof:.3f}")
st.warning("Christoffersen independence test and Basel regulatory traffic-light thresholds should be reviewed before submission; this dashboard's backtest is a simplified competition prototype.")
st.subheader("Historical and hypothetical stress")
sc=[("Global Financial Crisis", "2008-09-01","2009-03-31"),("COVID crash","2020-02-19","2020-03-23"),("2022 rate shock","2022-01-03","2022-10-12")]; rows=[]
for name,a0,b0 in sc:
    z=rets.loc[a0:b0]
    if len(z): rows.append((name,float(np.exp((z@weights).sum())-1),len(z)))
shock=pd.Series({t:(-.15 if t in ["SPY","EFA","EEM","XLK"] else -.10 if t=="XLE" else -.04 if t=="AGG" else .05) for t in assets})
rows.append(("Hypothetical equity selloff; bonds weak; gold +5%",float(weights@shock),1))
st.dataframe(pd.DataFrame(rows,columns=["Scenario","Portfolio impact","Days"]).style.format({"Portfolio impact":"{:.1%}"}),hide_index=True)
st.header("Assumptions & Limitations")
st.markdown("""- Historical daily covariance and a long-only 35% asset cap determine weights; this is not liability-aware. No transaction costs, taxes, FX effects, liquidity discounts, or active rebalancing are modelled.\n- Daily log returns are used for analysis. Adjusted prices are downloaded with yfinance; calendars are aligned by common observations and short gaps are forward-filled (up to three sessions). Flag data anomalies before relying on results.\n- Parametric/Monte Carlo VaR assume normal returns; the Monte Carlo method does not model changing volatility. EVT uses a generalized Pareto fit and is sensitive to threshold and sample length.\n- The test is a single chronological 70/30 split. Add rolling out-of-sample tests, Christoffersen independence testing, and validate traffic-light definitions before final judging.\n- Stress windows are historical observations; the hypothetical shock is illustrative, not a forecast. 3% risk-free rate is a placeholder; replace with FRED Treasury data. Run and report ADF stationarity tests.\n- Competition research prototype only; validate calculations and explain the choices before presenting. Not investment advice.""")
