# Quant Edge 1.0 | Serendib Endowment

Reproducible Streamlit research prototype for the Quant Edge competition brief. It downloads adjusted ETF prices from Yahoo Finance and displays a constrained long-only minimum-variance portfolio, performance, three VaR/ES estimates, a POT-GPD tail estimate, a chronological out-of-sample VaR check, and historical/hypothetical stresses.

## Run locally

Requires Python 3.10 or newer and internet access for Yahoo Finance data.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app uses SPY, EFA, EEM, AGG, GLD, XLE, and XLK. Choose dates and confidence in the sidebar. Save the selected date range, downloaded data timestamp, and settings with any reported results so they can be reproduced.

## Method notes

- Returns are daily log returns from Yahoo Finance auto-adjusted prices.
- Portfolio: long-only minimum variance, fully invested, maximum 35% per ETF. This is a transparent baseline, not a liability-aware optimizer.
- VaR/ES are one-day estimates. Historical uses the empirical loss tail; parametric and Monte Carlo use a normal distribution. EVT fits a generalized Pareto distribution to exceedances above the selected empirical threshold.
- Backtest is a chronological 70/30 holdout. The app currently shows Kupiec POF; add Christoffersen independence testing and validate Basel traffic-light thresholds before treating it as a complete submission.
- The 3% risk-free rate is only a placeholder. Replace with aligned FRED 3-month Treasury data. Run ADF tests and review flagged price anomalies before final results.

## Before submission

This is a competition prototype, not a complete submission. Complete rolling backtests including Christoffersen, data-quality/ADF reporting, EVT threshold stability and uncertainty, risk contributions, and documentation of calendar alignment. Independently review every number and be prepared to explain assumptions. Disclose AI assistance and human validation as required by the brief. Do not present provisional app outputs as investment advice.
