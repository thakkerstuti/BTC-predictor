import streamlit as st
import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from datetime import datetime
import json
import os

st.title("BTC Price Range Predictor")
# st.write("My dashboard is working")
@st.cache_data(ttl=60) 
def get_btc_data():

    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}

    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data, columns=[
        "time","open","high","low","close","volume",
        "close_time","qav","num_trades",
        "taker_base","taker_quote","ignore"
    ])

    df["close"] = df["close"].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")

    return df.set_index("time")["close"]


prices = get_btc_data()
current_price = prices.iloc[-1]

def predict_range(prices):
  log_ret = np.log(prices / prices.shift(1)).dropna()
  mu_daily = log_ret.mean()

  
  am = arch_model(log_ret * 100, vol='GARCH', p=1, q=1)
  res  = am.fit(disp='off')
  sigma_fig = res.conditional_volatility / 100
  resid      = (log_ret * 100 - res.params['mu']) / res.conditional_volatility
  nu         = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])



  def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

  H_series = rolling_entropy(resid)
  M_series = log_ret.abs().rolling(60).mean()
  bar_sigma2 = (sigma_fig**2).mean()
  redundancy  = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
  info_filter = (H_series > H_series.mean()).astype(float)
  H_max, M_max = H_series.max(), M_series.max()
  α0, δ0 = 0.5, 0.3
  if α0 * H_max + δ0 * M_max >= 1:
    fac = 0.95 / (α0 * H_max + δ0 * M_max)
    α0 *= fac
    δ0 *= fac
  base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}

  def update_params(p, sigma2, bar_sigma2, t):
    err = sigma2 - bar_sigma2
    lr  = p['eta'] / (1 + t**0.55)
    p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
    return p
  
  S0 = prices.iloc[-1]
  n_sims = 200
  n_days = 1
  dt = 1

  def simulate_cyber_gbm(S0, mu, sigma_fig, H, M,
                       params, bar_sigma2, n_steps, dt=1, eps=1e-6):
    S = np.zeros(n_steps + 1)
    V = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = sigma_fig.iloc[-1] ** 2
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    for t in range(1, n_steps + 1):
        current = -1
        H_val = min(H.iloc[current] / H_max, 1.0)
        M_val = min(M.iloc[current] / M_max, 1.0)
        crisis  = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        sigma2 = (
            sigma_fig.iloc[current]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        sigma2 *= max(1e-12, redundancy.iloc[current])
        sigma2 *= 1 + 0.5 * info_filter.iloc[current]
        sigma2 = max(eps, min(sigma2, 0.5))
        Z   = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t]= S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
        V[t]= sigma2
        params = update_params(params, sigma2, bar_sigma2, t)
    return S, V

  def simulate_mc(S0, mu, sigma_fig, H, M, bar_sigma2,
                n_sims=200, n_days=1):
    out = np.zeros((n_sims, n_days + 1))
    for i in range(n_sims):
        paths, _ = simulate_cyber_gbm(
            S0, mu, sigma_fig, H, M,
            base_params.copy(),
            bar_sigma2, n_days, dt
        )
        out[i] = paths
    return out

  paths = simulate_mc(S0, mu_daily, sigma_fig, H_series, M_series,
                    bar_sigma2, n_sims, n_days)

  S_t1    = paths[:, 1]
  low_t1, high_t1 = np.percentile(S_t1, [5, 95])
  return low_t1, high_t1

def save_prediction(timestamp, low, high):
    record = {
        "time": str(timestamp),
        "low": float(low),
        "high": float(high),
        "actual": None
    }

    file = "predictions_history.jsonl"

    with open(file, "a") as f:
        f.write(json.dumps(record) + "\n")

def load_history():
    file = "predictions_history.jsonl"

    if not os.path.exists(file):
        return pd.DataFrame()

    return pd.read_json(file, lines=True)

st.metric("Current BTC Price", f"{current_price:,.2f}")


@st.cache_data(ttl=60)
def predict_range_cached(prices):
    return predict_range(prices)

low95, high95 = predict_range_cached(prices) 
save_prediction(prices.index[-1], low95, high95)

st.subheader("Predicted Range (Next Hour)")
st.metric("Predicted 1h Range", f"{low95:,.2f} → {high95:,.2f}")

st.markdown("---")

st.subheader("Last 50 Hours")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(10, 4))
lower = low95
upper = high95

ax.axhline(lower, color='red', linestyle='--')
ax.axhline(upper, color='green', linestyle='--')

ax.plot(prices.tail(50), color='blue')


ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())

plt.xticks(rotation=45)   
plt.tight_layout()
future_time = prices.index[-1] + pd.Timedelta(hours=1)
pred_mid = (low95 + high95) / 2

ax.scatter(future_time, pred_mid, color='yellow', label='Prediction')
ax.legend()

st.pyplot(fig)

st.caption("Model: GARCH + Monte Carlo simulation (confidence ~90%)")
st.caption("Next 1-hour prediction (90% interval)")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

try:
    df_bt = pd.read_json("backtest_results.jsonl", lines=True)

    coverage_95 = df_bt["coverage_95"].mean()
    avg_width = df_bt["width_95"].mean()
    winkler_score = df_bt["winkler"].mean()
except:
    coverage_95, avg_width, winkler_score = 0, 0, 0

st.markdown("---")
st.subheader("Backtest Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Coverage (95%)", f"{coverage_95:.2%}")
col2.metric("Avg Width", f"{avg_width:,.2f}")
col3.metric("Winkler Score", f"{winkler_score:,.2f}")

st.markdown("---")
st.subheader("Prediction History")

df_hist = load_history()

if not df_hist.empty:
    st.dataframe(df_hist.tail(10))
else:
    st.write("No history yet")