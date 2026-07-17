import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.stats import skew, kurtosis
from curl_cffi import requests
from scipy.stats import norm
from scipy.optimize import brentq
import quantstats as qs
from scipy.stats import linregress
import statsmodels.api as sm
import scipy.stats as sps
import plotly.graph_objs as go
import streamlit as st
from datetime import datetime, timedelta
from urllib.parse import urlparse
import lxml


st.set_page_config(page_title='Riskmapp', layout='wide')

maincol1, maincol2 = st.columns(2)

co1, co2 = st.columns([1.5, 1])

ticker = 'COIN'

@st.cache_data(ttl=86400)
def get_company_info(ticker):
    return yf.Ticker(ticker).info


def compute_adx(df, period=14):

    """
    Compute ADX, +DI, -DI
    df must contain columns: 'High', 'Low', 'Close'
    """

    high = df['High']
    low = df['Low']
    close = df['Close']

    # 1. Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # 2. True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())

    tr = np.maximum.reduce([tr1, tr2, tr3])

    # 3. Wilder’s smoothing (EMA with alpha=1/period)
    atr = pd.Series(tr, index=tr1.index).ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=tr1.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=tr1.index).ewm(alpha=1/period, adjust=False).mean()

    # 4. Directional Indicators
    plus_di = 100 * (plus_dm_smoothed / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)

    # 5. DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return pd.DataFrame({
        'ADX': adx,
        '+DI': plus_di,
        '-DI': minus_di
    })


@st.cache_data(ttl=86400)
def market_maker_pressure(ticker, tf='2y'):

    stock = yf.Ticker(ticker)
    df= stock.history(tf)
    avg_vol = int(df.Volume.rolling(60).mean().iloc[-1])

    expirations = stock.options

    all_chains = []

    for exp in expirations:
        opt_chain = stock.option_chain(exp)
        
        # Extract relevant fields
        calls = opt_chain.calls[["strike", "openInterest"]].copy()
        puts  = opt_chain.puts[["strike", "openInterest"]].copy()
        
        # Add metadata
        calls["type"] = "call"
        puts["type"]  = "put"
        
        calls["expirationDate"] = exp
        puts["expirationDate"]  = exp
        
        # Append to list
        all_chains.append(calls)
        all_chains.append(puts)

    # Combine into a single DataFrame
    options_df = pd.concat(all_chains, ignore_index=True)


    # Price range boundaries
    low_price  = df["Close"].min()
    high_price = df["Close"].max()

    # --- Filter strikes FIRST ---
    filtered = options_df[
        (options_df["strike"] >= low_price) &
        (options_df["strike"] <= high_price)
    ].copy()

    # --- Then aggregate by strike + type ---
    pivot = filtered.pivot_table(
        index="strike",
        columns="type",
        values="openInterest",
        aggfunc="sum",
        fill_value=0
    )

    pivot["netOI"] = pivot["call"] - pivot["put"]
    pivot['volumeImpactATM'] = round((pivot['netOI'] * 50) / avg_vol, 2) * 100

    buypres = pivot.nsmallest(5, 'netOI')
    sellpres =pivot.nlargest(5, 'netOI')

    net_oi_levels = pd.concat([buypres, sellpres], axis=0)

    # --- 1. Extract strike levels and scaling values ---
    levels = net_oi_levels.copy()

    # We use absolute value of volumeImpactATM for scaling width
    abs_impact = levels["volumeImpactATM"].abs()

    # Line width scaling (min 1px, max 10px for visual clarity)
    min_w, max_w = 1, 10
    impact_norm = (abs_impact - abs_impact.min()) / (abs_impact.max() - abs_impact.min() + 1e-9)
    line_widths = min_w + impact_norm * (max_w - min_w)

    # Line colors based on sign:
    # Positive volumeImpactATM → call wall → SELL pressure → resistance → GREEN
    # Negative volumeImpactATM → put wall → BUY pressure → support → RED
    line_colors = [
        "rgba(0,255,0,0.75)" if val > 0 else "rgba(255,0,0,0.75)"
        for val in levels["volumeImpactATM"]
    ]

    # --- 2. Build candlestick chart ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00BFFF",
        decreasing_line_color="#FF6347",
    ))

    # --- 3. Add horizontal lines ---
    for (strike, row), lw, col in zip(levels.iterrows(), line_widths, line_colors):
        fig.add_hline(
            y=float(strike),
            line_width=float(lw),
            line_color=col,
            annotation_text=f" {round(row['volumeImpactATM'],1)}%", 
            annotation_position="right", 
            annotation_font_color="white"
            )

    # --- 4. Dark theme layout ---
        fig.update_layout(
        title="Options Market Dealer Pressure Map — Support/Resistance From Net OI",
        height=500,
        width=1100,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True
        ),
        margin=dict(l=40, r=40, t=60, b=20)
    )
    
    fig.update_xaxes(rangeslider_visible=False)

    return fig

def hurst_exponent(data, power=6):

    #specifying the maximum power of 2
    power = power
    #the rolling sample length
    n = 2**power
    #downloading data
    stock_data = data['Close'].squeeze()
    prices = np.array(stock_data)[1:]
    returns = np.array(stock_data)[1:]/np.array(stock_data)[:-1] - 1

    #initialising arrays
    hursts = np.array([])
    tstats = np.array([])
    pvalues = np.array([])
    #calculating the rolling Hurst exponent
    for t in np.arange(n,len(returns)+1):
        #specifying the subsample
        data = returns[t-n:t]
        X = np.arange(2, power+1)
        Y = np.array([])
        for p in X:
            m = 2**p
            s = 2**(power-p)
            rs_array = np.array([])
            #moving across subsamples
            for i in np.arange(0,s):
                subsample = data[i*m:(i+1)*m]
                mean = np.average(subsample)
                deviate = np.cumsum(subsample-mean)
                difference = max(deviate) - min(deviate)
                stdev = np.std(subsample)
                rescaled_range = difference/stdev
                rs_array = np.append(rs_array, rescaled_range)
            #calculating the log2 of average rescaled range
            Y = np.append(Y, np.log2(np.average(rs_array)))
        reg = sm.OLS(Y, sm.add_constant(X))
        res = reg.fit()
        hurst = res.params[1]
        tstat = (res.params[1]-0.5)/res.bse[1]
        pvalue = 2*(1 - sps.t.cdf(abs(tstat),res.df_resid))
        hursts = np.append(hursts, hurst)
        tstats = np.append(tstats, tstat)
        pvalues = np.append(pvalues, pvalue)

    he = pd.Series(hursts, index=stock_data.index[n:], name="He")

    return he


def trend_ind(data, trading_periods = 252):
    
    returns = data.pct_change()

    true_range = returns.rolling(60).std()*np.sqrt(trading_periods)
    true_range = true_range.squeeze()

    basic_upper_band = data * (1 + true_range)
    basic_lower_band = data * (1 - true_range)

    # Convert bands to Series we can modify
    final_upper_band = basic_upper_band.copy()
    final_lower_band = basic_lower_band.copy()

    # Initialize uptrend Series
    uptrend = pd.Series(index=data.index, dtype=bool)
    uptrend.iloc[0] = True  # Starting assumption: trend is up

    # Supertrend calculation loop
    for current in range(1, len(data)):

        previous = current - 1

        if data.iloc[current] > final_upper_band.iloc[previous]:
            uptrend.iloc[current] = True
        elif data.iloc[current] < final_lower_band.iloc[previous]:
            uptrend.iloc[current] = False
        else:
            uptrend.iloc[current] = uptrend.iloc[previous]

            if uptrend.iloc[current] and final_lower_band.iloc[current] < final_lower_band.iloc[previous]:
                final_lower_band.iloc[current] = final_lower_band.iloc[previous]

            if not uptrend.iloc[current] and final_upper_band.iloc[current] > final_upper_band.iloc[previous]:
                final_upper_band.iloc[current] = final_upper_band.iloc[previous]

    supertrend = pd.Series(index=data.index)
    supertrend[uptrend] = final_lower_band[uptrend]
    supertrend[~uptrend] = final_upper_band[~uptrend]

    return supertrend


def macd_zs(data, fast: int, slow: int, signal: int, zs_window: int):

    fast_ema = data.ewm(span=fast, adjust=False).mean()
    slow_ema = data.ewm(span=slow, adjust=False).mean()

    macd = fast_ema - slow_ema
    hist = macd.ewm(span=signal, adjust=False).mean()

    return zscore(macd-hist, window=zs_window)

def zscore(data, window: int):

    mean = data.rolling(window).mean()
    std = data.rolling(window).std()

    return (data - mean) / std

def make_tz_naive(s):
    s = s.copy()
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s

st.title('Risk Management Trading Application', text_alignment='center')

box1, box2 = st.columns([0.5,2])
box3, box4, box5 = st.columns([0.5, 0.45, 1.5], gap='small')
maincol1 = st.container()


with box1:
    title = st.text_input(label="Enter a Company Ticker", value=ticker, max_chars=10, placeholder="COIN")

company = yf.Ticker(title)
company_info = company.info

with box3:

    website = company_info.get("website")

    token = 'pk_dXqZ_rdKRuKuAIDMxnuCEw'

    if website:

            domain = urlparse(website).netloc.replace("www.", "")

            logo_url = f"https://img.logo.dev/{domain}?token={token}"

            st.image(logo_url, width=240)

    
with box4:
    
    curr_price = round(company_info.get('currentPrice'),2)
    high_target = company_info.get('targetHighPrice')
    low_target = company_info.get('targetLowPrice')

    exp_growth = round(((high_target / curr_price)-1)*100, 2)
    exp_drawdown = round(((low_target / curr_price)-1)*100, 2)
    
    st.markdown(f"**Company:** {company_info.get('longName')}")
    st.markdown(f"**Sector:** {company_info.get('sector')}")
    st.markdown(f"**Industry:** {company_info.get('industryDisp')}")
    st.markdown(f"**Current Price:** {curr_price}")
    st.markdown(f"**Expected Performance:** {exp_growth}%")
    st.markdown(f"**Expected Downside:** {exp_drawdown}%")
    st.markdown(f"**Website:** {company_info.get('website')}")

        


# sec = pd.DataFrame(company.get_sec_filings())
# tenk = sec[sec['type']=='10-K'].iloc[0].get('edgarUrl')
# tenq = sec[sec['type']=='10-Q'].iloc[0].get('edgarUrl')


website = company_info.get("website")

token = 'pk_dXqZ_rdKRuKuAIDMxnuCEw'

if website:

    domain = urlparse(website).netloc.replace("www.", "")

    logo_url = f"https://img.logo.dev/{domain}?token={token}"
    

st.set_page_config(page_title=f'{title} - Riskmapp', layout='wide', page_icon=logo_url)


with maincol1:

    st.info(company_info.get('longBusinessSummary'))

col1, col2 = st.columns([1.25, 1])


ticker = title

@st.cache_data(ttl=3600)
def load_price_data(ticker):
    data = yf.Ticker(ticker).history(period="max")
    return make_tz_naive(data)

@st.cache_data(ttl=3600)
def compute_regime_data(ticker):

    data = load_price_data(ticker)

    price = data["Close"]
    returns = price.pct_change()

    st_volatility = returns.rolling(21).std() * np.sqrt(252)
    lt_volatility = returns.rolling(63).std() * np.sqrt(252)
    vol_ratio = st_volatility / lt_volatility
    vol_vol = st_volatility.rolling(21).std()

    momentum = macd_zs(price, fast=16, slow=64, signal=21, zs_window=252)

    adx = compute_adx(data)
    trend_strength = adx["ADX"]

    volume = data["Volume"]
    rvol = volume.rolling(21).mean() / volume.rolling(63).mean()

    MA200 = price.ewm(span=256, adjust=False).mean()
    distance_200MA = price / MA200 - 1

    drawdown = price / price.cummax() - 1

    hurst = hurst_exponent(data, power=6)



    X = zscore(pd.DataFrame({
        "momentum": momentum,
        "trend_strength": trend_strength,
        "vol_ratio": vol_ratio,
        "vol_of_vol": vol_vol,
        "rvol": rvol,
        "extension": distance_200MA,
        "drawdown": drawdown,
        "trend_slope": hurst,
    }), window=252).dropna()

    x_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=4, random_state=42)
    trading_factors = pca.fit_transform(x_scaled)
    x_pca = StandardScaler().fit_transform(trading_factors)

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    model.fit(x_pca)
    states = model.predict(x_pca)

    states_series = pd.Series(states, index=X.index, name="regime")

    forward_returns = returns.shift(-1)

    regime_df = pd.DataFrame({
        "returns": forward_returns.reindex(states_series.index),
        "regime": states_series
    }).dropna()

    state_stats = regime_df.groupby("regime")["returns"].agg(
        mean_return="mean",
        volatility="std",
        count="count"
    )

    state_stats["ann_return"] = state_stats["mean_return"] * 252
    state_stats["ann_vol"] = state_stats["volatility"] * np.sqrt(252)
    state_stats["ann_sharpe"] = state_stats["ann_return"] / state_stats["ann_vol"]

    best_regime = state_stats["ann_sharpe"].idxmax()
    worst_regime = state_stats["ann_sharpe"].idxmin()

    trend = trend_ind(price, trading_periods=126).dropna()

    return data, states_series, trend, best_regime, worst_regime, state_stats

def market_maker_pressure(ticker, tf='2y'):

    stock = yf.Ticker(ticker)
    df= stock.history(tf)
    avg_vol = int(df.Volume.rolling(60).mean().iloc[-1])

    expirations = stock.options

    all_chains = []

    for exp in expirations:
        opt_chain = stock.option_chain(exp)
        
        # Extract relevant fields
        calls = opt_chain.calls[["strike", "openInterest"]].copy()
        puts  = opt_chain.puts[["strike", "openInterest"]].copy()
        
        # Add metadata
        calls["type"] = "call"
        puts["type"]  = "put"
        
        calls["expirationDate"] = exp
        puts["expirationDate"]  = exp
        
        # Append to list
        all_chains.append(calls)
        all_chains.append(puts)

    # Combine into a single DataFrame
    options_df = pd.concat(all_chains, ignore_index=True)


    # Price range boundaries
    low_price  = df["Close"].min()
    high_price = df["Close"].max()

    # --- Filter strikes FIRST ---
    filtered = options_df[
        (options_df["strike"] >= low_price) &
        (options_df["strike"] <= high_price)
    ].copy()

    # --- Then aggregate by strike + type ---
    pivot = filtered.pivot_table(
        index="strike",
        columns="type",
        values="openInterest",
        aggfunc="sum",
        fill_value=0
    )

    pivot["netOI"] = pivot["call"] - pivot["put"]
    pivot['volumeImpactATM'] = round((pivot['netOI'] * 50) / avg_vol, 2) * 100

    buypres = pivot.nsmallest(5, 'netOI')
    sellpres =pivot.nlargest(5, 'netOI')

    net_oi_levels = pd.concat([buypres, sellpres], axis=0)

    # --- 1. Extract strike levels and scaling values ---
    levels = net_oi_levels.copy()

    # We use absolute value of volumeImpactATM for scaling width
    abs_impact = levels["volumeImpactATM"].abs()

    # Line width scaling (min 1px, max 10px for visual clarity)
    min_w, max_w = 1, 10
    impact_norm = (abs_impact - abs_impact.min()) / (abs_impact.max() - abs_impact.min() + 1e-9)
    line_widths = min_w + impact_norm * (max_w - min_w)

    # Line colors based on sign:
    # Positive volumeImpactATM → call wall → SELL pressure → resistance → GREEN
    # Negative volumeImpactATM → put wall → BUY pressure → support → RED
    line_colors = [
        "rgba(0,255,0,0.75)" if val > 0 else "rgba(255,0,0,0.75)"
        for val in levels["volumeImpactATM"]
    ]

    # --- 2. Build candlestick chart ---
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#00BFFF",
        decreasing_line_color="#FF6347",
    ))

    # --- 3. Add horizontal lines ---
    for (strike, row), lw, col in zip(levels.iterrows(), line_widths, line_colors):
        fig.add_hline(
            y=float(strike),
            line_width=float(lw),
            line_color=col,
            annotation_text=f" {round(row['volumeImpactATM'],1)}%", 
            annotation_position="right", 
            annotation_font_color="white"
            )

    # --- 4. Dark theme layout ---
        fig.update_layout(
        title="Options Market Dealer Pressure Map — Support/Resistance From Net OI",
        height=500,
        width=1100,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True
        ),
        margin=dict(l=40, r=40, t=60, b=20)
    )
    
    fig.update_xaxes(rangeslider_visible=False)

    return fig


@st.cache_data(ttl=90000)
def build_dealer_pressure_chart(ticker, tf="2y"):
    return market_maker_pressure(ticker, tf=tf)


def plot_candles_with_regimes_and_trend(
    ticker,
    states_series,
    trend,
    best_regime,
    worst_regime,
    title="Price with Regimes",
    tf="2y",
    log_scale=False
):
    data = yf.Ticker(ticker).history(period=tf)

    data = data[["Open", "High", "Low", "Close"]].copy()
    data = make_tz_naive(data)

    states_series = make_tz_naive(states_series.copy())
    trend = trend.copy()

    # Align everything
    df = pd.concat([data, states_series.rename("regime"), trend.rename("trend")], axis=1).dropna()

    if log_scale:
        for col in ["Open", "High", "Low", "Close", "trend"]:
            df[col] = np.log(df[col])

    fig = go.Figure()

    regime_colors = {
        best_regime: "#00FF66",   # full green
        worst_regime: "#FF3333",  # full red
    }

    default_color = "#4A90E2"

    # One candlestick trace per regime
    for regime in sorted(df["regime"].unique()):
        regime_df = df[df["regime"] == regime]

        color = regime_colors.get(regime, default_color)

        fig.add_trace(go.Candlestick(
            x=regime_df.index,
            open=regime_df["Open"],
            high=regime_df["High"],
            low=regime_df["Low"],
            close=regime_df["Close"],
            name=f"Regime {regime}",
            increasing=dict(
                line=dict(color=color),
                fillcolor=color
            ),
            decreasing=dict(
                line=dict(color=color),
                fillcolor=color
            )
        ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["trend"],
        mode="lines",
        name="Trend",
        line=dict(
            color="cyan",
            width=2,
            dash="dot"
        )
    ))

    scale_text = " (Log Scale)" if log_scale else ""

    fig.update_layout(
        title=title + scale_text,
        height=500,
        width=1100,
        paper_bgcolor="#0E0E0E",
        plot_bgcolor="#0E0E0E",
        font=dict(color="white"),
        xaxis=dict(
            title="Date",
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title="Log Price" if log_scale else "Price",
            gridcolor="rgba(255,255,255,0.15)",
            showgrid=True
        ),
        margin=dict(l=40, r=40, t=60, b=20)
    )

    return fig

@st.cache_data(ttl=86400)  # 1 day
def get_analyst_targets_data(ticker, tf="2y"):
    
    ticker = ticker.upper().strip()

    stock_obj = yf.Ticker(ticker)
    company_info = stock_obj.info

    stock = stock_obj.history(period=tf)
    stock = make_tz_naive(stock)

    targets_data = {
        "Current Price": company_info.get("currentPrice"),
        "Target High Price": company_info.get("targetHighPrice"),
        "Target Low Price": company_info.get("targetLowPrice"),
        "Target Mean Price": company_info.get("targetMeanPrice"),
        "Target Median Price": company_info.get("targetMedianPrice"),
    }

    targets_df = pd.DataFrame([targets_data])

    return stock, targets_data, targets_df

def plot_analyst_targets(ticker, tf="2y"):

    stock, targets_data, targets_df = get_analyst_targets_data(ticker, tf=tf)

    price = stock["Close"].squeeze()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=price.index,
        y=price.values,
        mode="lines",
        name="Stock Price",
        line=dict(width=2, color='cyan')
    ))

    colors = {
        "Current Price": "blue",
        "Target High Price": "#32CD32",
        "Target Low Price": "red",
        "Target Mean Price": "orange",
        "Target Median Price": "purple",
    }

    for label, value in targets_data.items():
        if pd.notna(value):
            fig.add_trace(go.Scatter(
                x=[price.index.min(), price.index.max()],
                y=[value, value],
                mode="lines",
                name=label,
                line=dict(dash="dash", color=colors.get(label, "gray")),
                hoverinfo="text",
                text=f"{label}: {value}"
            ))

    fig.update_layout(
        title=f"{ticker.upper()} Analyst Targets",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig, targets_df


@st.cache_data(ttl=86400)  # 24 hours
def get_analyst_grades_data(ticker):
    
    ticker = ticker.upper().strip()

    try:
        grades = yf.Ticker(ticker).get_upgrades_downgrades()

        if grades is None or grades.empty:
            return None

        # Filter last 12 months
        grades = grades.loc[grades.index > datetime.now() - timedelta(days=365)].copy()

        # Normalize actions
        grades["Action"] = grades["Action"].replace({
            "main": "Maintain",
            "reit": "Reiterated",
            "up": "Upgrade",
            "down": "Downgrade",
            "init": "Initiated Coverage"
        })

        final_grades = grades["Action"].value_counts().to_frame("Count")

        return final_grades

    except Exception:
        return None

def plot_analyst_grades(ticker, company_name):
    final_grades = get_analyst_grades_data(ticker)

    if final_grades is None or final_grades.empty:
        return None

    color_map = {
        "Upgrade": "#00B86B",
        "Downgrade": "#FF4C4C",
        "Maintain": "#1E90FF",
        "Reiterated": "#104E8B",
        "Initiated Coverage": "#FFA500"
    }

    colors = final_grades.index.map(color_map)

    fig = go.Figure(go.Bar(
        x=final_grades["Count"],
        y=final_grades.index,
        orientation="h",
        marker_color=colors
    ))

    fig.update_layout(
        title=f"Top Institutional Analyst Upgrades/Downgrades for {company_name}",
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_dark",
        xaxis_title="Number of Analysts",
        yaxis_title="",
        height=500
    )

    return fig


@st.cache_data(ttl=86400)  # 24 hours
def get_institutional_holders_data(ticker):
    ticker = ticker.upper().strip()

    try:
        ih = yf.Ticker(ticker).get_institutional_holders()

        # Validate
        if not isinstance(ih, pd.DataFrame) or ih.empty:
            return None

        if "pctHeld" not in ih.columns or "Holder" not in ih.columns:
            return None

        # Clean + sort
        ih = ih.copy()
        ih = ih.sort_values(by="pctHeld", ascending=True)

        return ih

    except Exception:
        return None
    
def plot_institutional_holders(ticker, company_name):
    ih = get_institutional_holders_data(ticker)

    if ih is None or ih.empty:
        return None

    fig = go.Figure(go.Bar(
        x=ih["pctHeld"] * 100,
        y=ih["Holder"],
        orientation="h",
        marker_color="#85BB65"
    ))

    fig.update_layout(
        title=f"Top Institutional Holders of {company_name}",
        yaxis=dict(categoryorder="total ascending"),
        template="plotly_dark",
        xaxis_title="Percentage Held",
        yaxis_title="",
        height=500
    )

    return fig

@st.cache_data(ttl=86400)  # 24 hours
def get_major_holders_data(ticker):
    ticker = ticker.upper().strip()

    try:
        mh = yf.Ticker(ticker).get_major_holders()

        if not isinstance(mh, pd.DataFrame) or mh.empty:
            return None

        mh = mh.reset_index()
        mh.columns = ["Breakdown", "Value"]

        rename_map = {
            "insidersPercentHeld": "Held by Insiders",
            "institutionsPercentHeld": "Held by Institutions",
            "institutionsCount": "Number of Institutions"
        }

        mh = mh[mh["Breakdown"].isin(rename_map.keys())].copy()
        mh["Breakdown"] = mh["Breakdown"].map(rename_map)

        pie_df = mh[mh["Breakdown"].isin([
            "Held by Insiders",
            "Held by Institutions"
        ])].copy()

        if pie_df.empty:
            return None

        labels = pie_df["Breakdown"].tolist()
        values = pie_df["Value"].tolist()

        remaining = 1 - sum(values)
        if remaining > 0:
            labels.append("Other")
            values.append(remaining)

        return labels, values

    except Exception:
        return None

def plot_major_holders(ticker, company_name):
    data = get_major_holders_data(ticker)

    if data is None:
        return None

    labels, values = data

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#009CDF', '#00BFA5', '#FFC107']),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title=f'Ownership Breakdown of {company_name}',
        template='plotly_dark',
        height=500,
        margin=dict(t=60, b=40, l=0, r=0),
        font=dict(color='white')
    )

    return fig

with col1:

    chart_choice = st.radio(
        "Select chart",
        ["Algo Chart", "Options Dealer Pressure", "Analysts Price Targets"],
        horizontal=True
    )

    data, states_series, trend, best_regime, worst_regime, state_stats = compute_regime_data(ticker)

    if chart_choice == "Algo Chart":
        fig = plot_candles_with_regimes_and_trend(
            ticker=ticker,
            states_series=states_series,
            trend=trend,
            best_regime=best_regime,
            worst_regime=worst_regime,
            title="Trade & Trend Indicator"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Options Dealer Pressure":
        fig = build_dealer_pressure_chart(ticker, tf="2y")
        st.plotly_chart(fig, use_container_width=True)

    else:
        
        fig, analysts_targets = plot_analyst_targets(ticker, tf="2y")
        st.plotly_chart(fig, use_container_width=True)

company_name = yf.Ticker(ticker).get_info()['longName']

co1a, col1b, col1c = st.columns(3)

with co1a:
    fig_grades = plot_analyst_grades(ticker, company_name)

    if fig_grades:
        st.plotly_chart(
            fig_grades,
            use_container_width=True,
            key=f"grades_chart_{ticker}_col7"
        )
    else:
        st.write("This data is not available.")

with col1b:

    fig_holders = plot_institutional_holders(ticker, company_name)

    if fig_holders:
        st.plotly_chart(
            fig_holders,
            use_container_width=True,
            key=f"institutional_holders_chart_{ticker}"
        )
    else:
        st.write("Institutional holder data not available.")

with col1c:

    fig_pie = plot_major_holders(ticker, company_name)

    if fig_pie:
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.write("Ownership data not available.")

def safe_round(value, multiplier=1, decimals=2, suffix=''):
    if value is None:
        return 'N/A'
    try:
        return f"{round(value * multiplier, decimals)}{suffix}"
    except:
        return 'N/A'

@st.cache_data(ttl=86400)  # 24 hours
def get_quick_metrics(ticker):
    
    ticker = ticker.upper().strip()

    try:
        company_stock = yf.Ticker(ticker)

        company_info = company_stock.info or {}
        company_name = company_info.get("shortName", ticker)

        ev = company_info.get("enterpriseValue", np.nan)

        q_income = company_stock.quarterly_income_stmt

        if isinstance(q_income, pd.DataFrame) and "EBIT" in q_income.index:
            ebit = q_income.loc["EBIT"].iloc[0]
        else:
            ebit = np.nan

        ebit_yield = ebit / ev if pd.notna(ebit) and pd.notna(ev) and ev else np.nan

        free_cash_flow = company_info.get("freeCashflow", np.nan)
        fcf_yield = (
            free_cash_flow / ev
            if pd.notna(free_cash_flow) and pd.notna(ev) and ev
            else np.nan
        )

        market_cap = company_info.get("marketCap")
        trailing_pe = company_info.get("trailingPE")
        forward_pe = company_info.get("forwardPE")

        earnings_growth_expectations = (
            (trailing_pe / forward_pe) - 1
            if trailing_pe and forward_pe
            else np.nan
        )

        earnings_date = pd.to_datetime(
            company_info.get('earningsTimestamp'),
            unit='s'
        ).date()

        today = pd.Timestamp.now().date()

        days_to_earnings = (earnings_date - today).days


        metrics = {
            "Market Cap": f"${market_cap / 1e6:,.0f}M" if market_cap else "N/A",
            "Trailing P/E": safe_round(trailing_pe),
            "Forward P/E": safe_round(forward_pe),
            "Earnings Growth Expectations": safe_round(
                earnings_growth_expectations, 100, 2, "%"
            ),
            "EBIT Yield": safe_round(ebit_yield, 100, 2, "%"),
            "FCF Yield": safe_round(fcf_yield, 100, 2, "%"),
            "Revenue Growth": safe_round(company_info.get("revenueGrowth"), 100, 2, "%"),
            "Gross Margins":safe_round(company_info.get("grossMargins"), 100, 2, "%"),
            "Operating Margins": safe_round(company_info.get("operatingMargins"), 100, 2, "%"),
            "Return on Equity": safe_round(company_info.get("returnOnEquity"), 100, 2, "%"),
            "Return on Assets": safe_round(company_info.get("returnOnAssets"), 100, 2, "%"),
            "Debt to Equity": safe_round(company_info.get("debtToEquity"), 1, 2, "%"),
            "Current Ratio": safe_round(company_info.get("currentRatio")),
            "Quick Ratio": safe_round(company_info.get("quickRatio")),
            "Total Cash Per Share": safe_round(company_info.get("totalCashPerShare")),
            "Price to Book": safe_round(company_info.get("priceToBook")),
            "Next Earnings Date": earnings_date,
            "Days to Earnings": days_to_earnings
        }

        quick_metrics = pd.DataFrame.from_dict(
            metrics,
            orient="index",
            columns=[company_name]
        )

        return quick_metrics

    except Exception:
        return pd.DataFrame()
    


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

path = BASE_DIR / "us_stock_market_watchlist2 2025-11.xlsx"

@st.cache_data
def open_screener(path):
    return pd.read_excel(path, engine="openpyxl")

screener = open_screener(path)

# Clean ticker columns
ticker = ticker.upper().strip()
screener["Ticker"] = screener["Ticker"].astype(str).str.upper().str.strip()

match = screener.loc[
    screener["Ticker"] == ticker,
    ["Company Size", "Industry"]
]

if not match.empty:

    company_size, industry = match.iloc[0]

    comps = screener[
        (screener["Company Size"] == company_size) &
        (screener["Industry"] == industry)
    ].copy()

    comps_cols = [
        "Ticker", "Company Name", "NAICS Industry", "Country", "Company Size",
        "Current P/E", "Forward P/E", "Current PEG Ratio",
        "EPS Growth Expectation", "Short Ratio", "(%)Float Short",
        "Growth Score", "Efficiency Score", "Pain Score", "Venture Score",
        "Fragility Score", "Ex-Growth Score", "Value Score"
    ]

    available_cols = [col for col in comps_cols if col in comps.columns]

    comparative_analysis = comps[available_cols]

    st.title("Sector Comparative Analysis")

    st.dataframe(
        comparative_analysis,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(f"{ticker} not found in screener table.")

def to_float(value):
    if value in [None, "N/A"]:
        return np.nan
    if isinstance(value, str):
        value = value.replace("%", "").replace(",", "").strip()
    return float(value)

quick_metrics = get_quick_metrics(ticker)


company_pe = to_float(quick_metrics.loc["Trailing P/E"].iloc[0])
company_forward_pe = to_float(quick_metrics.loc["Forward P/E"].iloc[0])
company_forwardeps = to_float((company_pe/company_forward_pe-1)*100)
company_forwardeps = round(company_forwardeps, 2) if pd.notna(company_forwardeps) else np.nan

avg_currpe = comparative_analysis.loc[
    comparative_analysis["Current P/E"] > 0,
    "Current P/E"
].mean()

avg_fwdpe = comparative_analysis.loc[
    comparative_analysis["Forward P/E"] > 0,
    "Forward P/E"
].mean()

avg_currpe = round(avg_currpe, 2) if pd.notna(avg_currpe) else np.nan
avg_fwdpe = round(avg_fwdpe, 2) if pd.notna(avg_fwdpe) else np.nan

avg_epsexp = (avg_currpe / avg_fwdpe - 1)*100

avg_epsexp=round(avg_epsexp, 2) if pd.notna(avg_epsexp) else np.nan

compcol1, compcol2, compcol3 = st.columns(3)

with compcol1:
    fig_pe = go.Figure()

    fig_pe.add_trace(go.Bar(
        x=["Current P/E", "Forward P/E"],
        y=[company_pe, company_forward_pe],
        name="Company",
        marker_color="#009CDF",
        text=[company_pe, company_forward_pe],
        textposition="outside"
    ))

    fig_pe.add_trace(go.Bar(
        x=["Current P/E", "Forward P/E"],
        y=[avg_currpe, avg_fwdpe],
        name="Sector Avg",
        marker_color="#85BB65",
        text=[avg_currpe, avg_fwdpe],
        textposition="outside"
    ))

    fig_pe.update_layout(
        title="Company vs Sector Valuation",
        template="plotly_dark",
        barmode="group",
        height=380,
        yaxis_title="P/E",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig_pe, use_container_width=True)


with compcol2:
    fig_eps = go.Figure()

    fig_eps.add_trace(go.Bar(
        x=["EPS Growth Expectation"],
        y=[company_forwardeps],
        name="Company",
        marker_color="#009CDF",
        text=[company_forwardeps],
        textposition="outside"
    ))

    fig_eps.add_trace(go.Bar(
        x=["EPS Growth Expectation"],
        y=[avg_epsexp],
        name="Sector Avg",
        marker_color="#85BB65",
        text=[avg_epsexp],
        textposition="outside"
    ))

    fig_eps.update_layout(
        title="EPS Growth Expectation",
        template="plotly_dark",
        barmode="group",
        height=380,
        yaxis_title="%",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig_eps, use_container_width=True)


radar_metrics = [
    "Growth Score",
    "Efficiency Score",
    "Pain Score",
    "Venture Score",
    "Fragility Score",
    "Ex-Growth Score",
    "Value Score",
]

company_row = comps.loc[
    comps["Ticker"] == ticker,
    radar_metrics
]

if company_row.empty:
    company_scores = pd.Series(
        [np.nan] * len(radar_metrics),
        index=radar_metrics
    )
else:
    company_scores = company_row.iloc[0]

sector_scores = comparative_analysis[radar_metrics].mean(skipna=True)

radar_df = pd.DataFrame({
    "Metric": radar_metrics,
    "Company": company_scores.values,
    "Sector Avg": sector_scores.values,
})

radar_df[["Company", "Sector Avg"]] = radar_df[["Company", "Sector Avg"]].apply(
    pd.to_numeric,
    errors="coerce"
)

# Drop metrics where both company and sector are NaN
radar_df = radar_df.dropna(subset=["Company", "Sector Avg"], how="all")

# Fill remaining NaNs with 0 so the radar chart does not break
radar_df[["Company", "Sector Avg"]] = radar_df[["Company", "Sector Avg"]].fillna(0)

with compcol3:
    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["Company"],
        theta=radar_df["Metric"],
        fill="toself",
        name="Company",
        line=dict(color="#009CDF")
    ))

    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["Sector Avg"],
        theta=radar_df["Metric"],
        fill="toself",
        name="Sector Avg",
        line=dict(color="#85BB65")
    ))

    fig_radar.update_layout(
        title="Company vs Sector Score Profile",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=60, b=30),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[
                    0,
                    max(
                        radar_df["Company"].max(),
                        radar_df["Sector Avg"].max(),
                        1
                    )
                ]
            )
        ),
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig_radar, use_container_width=True)

company_info = get_company_info(ticker)

def safe_get(info, key, multiplier=None):

    value = info.get(key)

    if pd.notna(value):

        if multiplier is not None:
            return value * multiplier

        return value

    return np.nan


@st.cache_data(ttl=86400)
def build_quant_table(ticker, start="2017-01-01"):

    
    ticker = ticker.upper().strip()
    company = yf.Ticker(ticker)
    company_info = company.info

    balance_sheet = company.balance_sheet
    income_statement = company.get_income_stmt()
    financials = company.get_financials()

    q_income = company.quarterly_income_stmt
    
    price = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False
    )["Close"].squeeze()

    # Financial statement dates
    dates = pd.to_datetime(income_statement.columns)
    
    # Align price to next available trading day
    price_on_dates = price.reindex(dates, method="bfill")
    
    def get_row(df, possible_names, columns=None):
        
        if columns is None:

            columns = df.columns

        for name in possible_names:
            if name in df.index:
                return df.loc[name].reindex(columns)

        return pd.Series(index=columns, dtype=float)

    # Core financial items
    shares = get_row(balance_sheet, [
        "Ordinary Shares Number",
        "Share Issued",
        "Common Stock Shares Outstanding"
    ])

    revenue = get_row(income_statement, [
        "TotalRevenue",
        "OperatingRevenue"
    ])

    net_income = get_row(income_statement, [
        "NetIncome",
        "NetIncomeCommonStockholders",
        "NetIncomeContinuousOperations"
    ])

    eps = get_row(financials, [
        "DilutedEPS",
        "BasicEPS"
    ])

    earnings_growth = eps.pct_change(-1)

    # Quarterly rows
    q_revenue = safe_get(company_info, 'totalRevenue')
    q_net_income = get_row(q_income, [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Continuous Operations"
    ], q_income.columns)

    q_eps = safe_get(company_info, 'epsTrailingTwelveMonths')

    q_shares = safe_get(company_info, 'sharesOutstanding')
    # Make sure quarters go newest -> oldest as Yahoo usually gives them

    q_net_income = q_net_income.sort_index(ascending=False)


    # TTM values = latest 4 quarters
    ttm_revenue = q_revenue
    ttm_net_income = q_net_income.iloc[:4].sum()
    ttm_eps = q_eps

    latest_price = safe_get(company_info, 'currentPrice')

    ttm_market_cap = safe_get(company_info, 'marketCap')
    ttm_sales_multiple = ttm_market_cap / ttm_revenue
    ttm_pe = safe_get(company_info, 'trailingPE')

    # Fallback EPS if unavailable
    eps = eps.where(eps.notna(), np.nan)

    # Calculations
    market_cap = shares * price_on_dates
    sales_multiple = market_cap / revenue
    pe = price_on_dates / eps

    earnings_growth = eps.pct_change(periods=-1) * 100
    sales_growth = revenue.pct_change(periods=-1) * 100

    ttm_eg = safe_get(company_info, 'earningsGrowth') * 100
    ttm_rvg = safe_get(company_info, 'revenueGrowth') * 100


    peg = pe / earnings_growth
    ttm_peg = safe_get(company_info, 'trailingPegRatio')

    # Build raw table
    table = pd.DataFrame({
        "Stock Price $": price_on_dates,
        "Market Cap $M": market_cap / 1e6,
        "EPS": eps,
        "Earnings Growth %": earnings_growth,
        "Price/Earnings": pe,
        "PEG": peg,
        "Sales $M": revenue / 1e6,
        "Sales Growth %": sales_growth,
        "Sales Multiple": sales_multiple,
        "Net Income (GAAP) $": net_income / 1e6,
    }).T

    table["TTM"] = pd.Series({
    "Stock Price $": latest_price,
    "Market Cap $M": ttm_market_cap / 1e6,
    "EPS": ttm_eps,
    "Earnings Growth %": table.loc['EPS'].pct_change(1).iloc[-1] * 100,
    "Price/Earnings": ttm_pe,
    "PEG": ttm_peg,
    "Sales $M": ttm_revenue / 1e6,
    "Sales Growth %": table.loc['Sales $M'].pct_change(1).iloc[-1] * 100,
    "Sales Multiple": ttm_sales_multiple,
    "Net Income (GAAP) $": ttm_net_income / 1e6,
    })

    
    # Rename columns to years
    table.columns = [
    str(d.year) if isinstance(d, pd.Timestamp) else d
    for d in table.columns
    ]
    
    # Sort columns oldest to newest
    table = table.reindex(sorted(table.columns), axis=1)

    fys= safe_get(company_info, 'nextFiscalYearEnd')

    y0 = pd.to_datetime(
        fys,
        unit='s'
        ).year
    y1 = y0 + 1

    rev_estimates = (
        yf.Ticker(ticker)
        .revenue_estimate
        .T[['0y', '+1y']]
        .rename(columns={"0y": y0, "+1y": y1})
        .loc[['avg', 'growth']]
        .rename(index={
            'avg': 'Sales $M',
            'growth': 'Sales Growth %'
        })
    )

    eps_estimates = (
        yf.Ticker(ticker)
        .earnings_estimate
        .T[['0y', '+1y']]
        .rename(columns={"0y": y0, "+1y": y1})
        .loc[['avg', 'growth']]
        .rename(index={
            'avg': 'EPS',
            'growth': 'Earnings Growth %'
        })
    )

    pe_base = (
        company_info.get('trailingPE')
        if pd.notna(company_info.get('trailingPE'))
        else (
            company_info.get('priceEpsCurrentYear')
            if pd.notna(company_info.get('priceEpsCurrentYear'))
            else np.nan
        )
        )

    price_targets = pe_base  * eps_estimates.loc['EPS']
    pe_targets = safe_get(company_info, 'currentPrice') / eps_estimates.loc['EPS']

    mcap_targets = safe_get(company_info, 'sharesOutstanding') * price_targets
    ps_targets = mcap_targets / rev_estimates.loc['Sales $M']
    peg_targets = pe_targets / (eps_estimates.loc['Earnings Growth %'] * 100)

    sales_target = rev_estimates.loc['Sales $M']
    rvg_target = rev_estimates.loc['Sales Growth %']

    eps_target = eps_estimates.loc['EPS']
    eg_target = eps_estimates.loc['Earnings Growth %']

    table.loc['Sales $M', sales_target.index] = sales_target / 1e6
    table.loc['Sales Growth %', rvg_target.index] = rvg_target * 100
    table.loc['Earnings Growth %', eg_target.index] = eg_target * 100
    table.loc['EPS', eps_target.index] = eps_target
    table.loc['Stock Price $', price_targets.index] = price_targets
    table.loc['Price/Earnings', pe_targets.index] = pe_targets
    table.loc['Market Cap $M', mcap_targets.index] = mcap_targets / 1e6
    table.loc['Sales Multiple', ps_targets.index] = ps_targets 
    table.loc['PEG', peg_targets.index] = peg_targets


    return table

def format_quant_table(df):
    formatted = df.copy().astype(object)

    for col in formatted.columns:
        formatted.loc["Stock Price $", col] = f"{df.loc['Stock Price $', col]:,.2f}"
        formatted.loc["Market Cap $M", col] = f"{df.loc['Market Cap $M', col]:,.0f}"
        formatted.loc["EPS", col] = f"{df.loc['EPS', col]:,.2f}"
        formatted.loc["Earnings Growth %", col] = f"{df.loc['Earnings Growth %', col]:,.1f}%"
        formatted.loc["Price/Earnings", col] = f"{df.loc['Price/Earnings', col]:,.2f}"
        formatted.loc["PEG", col] = f"{df.loc['PEG', col]:,.2f}"
        formatted.loc["Sales $M", col] = f"{df.loc['Sales $M', col]:,.0f}"
        formatted.loc["Sales Growth %", col] = f"{df.loc['Sales Growth %', col]:,.1f}%"
        formatted.loc["Sales Multiple", col] = f"{df.loc['Sales Multiple', col]:,.2f}"
        formatted.loc["Net Income (GAAP) $", col] = f"{df.loc['Net Income (GAAP) $', col]:,.0f}"

    formatted = formatted.replace(["nan", "NaN"], "N/A")
    return formatted

quant_table = build_quant_table(ticker)
display_table = format_quant_table(quant_table.iloc[:, 1:])

colq1, colq2 = st.columns([1.5, 1])

with colq1:


    fy0 = pd.to_datetime(yf.Ticker(ticker).info.get('nextFiscalYearEnd'),
                     unit='s').date()

    st.write(f'Next Fiscal Year End: {fy0}')
    st.dataframe(display_table)
    

company_stats = screener.loc[screener['Ticker']==ticker].T



@st.cache_data(ttl=86400)
def get_earnings_dates_data(ticker, n=5):
    ticker = ticker.upper().strip()

    try:
        company = yf.Ticker(ticker)

        next_eps_raw = company.info.get("earningsTimestamp")
        
        if next_eps_raw is None:
            return pd.DataFrame(), "Next earnings timestamp not available."

        next_eps = pd.to_datetime(next_eps_raw, unit="s").date()
        next_eps_plus_1q = (next_eps + pd.DateOffset(months=3)).date()

        ed = company.get_earnings_dates()

        if not isinstance(ed, pd.DataFrame) or ed.empty:
            return pd.DataFrame(), "Earnings date data not available."

        ed = (
            ed.sort_index(ascending=False)
              .head(n)
              .reset_index()
        )

        if "Earnings Date" not in ed.columns:
            ed = ed.rename(columns={"index": "Earnings Date"})

        eq_est = (
            company.get_eps_trend()
            .T[["0q", "+1q"]]
            .rename(columns={"0q": next_eps, "+1q": next_eps_plus_1q})
            .rename(index={"current": "EPS Estimate"})
            .loc["EPS Estimate"]
        )

        future_ed = pd.DataFrame({
            "Earnings Date": [next_eps, next_eps_plus_1q],
            "EPS Estimate": eq_est.values,
            "Reported EPS": [np.nan, np.nan],
            "Surprise(%)": [np.nan, np.nan],
        })

        ed = pd.concat([ed, future_ed], ignore_index=True)

        ed["Earnings Date"] = pd.to_datetime(
            ed["Earnings Date"],
            utc=True,
            errors="coerce"
        )

        ed = ed.dropna(subset=["Earnings Date"])
        ed = ed.sort_values("Earnings Date", ascending=False)

        ed["Earnings Date"] = ed["Earnings Date"].dt.strftime("%Y-%m-%d")

        needed_cols = ["EPS Estimate", "Reported EPS"]
        missing = [c for c in needed_cols if c not in ed.columns]

        if missing:
            return pd.DataFrame(), f"Earnings fields not available: {', '.join(missing)}"

        df = ed.dropna(subset=["EPS Estimate"])

        if df.empty:
            return pd.DataFrame(), "No EPS estimate data to plot."

        return df, None

    except Exception as e:
        return pd.DataFrame(), f"Error fetching earnings dates: {str(e)}"
    
def plot_earnings_dates(ticker, company_name):
    df, error = get_earnings_dates_data(ticker)

    if error:
        return None, error

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Earnings Date"],
        y=df["EPS Estimate"],
        mode="markers",
        name="EPS Estimate",
        marker=dict(
            symbol="circle-open",
            size=18,
            color="#87CEFA"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=df["Earnings Date"],
        y=df["Reported EPS"],
        mode="markers",
        name="Reported EPS",
        marker=dict(
            symbol="x",
            size=12,
            color="#00B86B"
        ),
    ))

    fig.update_layout(
        title=f"{company_name} EPS Estimates vs Reported EPS",
        xaxis_title="Earnings Date",
        yaxis_title="EPS",
        template="plotly_dark",
        font=dict(color="white"),
        height=400,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig, None

with colq2:
    earnings_fig, earnings_error = plot_earnings_dates(ticker, company_name)

    if earnings_fig:
        st.plotly_chart(
            earnings_fig,
            use_container_width=True,
            key=f"earnings_scatter_{ticker}"
        )
    else:
        st.write(earnings_error)


def bs_price(S, K, T, r, sigma, q=0.0, option_type="Call"):
    """
    Black–Scholes price with continuous dividend yield q.
    option_type: "Call" / "Put" / "C" / "P"
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan
    
    option_type = option_type.upper()

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)

    if option_type in ["CALL", "C"]:
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    elif option_type in ["PUT", "P"]:
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'Call' or 'Put'")

def implied_vol_bs_brent_q(S, K, T, r, market_price,
                           option_type="Call",
                           q=0.0,
                           sigma_low=1e-4,
                           sigma_high=5.0,
                           tol=1e-6,
                           max_iter=100):
    """
    Implied volatility via Brent's method for BS with continuous dividend yield q.
    
    - S, K, T, r as usual
    - market_price: option market price (mid ideally)
    - option_type: "Call" / "Put" / "C" / "P"
    - q: continuous dividend yield
    - sigma_low, sigma_high: vol search bounds
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return np.nan

    option_type = option_type.upper()

    # Objective: model_price(sigma) - market_price = 0
    def f(sigma):
        return bs_price(S, K, T, r, sigma, q=q, option_type=option_type) - market_price

    f_low = f(sigma_low)
    f_high = f(sigma_high)

    # If bracket doesn't contain a root, try expanding once
    if f_low * f_high > 0:
        sigma_high2 = 10.0
        f_high2 = f(sigma_high2)

        if f_low * f_high2 > 0:
            # No root in [sigma_low, sigma_high2]
            if abs(f_low) < tol:
                return sigma_low
            if abs(f_high2) < tol:
                return sigma_high2
            return np.nan
        else:
            sigma_high = sigma_high2
            f_high = f_high2

    try:
        implied_vol = brentq(f, sigma_low, sigma_high, xtol=tol, maxiter=max_iter)
        return implied_vol
    except ValueError:
        return np.nan

def get_option_mid(row):
    bid = row.get("bid", np.nan)
    ask = row.get("ask", np.nan)
    last = row.get("lastPrice", np.nan)

    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
        return 0.5 * (bid + ask)

    if pd.notna(last) and last > 0:
        return last

    return np.nan

def atm_iv_for_expiration(stock, expiration, spot, r=0.045, q=0.0):
    chain = stock.option_chain(expiration)

    exp_dt = pd.to_datetime(expiration)
    today = pd.Timestamp.today().normalize()

    dte = (exp_dt - today).days
    T = max(dte / 365, 1 / 365)

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    if calls.empty or puts.empty:
        return np.nan, dte

    # Find nearest ATM strike
    calls["distance"] = (calls["strike"] - spot).abs()
    puts["distance"] = (puts["strike"] - spot).abs()

    atm_call = calls.loc[calls["distance"].idxmin()]
    atm_put = puts.loc[puts["distance"].idxmin()]

    call_mid = get_option_mid(atm_call)
    put_mid = get_option_mid(atm_put)

    call_iv = implied_vol_bs_brent_q(
        S=spot,
        K=atm_call["strike"],
        T=T,
        r=r,
        market_price=call_mid,
        option_type="Call",
        q=q
    )

    put_iv = implied_vol_bs_brent_q(
        S=spot,
        K=atm_put["strike"],
        T=T,
        r=r,
        market_price=put_mid,
        option_type="Put",
        q=q
    )

    atm_iv = np.nanmean([call_iv, put_iv])

    return atm_iv, dte

@st.cache_data(ttl=86400)
def get_30d_atm_iv(ticker, r=0.045):

    ticker = ticker.upper().strip()
    stock = yf.Ticker(ticker)

    info = stock.info or {}

    spot = info.get("currentPrice")

    if spot is None:
        hist = stock.history(period="5d")
        if hist.empty:
            return np.nan, pd.DataFrame()
        spot = hist["Close"].iloc[-1]

    q = info.get("dividendYield") or 0.0

    rows = []

    for exp in stock.options:
        try:
            atm_iv, dte = atm_iv_for_expiration(
                stock=stock,
                expiration=exp,
                spot=spot,
                r=r,
                q=q
            )

            if np.isfinite(atm_iv) and dte > 0:
                rows.append({
                    "expiration": pd.to_datetime(exp),
                    "DTE": dte,
                    "ATM IV": atm_iv
                })

        except Exception:
            continue

    iv_curve = pd.DataFrame(rows)

    if iv_curve.empty:
        return np.nan, pd.DataFrame(columns=["expiration", "DTE", "ATM IV"])

    iv_curve = iv_curve.sort_values("DTE")

    if iv_curve.empty:
        return np.nan, iv_curve

    # Need at least two expirations around 30D for interpolation
    iv_30d = np.interp(
        30,
        iv_curve["DTE"],
        iv_curve["ATM IV"]
    )

    return iv_30d, iv_curve
    
@st.cache_data(ttl=86400)
def get_stress_metrics(ticker):

    info = yf.Ticker(ticker).info

    price = yf.download(ticker, start='2020-01-01', auto_adjust=True)['Close'].squeeze()
    returns = price.pct_change()
    ann_vol = returns.rolling(504).std() * np.sqrt(252)

    curr_vol = ann_vol.iloc[-1] * 100

    days_to_cover = info.get('shortRatio')
    short_pct = info.get('shortPercentOfFloat')
    if short_pct is not None:
        float_short = short_pct * 100
    else:
        float_short = None
    
    analyst_opinion = info.get('recommendationKey')
    analyst_opinion = analyst_opinion.replace('_', " ").capitalize()
    dividend_yield = info.get('dividendYield')
    beta = info.get('beta')

    iv_30d, iv_curve = get_30d_atm_iv(ticker)
    
    stress_metrics = {
        "Days to Cover": days_to_cover,
        "Short Float" : float_short,
        "Dividend Yield":dividend_yield,
        "Analyst Recommendation" : analyst_opinion,
        "Beta" : beta,
        "Annualized Realized Volatility" : curr_vol,
        "30D Implied Volatility": iv_30d * 100
    }

    return stress_metrics

stress_metrics = get_stress_metrics(ticker)

def format_kpi(label, value):
    if value is None:
        return "N/A"

    if label in ["Short Float", "Dividend Yield",
                 "Annualized Realized Volatility",
                 "30D Implied Volatility"]:
        return f"{value:.2f}%"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value).title()

def kpi_card(label, value):
    st.markdown(f"""
        <div style="
            background-color:#111827;
            padding:14px;
            border-radius:10px;
            margin-bottom:10px;
            box-shadow:0px 2px 6px rgba(0,0,0,0.3);
        ">
            <div style="font-size:13px; color:#9CA3AF;">
                {label}
            </div>
            <div style="font-size:26px; font-weight:600; color:white;">
                {value}
            </div>
        </div>
    """, unsafe_allow_html=True)


with col2:

    col2a, col2b = st.columns([1.75, 1])

    with col2a:    


        if not quick_metrics.empty:
            st.dataframe(quick_metrics, use_container_width=True, height=700)
        else:
            st.write("Quick fundamental metrics not available.")
    
    with col2b:

        for label, value in stress_metrics.items():
            formatted_value = format_kpi(label, value)
            kpi_card(label, formatted_value)


reg_BASE_DIR = Path(__file__).resolve().parent
MARKET_REGIMES_PATH = reg_BASE_DIR / "market_regimes.csv"

market_regimes = pd.read_csv(
    MARKET_REGIMES_PATH,
    index_col=0,
    parse_dates=True,
)

def rolling_sortino_ratio(price, window=252, periods_per_year=252):

    returns = price.pct_change()

    rolling_mean = (
        returns.rolling(window).mean()
        * periods_per_year
    )

    downside = returns.clip(upper=0)

    rolling_downside_std = (
        downside.rolling(window).std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    return rolling_mean / rolling_downside_std

def rolling_tail_ratio(df, window=252, upper_q=0.90, lower_q=0.10):
    upper = df.rolling(window).quantile(upper_q)
    lower = df.rolling(window).quantile(lower_q).abs()
    tail_ratio = upper / lower
    zs_tail = (tail_ratio - tail_ratio.rolling(756).mean()) / tail_ratio.rolling(756).std()
    return zs_tail

price = load_price_data(ticker)['Close']
returns = price.pct_change()
sortino = rolling_sortino_ratio(price)
tail_ratio = rolling_tail_ratio(price)

performance = (0.7 * sortino) + (0.3 * tail_ratio)

market_regimes = market_regimes.copy()

market_regimes.index = pd.to_datetime(
    market_regimes.index,
    errors="coerce"
)

market_regimes = market_regimes[market_regimes.index.notna()]
market_regimes = market_regimes.sort_index()

price.index = pd.to_datetime(price.index)
price = price.sort_index()

regimes_daily = market_regimes.reindex(price.index, method="ffill").squeeze()

performance_df = pd.DataFrame({
    "performance": performance,
    "regime": regimes_daily
}).dropna()

performance_df["regime"] = performance_df["regime"].astype(int).astype(str)
performance_df["performance"] = pd.to_numeric(
    performance_df["performance"],
    errors="coerce"
)

performance_regime = (
    performance_df
    .groupby("regime", as_index=False)["performance"]
    .mean()
    .sort_values("regime")
)

current_regime = market_regimes.iloc[-1].iloc[0]
current_performance = performance.iloc[-1]

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:

    fig_perf_regime = go.Figure()

    fig_perf_regime.add_trace(go.Bar(
        x=performance_regime["regime"],
        y=performance_regime["performance"],
        text=performance_regime["performance"].round(2),
        textposition="outside",
        marker_color="cyan",
    ))

    fig_perf_regime.add_hline(
    y=performance_df["performance"].mean(),
    line_dash="dot",
    line_color="white",
    annotation_text="Avg",
    annotation_position="top right"
    )

    fig_perf_regime.update_layout(
        title="Average Performance by Market Regime",
        xaxis_title="Market Regime",
        yaxis_title="Performance Score",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        bargap=0.35,
        xaxis=dict(type="category")
    )

    st.plotly_chart(fig_perf_regime, use_container_width=True)

    box1, box2=st.columns(2)

    with box1:

        st.markdown(f'Current Regime: {current_regime}')

    with box2:

        st.markdown(f'Current performance: {round(current_performance,2)}')

@st.cache_data(ttl=86400)
def get_factors(returns):
    
    factors = ['MTUM', 'QUAL', 'SIZE', 'USMV', 'VLUE', 'VUG', 'SPY']

    factors_rets = (yf.download(factors, start='2000-01-01', 
                                auto_adjust=True)
                                ['Close']
                                .squeeze()).pct_change()
    
    correlation = returns.ewm(span=504, adjust=False).corr(factors_rets).iloc[-1]

    return correlation

with mcol2:

    factor_corr = get_factors(returns)

    fig_factors = go.Figure()

    fig_factors.add_trace(go.Bar(
        x=factor_corr.index,
        y=factor_corr.values,
        text=np.round(factor_corr.values, 2),
        textposition="outside",
        marker_color=[
            "#00B86B" if val > 0 else "#FF4C4C"
            for val in factor_corr.values
        ]
    ))

    fig_factors.add_hline(
        y=1,
        line_dash="dot",
        line_color="gray",
        opacity=0.8
    )

    fig_factors.update_layout(
        title="Factor Correlation Exposure",
        xaxis_title="Factor ETF",
        yaxis_title="Correlation",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        bargap=0.3
    )

    st.plotly_chart(fig_factors, use_container_width=True)

with mcol3:

    @st.cache_data(ttl=86400)
    def plot_peer_beta_chart(ticker, comps, returns):

        comps_tickers = [t for t in comps.Ticker.dropna().unique() if t != ticker]

        if len(comps_tickers) == 0:
            return None, "No comparable stocks available."

        comps_prices = yf.download(
            comps_tickers,
            start="2000-01-01",
            auto_adjust=True,
            progress=False
        )["Close"]

        if comps_prices.empty:
            return None, "Comparable stock price data not available."

        comps_rets = comps_prices.pct_change()

        covs = comps_rets.apply(lambda x: returns.rolling(504).cov(x))
        vars_ = comps_rets.rolling(504).var()

        betas = (
            (covs / vars_)
            .iloc[-1]
            .dropna()
            .sort_values(ascending=True)
        )

        if betas.empty:
            return None, "Peer beta chart not available."

        fig_beta = go.Figure()

        fig_beta.add_trace(go.Bar(
            x=betas.values,
            y=betas.index,
            orientation="h",
            marker_color="#FF9F1C",  # Bloomberg orange
            text=np.round(betas.values, 2),
            textposition="outside"
        ))

        fig_beta.add_vline(
            x=1,
            line_dash="dot",
            line_color="white",
        )

        fig_beta.update_layout(
            title=f"{ticker} Peer Beta Exposure",
            xaxis_title="Beta",
            yaxis_title="Comparable Stocks",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=40, t=60, b=40),
            bargap=0.35
        )

        return fig_beta, None
    
    fig_beta, beta_error = plot_peer_beta_chart(ticker, comps, returns)

    if fig_beta:
        st.plotly_chart(fig_beta, use_container_width=True)
    else:
        st.write(beta_error)
