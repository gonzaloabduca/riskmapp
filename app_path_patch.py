"""Copy these patterns into app.py; this is not a standalone application."""

import os
from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SCREENER_PATH = Path(
    os.getenv("SCREENER_PATH", DATA_DIR / "us_stock_market_watchlist.xlsx")
)
MARKET_REGIMES_PATH = Path(
    os.getenv("MARKET_REGIMES_PATH", DATA_DIR / "market_regimes.csv")
)
LOGO_DEV_TOKEN = os.getenv("LOGO_DEV_TOKEN", "")

# Keep exactly one set_page_config call, near the top of app.py.
st.set_page_config(page_title="Riskmapp", layout="wide")

# Replace the two absolute Windows paths with:
# screener = open_screener(SCREENER_PATH)
# market_regimes = pd.read_csv(MARKET_REGIMES_PATH, index_col=0)
#
# Build a logo only when both website and secret exist:
# if website and LOGO_DEV_TOKEN:
#     domain = urlparse(website).netloc.replace("www.", "")
#     logo_url = f"https://img.logo.dev/{domain}?token={LOGO_DEV_TOKEN}"
#     st.image(logo_url, width=240)
#
# Delete the later/second st.set_page_config(...) call entirely.
