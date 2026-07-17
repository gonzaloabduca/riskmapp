# RiskMapp

An interactive equity-research and risk-management platform built with Python, machine learning, Streamlit, Docker, GitHub, and Microsoft Azure.

**Live application:** Add Azure URL
**Author:** Gonzalo Abduca

---

## Overview

RiskMapp consolidates multiple parts of the equity-research process into one interactive dashboard.

The application combines:

* Company fundamentals
* Historical and forward valuation
* Machine-learning market-regime detection
* Technical and volatility analysis
* Options-market positioning
* Analyst expectations
* Institutional ownership
* Peer comparisons
* Earnings analysis
* Statistical risk metrics

Its purpose is to help investors answer four central questions:

> What is the company worth, how is the stock behaving, what risks are present, and how does it compare with similar businesses?

---

## Application Workflow

The user enters a stock ticker such as:

```text
AAPL
MSFT
NVDA
COIN
```

RiskMapp retrieves available market, company, financial-statement, analyst, institutional-ownership, and options data.

The application then produces analysis across four dimensions:

```text
Company
│
├── Business and fundamentals
├── Price behavior and market regime
├── Valuation and peer positioning
└── Risk, options, and market expectations
```

---

## Main Features

### Company Snapshot

Provides an immediate overview of the selected company:

* Company name
* Sector and industry
* Current market price
* Analyst price targets
* Implied upside and downside
* Business description
* Company website

### Machine-Learning Market Regimes

RiskMapp uses an unsupervised Gaussian Hidden Markov Model to identify recurring statistical regimes in a stock’s historical behavior.

The model uses features such as:

* Price momentum
* ADX trend strength
* Short-term and long-term volatility
* Volatility of volatility
* Relative trading volume
* Distance from the long-term moving average
* Drawdown from historical highs
* Hurst exponent

The feature pipeline is:

```text
Market data
    ↓
Feature engineering
    ↓
Rolling normalization
    ↓
StandardScaler
    ↓
Principal Component Analysis
    ↓
Gaussian Hidden Markov Model
    ↓
Latent market regimes
```

Each regime is evaluated using:

* Average forward return
* Annualized return
* Annualized volatility
* Sharpe ratio
* Number of observations

The strongest historical risk-adjusted state is classified as favorable, while the weakest is classified as unfavorable.

The model is intended as a risk-classification tool, not as a guaranteed prediction system.

### Trend and Volatility Analysis

The dashboard includes:

* Average Directional Index
* Positive and negative directional movement
* Standardized MACD momentum
* Volatility-adjusted trend bands
* Long-term moving-average extension
* Relative volume
* Rolling drawdown
* Hurst exponent

### Options Positioning

RiskMapp aggregates call and put open interest by strike and estimates potential positioning levels.

It highlights:

* Call walls
* Put walls
* Support zones
* Resistance zones
* High-concentration option strikes

These levels are displayed on an interactive candlestick chart.

This analysis is an approximation based on publicly available open-interest data. It does not observe dealers’ actual hedge positions.

### Implied Volatility

The options module includes:

* Black–Scholes option pricing
* Numerical implied-volatility estimation
* Brent root-finding
* Call and put support
* Continuous dividend yield
* Multiple expiration dates
* At-the-money option selection
* Bid–ask midpoint pricing

### Fundamental Analysis

The dashboard evaluates:

**Valuation**

* Market capitalization
* Enterprise value
* Trailing P/E
* Forward P/E
* PEG ratio
* Price-to-book ratio
* Sales multiple

**Growth**

* Revenue growth
* Earnings growth
* Historical sales growth
* Historical EPS growth
* Expected EPS growth

**Profitability**

* Gross margin
* Operating margin
* EBIT yield
* Free-cash-flow yield
* Return on equity
* Return on assets

**Balance sheet**

* Debt-to-equity ratio
* Current ratio
* Quick ratio
* Cash per share

### Historical Valuation

RiskMapp reconstructs historical valuation using reported financial statements and market prices.

The table includes:

* Stock price
* Market capitalization
* Revenue
* Revenue growth
* EPS
* Earnings growth
* P/E ratio
* PEG ratio
* Sales multiple
* Net income
* Forward analyst estimates

This helps distinguish between:

* Business growth
* Multiple expansion
* Multiple compression
* Changes in market expectations

### Peer Comparison

The selected company is compared with businesses from a similar industry and size category.

Available comparison metrics include:

* Current P/E
* Forward P/E
* PEG ratio
* EPS growth expectations
* Short interest
* Growth score
* Efficiency score
* Value score
* Fragility score

### Risk Analytics

Depending on data availability, RiskMapp calculates:

* Total return
* Annualized return
* Annualized volatility
* Sharpe ratio
* Sortino ratio
* Maximum drawdown
* Return skewness
* Return kurtosis
* Rolling volatility
* Benchmark-relative performance
* Return distribution

---

## Technical Architecture

```text
User enters ticker
        ↓
Yahoo Finance and local datasets
        ↓
Data processing
Pandas · NumPy · SciPy
        ↓
┌───────────────────┬───────────────────┬───────────────────┐
│ Market-regime ML  │ Fundamentals      │ Options and risk  │
│ model             │ and valuation     │ analytics         │
└───────────────────┴───────────────────┴───────────────────┘
        ↓
Plotly visualizations
        ↓
Streamlit interface
        ↓
Docker container
        ↓
Azure Container Registry
        ↓
Azure Container Apps
```

---

## Technology Stack

| Layer                 | Technology                                     |
| --------------------- | ---------------------------------------------- |
| Language              | Python 3.11                                    |
| Web application       | Streamlit                                      |
| Market data           | yfinance                                       |
| Data processing       | pandas, NumPy                                  |
| Machine learning      | scikit-learn, hmmlearn                         |
| Statistical analysis  | SciPy, statsmodels                             |
| Performance analytics | QuantStats                                     |
| Visualization         | Plotly, Matplotlib                             |
| Spreadsheet support   | Excel, openpyxl                                |
| Containerization      | Docker                                         |
| Source control        | Git, GitHub                                    |
| Cloud deployment      | Azure Container Registry, Azure Container Apps |

---

## Running Locally

### Clone the repository

```bash
git clone https://github.com/gonzaloabduca/riskmapp.git
cd riskmapp
```

### Create a virtual environment

Using Conda:

```bash
conda create -n riskmapp python=3.11 -y
conda activate riskmapp
```

Using `venv`:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start Streamlit

```bash
streamlit run riskmapp.py
```

Open:

```text
http://localhost:8501
```

---

## Running with Docker

Build the image:

```bash
docker build -t riskmapp:latest .
```

Run the container:

```bash
docker run --rm -p 8501:8501 riskmapp:latest
```

Open:

```text
http://localhost:8501
```

---

## Azure Deployment

The production deployment follows this architecture:

```text
Local source code
        ↓
Docker image build
        ↓
Azure Container Registry
        ↓
Azure Container Apps
```

The Docker image is built locally and pushed to Azure Container Registry.

Azure Container Apps then runs the image as a public web application with Streamlit exposed on port `8501`.

Future releases can automate this process through GitHub Actions:

```text
Push to main
    ↓
Build Docker image
    ↓
Push image to ACR
    ↓
Update Azure Container App
```

---

## Environment Variables

Credentials and external-service tokens should be passed through environment variables and must not be committed to GitHub.

Example:

```text
LOGO_DEV_TOKEN=your_token
```

Python:

```python
import os

logo_token = os.getenv("LOGO_DEV_TOKEN")
```

Docker:

```bash
docker run --rm \
  -p 8501:8501 \
  -e LOGO_DEV_TOKEN="your_token" \
  riskmapp:latest
```

For Azure Container Apps, secrets and environment variables can be configured through the Azure portal or Azure CLI.

---

## Repository Structure

```text
riskmapp/
│
├── .streamlit/
│   └── config.toml
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── app_path_patch.py
├── deploy-container-app.ps1
├── market_regimes.csv
├── requirements.txt
├── riskmapp.py
└── us_stock_market_watchlist2 2025-11.xlsx
```

---

## Engineering Decisions

### Streamlit Caching

External data requests and computationally expensive transformations are cached to:

* Improve response times
* Reduce repeated API calls
* Avoid unnecessary model retraining
* Improve dashboard stability

### Relative File Paths

Project files are loaded relative to the application directory rather than through local Windows paths.

This allows the application to run consistently across:

* Windows
* Linux
* Docker
* Azure

### Containerization

Docker packages the application, Python runtime, dependencies, and required files into a reproducible image.

This reduces differences between local and production environments.

### Defensive Data Handling

Financial APIs do not always provide every field for every security.

The application uses validation, fallback values, and exception handling so unavailable information does not necessarily stop the complete dashboard.

---

## Current Limitations

* Market and financial data depend on Yahoo Finance availability.
* Some securities have incomplete analyst, ownership, options, or financial-statement information.
* Open interest does not reveal actual dealer positioning.
* Market regimes are statistical classifications, not certain forecasts.
* Historical relationships may not persist in future market conditions.
* The project is a portfolio and research application rather than an institutional trading system.

---

## Future Improvements

* Refactor the main application into modular services
* Add automated unit and integration tests
* Implement GitHub Actions CI/CD
* Add persistent storage and historical snapshots
* Expand portfolio-level analysis
* Add model-stability and drift monitoring
* Improve scenario and stress testing
* Add Azure monitoring and structured logs
* Add authentication and user-specific watchlists
* Generate downloadable research reports

---

## What This Project Demonstrates

RiskMapp demonstrates practical experience across:

* Financial-domain problem definition
* Market-data ingestion
* Financial-statement processing
* Quantitative feature engineering
* Statistical modeling
* Unsupervised machine learning
* Equity valuation
* Options analysis
* Risk measurement
* Interactive dashboard development
* Dependency management
* Git and GitHub workflows
* Docker containerization
* Azure cloud deployment

```text
Finance + Data Science + Software Engineering + Cloud Deployment
```

---

## Disclaimer

RiskMapp is an educational and portfolio project.

It does not provide investment advice, trading recommendations, or guarantees of future performance. Financial markets involve substantial risk, and historical results do not ensure future outcomes.

The displayed data may be delayed, incomplete, or inaccurate. Users should independently verify all information before making investment decisions.

---

## License

This project is available under the MIT License.

---

## Contact

**Gonzalo Abduca**

* LinkedIn: https://www.linkedin.com/in/gonzaloabduca/
* GitHub: https://github.com/gonzaloabduca/
* Email: [abducagonzalo@gmail.com](mailto:abducagonzalo@gmail.com)
