RiskMapp
An interactive stock research and risk-management platform built with Python, machine learning, Streamlit, Docker, and Azure
Live application: 
Author: Gonzalo Abduca

Overview
RiskMapp is an interactive equity-research and risk-management application designed to bring several parts of the investment process into one dashboard.

Instead of evaluating a stock using only price charts or isolated valuation ratios, RiskMapp combines:

Company fundamentals
Historical and forward valuation
Machine-learning market-regime detection
Trend and volatility analysis
Options-market positioning
Analyst expectations
Institutional ownership
Peer and sector comparisons
Earnings-event analysis
Statistical risk metrics
The objective is to help investors move from raw financial data to a more structured investment view:

What is the company worth, how is the stock behaving, what risks are present, and how does it compare with similar businesses?

Why I Built This Project
Equity analysis is often fragmented across multiple tools.

An investor may use one platform for financial statements, another for technical charts, another for options data, and a spreadsheet for peer comparisons. This makes the research process slower and increases the risk of evaluating signals independently rather than as part of a complete investment framework.

RiskMapp was built to consolidate those workflows into one application.

The project demonstrates the ability to:

Retrieve and process financial-market data.
Engineer quantitative trading and risk features.
Apply unsupervised machine learning to financial time series.
Translate analytical outputs into interactive visualizations.
Build a usable Streamlit product.
Package the application with Docker.
Deploy the container as a cloud web application on Microsoft Azure.
Application Workflow
The user enters a stock ticker such as:

AAPL
MSFT
NVDA
COIN
RiskMapp then retrieves and processes the available company, market, analyst, financial-statement, and options data.

The dashboard produces a structured analysis across four main dimensions:

Company
   │
   ├── Business and fundamentals
   ├── Price behavior and market regime
   ├── Valuation and peer positioning
   └── Risk, options and market expectations
This allows the user to evaluate both the underlying business and the behavior of the traded security.

Main Features
1. Company Snapshot
The application begins with a concise company overview that includes:

Company name
Sector and industry
Current market price
Analyst high and low price targets
Implied upside
Implied downside
Company website
Business description
Company logo
This gives the user immediate context before moving into deeper analysis.

2. Machine-Learning Market-Regime Detection
RiskMapp uses an unsupervised Gaussian Hidden Markov Model to identify latent market regimes in each stock’s historical behavior.

The model does not receive predefined bull or bear labels. Instead, it learns recurring statistical states from engineered market features.

Input features
The regime model incorporates information such as:

Price momentum
ADX trend strength
Short-term versus long-term volatility
Volatility of volatility
Relative trading volume
Distance from the long-term moving average
Drawdown from historical highs
Hurst exponent
These features are standardized and reduced using Principal Component Analysis before being passed to the Hidden Markov Model.

Raw market data
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
Regime interpretation
For every detected regime, the application calculates:

Average forward return
Annualized return
Annualized volatility
Annualized Sharpe ratio
Number of observations
The regime with the strongest historical risk-adjusted return is classified as the favorable state, while the weakest is treated as the unfavorable state.

The results are displayed directly on an interactive candlestick chart.

Why this matters
Markets do not behave according to one stable statistical distribution. Momentum, volatility, liquidity, and trend persistence can change considerably through time.

Regime detection helps answer:

Is the stock currently behaving in a historically favorable or unfavorable market state?

This is intended as a risk-classification tool rather than a guaranteed price-prediction system.

3. Quantitative Trend Analysis
The application combines regime classification with several quantitative trend indicators.

These include:

Average Directional Index
Positive and negative directional movement
Standardized MACD momentum
Volatility-adjusted trend bands
Long-term moving-average extension
Relative volume
Rolling drawdown
Hurst exponent
The Hurst exponent is used to assess whether recent returns exhibit characteristics associated with:

Trend persistence
Random behavior
Mean reversion
The resulting trend line is displayed together with market-regime information and historical price action.

4. Options Dealer-Pressure Map
RiskMapp retrieves available option chains and aggregates call and put open interest by strike.

It estimates the most relevant positioning levels by examining:

Net open interest = Call open interest − Put open interest
The application then compares the estimated positioning at each strike with the stock’s average trading volume.

The output highlights potential:

Call walls
Put walls
Support zones
Resistance zones
High-concentration option strikes
These levels are overlaid on an interactive candlestick chart.

The width of each level reflects its estimated relative importance.

This module is an approximation based on publicly available open-interest data. It does not observe dealers’ complete books or actual hedge positions.

5. Implied-Volatility Analysis
The options module also contains a Black–Scholes pricing implementation and a numerical implied-volatility solver.

Implied volatility is estimated using Brent’s root-finding method, which solves for the volatility that equates the theoretical option value with its observed market price.

The analysis supports:

Calls and puts
Continuous dividend yield
Multiple expiration dates
At-the-money option selection
Bid–ask midpoint pricing
Historical implied-volatility comparison
This helps evaluate whether current option pricing implies unusually high or low expected movement.

6. Analyst Expectations
RiskMapp summarizes market expectations through:

Analyst high price target
Analyst low price target
Mean price target
Median price target
Historical upgrades
Historical downgrades
Maintained ratings
Reiterated ratings
Initiated coverage
Price targets are shown directly against the stock’s historical price.

This makes it easier to compare current market pricing with the range of published analyst expectations.

7. Institutional Ownership
The ownership module analyzes:

Major institutional holders
Percentage held by institutions
Percentage held by insiders
Remaining public ownership
Number of reporting institutions
The results are displayed through interactive bar charts and ownership-distribution charts.

This provides additional context about the composition and concentration of the shareholder base.

8. Fundamental Company Analysis
The dashboard collects and calculates a broad set of financial indicators, including:

Valuation
Market capitalization
Enterprise value
Trailing P/E
Forward P/E
PEG ratio
Price-to-book ratio
Sales multiple
Growth
Revenue growth
Earnings growth
Expected EPS growth
Historical sales growth
Historical EPS growth
Profitability and efficiency
Gross margin
Operating margin
EBIT yield
Free-cash-flow yield
Return on equity
Return on assets
Balance-sheet and liquidity
Debt-to-equity ratio
Current ratio
Quick ratio
Cash per share
Event risk
Next earnings date
Days remaining until earnings
These metrics provide a quick but broad view of the company’s financial profile.

9. Historical and Forward Valuation Table
RiskMapp reconstructs a multi-year financial and valuation history using market prices and reported financial statements.

The table includes:

Stock price
Market capitalization
Revenue
Revenue growth
EPS
Earnings growth
Price-to-earnings ratio
PEG ratio
Sales multiple
Net income
Trailing-twelve-month results
Forward analyst estimates
The application aligns reported financial-statement dates with available market prices to show how the stock was valued at different points in time.

This helps distinguish between:

Business growth
Multiple expansion
Multiple compression
Changes in market expectations
10. Sector and Peer Comparison
The selected stock is compared with companies that share a similar:

Industry
Company-size classification
The peer table includes available fields such as:

Current P/E
Forward P/E
PEG ratio
EPS growth expectations
Short interest
Growth score
Efficiency score
Value score
Fragility score
Venture score
Pain score
The application also produces:

Company versus sector valuation chart
Company versus sector EPS-growth comparison
Multi-factor radar chart
Peer-level comparison table
This places company-specific metrics in the context of relevant competitors rather than evaluating them in isolation.

11. Earnings Analysis
The application retrieves historical and expected earnings information and can evaluate:

Upcoming earnings dates
Historical EPS estimates
Reported EPS
Earnings surprises
Changes in expectations
Event timing
Earnings events are an important source of discontinuous risk. Displaying them alongside the stock’s broader analytical profile helps the user identify when additional caution may be required.

12. Risk and Performance Analytics
RiskMapp uses statistical and portfolio-analysis tools to evaluate return behavior.

Depending on data availability, the application can calculate or visualize measures such as:

Total return
Annualized return
Annualized volatility
Sharpe ratio
Sortino ratio
Maximum drawdown
Return skewness
Return kurtosis
Rolling volatility
Benchmark-relative performance
Drawdown history
Return distribution
The purpose is not only to identify potential upside, but also to quantify the shape and severity of downside risk.

Technical Architecture
                         ┌─────────────────────┐
                         │   User enters       │
                         │   stock ticker      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Yahoo Finance     │
                         │   Financial data    │
                         │   Options data      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Data processing         │
                    │ Pandas · NumPy · SciPy       │
                    └──────────────┬───────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
       ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
       │ Market-regime  │ │ Fundamentals   │ │ Options and    │
       │ model          │ │ and valuation  │ │ risk analytics │
       └───────┬────────┘ └───────┬────────┘ └───────┬────────┘
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ Plotly visualizations│
                       │ Streamlit interface  │
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ Docker container     │
                       │ Azure web deployment │
                       └──────────────────────┘
Technology Stack
Layer

Technology

Language

Python 3.11

Web application

Streamlit

Market data

yfinance

Data processing

pandas, NumPy

Machine learning

scikit-learn, hmmlearn

Statistical analysis

SciPy, statsmodels

Performance analytics

QuantStats

Visualization

Plotly, Matplotlib, Seaborn

HTML/XML processing

lxml

Data files

Excel, openpyxl

Containerization

Docker

Source control

Git and GitHub

Cloud deployment

Microsoft Azure

Machine-Learning Methodology
Feature engineering
Historical market data is converted into normalized signals that represent:

Direction
Momentum
Trend strength
Volatility
Trading activity
Price extension
Drawdown
Persistence
Standardization
Features with different units and scales are normalized using StandardScaler.

Dimensionality reduction
Principal Component Analysis compresses correlated trading indicators into a smaller set of orthogonal factors.

This reduces redundancy and makes the regime model less dependent on any single raw indicator.

Hidden Markov Model
A Gaussian Hidden Markov Model identifies hidden states that may have generated the observed market factors.

The state labels themselves have no predefined economic meaning. Their interpretation is assigned after evaluating the return and risk characteristics associated with each state.

Regime evaluation
Each state is evaluated using its subsequent returns:

Annualized return = Mean daily return × 252

Annualized volatility = Daily volatility × √252

Sharpe ratio = Annualized return ÷ Annualized volatility
The most favorable and unfavorable regimes are then displayed on the price chart.

Running the Project Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/riskmapp.git
cd riskmapp
2. Create a virtual environment
Using Conda:

conda create -n riskmapp python=3.11 -y
conda activate riskmapp
Or using venv:

python -m venv .venv
Windows:

.venv\Scripts\activate
Linux or macOS:

source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Run the application
streamlit run riskmapp.py
Open:

http://localhost:8501
Running with Docker
Build the image
docker build -t riskmapp:latest .
Run the container
docker run --rm -p 8501:8501 riskmapp:latest
Open:

http://localhost:8501
Environment Variables
Credentials and external-service tokens should be passed as environment variables rather than committed to the repository.

Example:

LOGO_DEV_TOKEN=your_token
In Python:

import os

logo_token = os.getenv("LOGO_DEV_TOKEN")
For Docker:

docker run --rm \
  -p 8501:8501 \
  -e LOGO_DEV_TOKEN="your_token" \
  riskmapp:latest
For Azure, add the variable under:

App Service
→ Settings
→ Environment variables
Azure Deployment
The application is designed to run as a Linux Docker container on Azure.

GitHub repository
        ↓
Docker image build
        ↓
Azure Container Registry
        ↓
Azure App Service
The container exposes Streamlit on port 8501.

The following Azure environment setting is required:

WEBSITES_PORT=8501
The application can be connected to GitHub Actions so every push to the main branch automatically:

Builds a new Docker image.
Pushes the image to Azure Container Registry.
Deploys the updated image to Azure App Service.
Restarts the web application.
Repository Structure
riskmapp/
│
├── riskmapp.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── us_stock_market_watchlist2 2025-11.xlsx
└── .github/
    └── workflows/
        └── azure-deployment.yml
As the project evolves, the application can be refactored into modules:

riskmapp/
│
├── app.py
├── src/
│   ├── data_loader.py
│   ├── regime_model.py
│   ├── options_analysis.py
│   ├── fundamentals.py
│   ├── risk_metrics.py
│   └── visualizations.py
├── data/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
Key Engineering Decisions
Streamlit caching
External financial-data requests and computationally expensive transformations are cached to:

Improve response times
Reduce repeated API calls
Avoid unnecessary model retraining
Make the dashboard more stable
Relative file paths
Project files are loaded relative to the application directory rather than through local Windows paths.

This ensures the application works consistently across:

Windows
Linux
Docker
Azure
Containerization
Docker packages the application, Python runtime, dependencies, and required local files into a reproducible image.

This avoids the common problem of an application working locally but failing in production because of different package versions or operating-system dependencies.

Defensive data handling
Financial APIs do not always return every field for every company.

The application uses validation, fallback values, and exception handling so unavailable information does not necessarily stop the full dashboard.

Current Limitations
Market and company data depend on the availability and structure of Yahoo Finance data.
Some securities do not have complete analyst, ownership, options, or financial-statement information.
Option open interest does not reveal actual dealer positioning or hedge direction.
Market regimes are identified statistically and should not be interpreted as certain forecasts.
Analyst estimates may change and should not be treated as intrinsic value.
Historical relationships may not persist in future market conditions.
The current application is primarily a research and portfolio project rather than an institutional trading system.
Future Improvements
Planned improvements include:

Refactoring the single application file into modular services
Automated unit and integration testing
More robust data validation
Additional market-data providers
Persistent storage and historical snapshots
Portfolio-level analysis
Fundamental factor ranking
Explainable regime-classification outputs
Model stability and drift monitoring
Scenario and stress testing
CI/CD through GitHub Actions
Azure monitoring and structured application logs
Authentication and user-specific watchlists
Downloadable research reports
What This Project Demonstrates
RiskMapp demonstrates practical experience across the full development lifecycle:

Financial-domain problem definition
Market and financial-statement data ingestion
Quantitative feature engineering
Statistical modeling
Unsupervised machine learning
Equity valuation
Options analysis
Risk measurement
Interactive dashboard development
Dependency management
Docker containerization
Git and GitHub workflows
Cloud deployment on Azure
The project sits at the intersection of:

Finance + Data Science + Software Engineering + Cloud Deployment
Disclaimer
RiskMapp is an educational and portfolio project.

It does not provide investment advice, trading recommendations, or guarantees of future performance. Financial markets involve substantial risk, and historical results do not ensure future outcomes.

The data displayed by the application may be delayed, incomplete, or inaccurate. Users should independently verify all information before making investment decisions.

License
This project is available under the MIT License.

Contact
Gonzalo Abduca

LinkedIn: https://www.linkedin.com/in/gonzaloabduca/
GitHub: https://github.com/gonzaloabduca/
Email: abducagonzalo@gmail.com