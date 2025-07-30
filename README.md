# ML-Driven Equity Trading Strategy

A QuantConnect trading strategy that uses machine learning to trade SPY, TLT, and QQQ based on RSI, MACD, and the U.S. Unemployment Rate.

## Overview

This repository implements a Random Forest–based trading algorithm that:

- **Predicts** short-term direction (“up” or “down”) for equities  
- **Trades** both long and short positions  
- **Combines** technical indicators (RSI, MACD) with a macroeconomic feature (unemployment rate)  
- **Backtests** performance over 2022–2024  

## Technologies

- **Python**  
- **QuantConnect Lean** framework  
- **scikit-learn** (RandomForestClassifier, StandardScaler)  
- **NumPy**  
- **datetime** (for feature engineering)  

## How It Works

1. **Data Ingestion**  
   - Equity price data for SPY, TLT, QQQ (daily)  
   - Custom `UnemploymentData` PythonData class reads U.S. unemployment CSV from GitHub  

2. **Feature Engineering**  
   - **RSI (14-day)**  
   - **MACD (12,26,9)**  
   - **Unemployment rate**  

3. **ML Pipeline**  
   - Maintain a rolling buffer of the last 252 days  
   - Retrain every 60 trading days if validation accuracy ≥ 0.55  
   - Random Forest with 50 trees, max depth 6  

4. **Trading Logic**  
   - Use `predict_proba` to get “up” vs “down” probabilities  
   - Only trade when confidence ≥ 0.60  
   - Position sizing: up to 30% of portfolio, scaled by confidence  
   - Risk management: 2% stop-loss, 4% take-profit, 1.5% trailing stop  

5. **Backtest Summary**  
   - Logs final portfolio value and total return  
   - Indicates whether each model remained active  

## Repository Structure
```text
ML-Trading-Algorithm/
├── main.py        # SimplifiedMLStrategy + UnemploymentData classes
├── UNRATE.csv     # Unemployment rate data
└── README.md      # This documentation

## To get started
1. Clone this repo
- git clone https://github.com/BrendaMoragwa/ml-trading-algorithm.git
- cd ml-trading-algorithm
  
2. Backtest in QuantConnect IDE or with Lean CLI
- lean backtest main.py --start 20220101 --end 20241231 --cash 100000

3. Review logs and performance metrics


