# ML Trading Strategy

A Python trading algorithm that uses machine learning to trade SPY, TLT, and QQQ stocks automatically.

## What It Does

This algorithm trades three popular ETFs:
- **SPY** - S&P 500 stocks
- **TLT** - Treasury bonds  
- **QQQ** - Tech stocks

It uses AI to predict if prices will go up or down, then buys or sells automatically.

## How It Works

1. **Gets Data**
   - Stock prices
   - RSI and MACD indicators
   - US unemployment rate

2. **AI Predictions**
   - Random Forest machine learning model
   - Predicts 5-day price movements
   - Only trades when 60%+ confident

3. **Makes Trades**
   - Buys when predicting price increases
   - Sells when predicting price decreases
   - Uses up to 30% of money per stock

4. **Risk Management**
   - Stop loss: Sells if losing 2%
   - Take profit: Sells if gaining 4%
   - Trailing stop: Protects profits at 1.5%

## Results

**Period:** 2022-2024 (3 years)
**Starting Money:** $100,000
**Final Value:** $100,744
**Return:** 0.74%
**Profit:** $744

<img width="1868" height="1028" alt="image" src="https://github.com/user-attachments/assets/d7e46c9a-1967-43cd-9271-01ed92f3806f" />

## Key Features

- Automated trading
- Risk controls
- Multi-asset portfolio
- Self-learning AI
- Backtested on real data

## Architecture


Data → ML Model → Trading → Risk Controls
 ↓        ↓         ↓          ↓
Prices  Random   Buy/Sell   Stop Loss
RSI     Forest   Orders     Take Profit
MACD    (50 trees)          Trailing Stop
Unemployment


## Code Structure

- **UnemploymentData** - Gets economic data
- **SimplifiedMLStrategy** - Main trading logic
- **Risk Management** - Safety controls
- **ML Pipeline** - AI training and predictions

## Setup

1. Copy code to QuantConnect
2. Set date range and starting money
3. Run backtest
4. Check results

## Key Parameters

- Training window: 252 days (1 year)
- Retrain every: 60 days
- Minimum accuracy: 55%
- Confidence threshold: 60%
- Max position: 30% per asset
- Stop loss: 2%
- Take profit: 4%
- Trailing stop: 1.5%

## What Worked

- No major losses
- Automatic operation
- Good risk management
- Clean code architecture

## What Could Improve

- Low returns (0.74%)
- Need better features
- Market timing is hard
- Could use more data
