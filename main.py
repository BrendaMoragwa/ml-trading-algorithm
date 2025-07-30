from AlgorithmImports import *
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Using U.S. Unemployment Rate

class UnemploymentData(PythonData):
    def GetSource(self, config, date, isLive):
        return SubscriptionDataSource(
            "https://raw.githubusercontent.com/BrendaMoragwa/trading-algorithm/main/UNRATE.csv",
            SubscriptionTransportMedium.RemoteFile
        )

    def Reader(self, config, line, date, isLive):
        if not line or line.lower().startswith("observation_date"):
            return None

        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or parts[1] in (".", ""):
            return None

        try:
            time = datetime.strptime(parts[0], "%Y-%m-%d" if "-" in parts[0] else "%m/%d/%Y")
            data = UnemploymentData()
            data.Time = time
            data.Value = float(parts[1])
            data.Symbol = config.Symbol
            return data
        except (ValueError, IndexError):
            return None

class SimplifiedMLStrategy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)

        # Add equities
        self.symbols = {
            "SPY": self.AddEquity("SPY", Resolution.Daily).Symbol,
            "TLT": self.AddEquity("TLT", Resolution.Daily).Symbol,
            "QQQ": self.AddEquity("QQQ", Resolution.Daily).Symbol
        }

        # Add unemployment data
        self.unrate_symbol = self.AddData(UnemploymentData, "UNEMPLOYMENT", Resolution.Daily).Symbol
        self.unemployment_rate = 0.0

        # Setup indicators for each asset
        self.indicators = {}
        for name, symbol in self.symbols.items():
            self.indicators[name] = {
                'rsi': self.RSI(symbol, 14, Resolution.Daily),
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily)
            }

        # ML parameters
        self.lookback_days = 252
        self.retrain_frequency = 60
        self.min_accuracy = 0.55
        self.confidence_threshold = 0.60
        # 15% per pair
        self.max_position = 0.30

        # Risk Management
        # 2% stop loss
        self.stop_loss_pct = 0.02
        # 4% take profit
        self.take_profit_pct = 0.04 
        # 1.5% trailing stop
        self.trailing_stop_pct = 0.015

        # Storage
        self.data_buffer = {name: [] for name in self.symbols}
        self.models = {}
        self.scalers = {}
        self.last_train_date = {}
        self.model_ready = {name: False for name in self.symbols}
        self.entry_prices = {} 
        self.highest_prices = {}

        self.Log("ML EQUITY STRATEGY (SPY, TLT, QQQ)")

    def OnData(self, data):
        # Update unemployment rate
        if self.unrate_symbol in data:
            self.unemployment_rate = data[self.unrate_symbol].Value

        # Process each asset
        for name, symbol in self.symbols.items():
            if symbol in data and self._indicators_ready(name):
                if data[symbol]:
                    current_price = data[symbol].Close
                    
                    # Check stop loss/take profit first
                    self._check_risk_management(name, symbol, current_price)
                    
                    # process normal trading logic
                    self._process_pair(name, symbol, current_price)
            

    def _check_risk_management(self, name, symbol, current_price):
        # Check stop loss and take profit conditions
        if not self.Portfolio[symbol].Invested:
            return
        
        position = self.Portfolio[symbol]
        if name not in self.entry_prices:
            # Initialize entry price if missing
            self.entry_prices[name] = position.AveragePrice
            self.highest_prices[name] = current_price
        
        entry_price = self.entry_prices[name]
        is_long = position.IsLong
        
        # Calculate returns
        if is_long:
            pnl_pct = (current_price - entry_price) / entry_price
            # Update trailing stop high
            if current_price > self.highest_prices[name]:
                self.highest_prices[name] = current_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            # Update trailing stop low for short positions
            if current_price < self.highest_prices[name]:
                self.highest_prices[name] = current_price
        
        should_close = False
        close_reason = ""
        
        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            should_close = True
            close_reason = "STOP_LOSS"
        
        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            should_close = True
            close_reason = "TAKE_PROFIT"
        
        # Trailing Stop
        elif is_long:
            trailing_stop_price = self.highest_prices[name] * (1 - self.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                should_close = True
                close_reason = "TRAILING_STOP"
        else:  # Short position
            trailing_stop_price = self.highest_prices[name] * (1 + self.trailing_stop_pct)
            if current_price >= trailing_stop_price:
                should_close = True
                close_reason = "TRAILING_STOP"
        
        # Execute close if needed
        if should_close:
            self.Liquidate(symbol)
            self.Log(f"{close_reason} {name} @ {current_price:.5f} | P&L: {pnl_pct:.2%}")
            # Reset tracking
            if name in self.entry_prices:
                del self.entry_prices[name]
            if name in self.highest_prices:
                del self.highest_prices[name]

    def _indicators_ready(self, name):
        # Check if indicators are ready for this pair
        inds = self.indicators[name]
        return inds['rsi'].IsReady and inds['macd'].IsReady

    def _process_pair(self, name, symbol, price):
        # Main processing pipeline for each currency pair
        # Create feature vector
        features = self._get_features(name)
        
        # Store data point
        self._store_data(name, features, price)
        
        # Train model if needed
        if self._should_retrain(name):
            self._train_model(name)
        
        # Make trading decision if model is ready
        if self.model_ready[name]:
            self._make_trade_decision(name, symbol, features)

    def _get_features(self, name):
        #Extract features for ML model
        inds = self.indicators[name]
        return np.array([
            float(inds['rsi'].Current.Value),
            float(inds['macd'].Current.Value),
            float(self.unemployment_rate)
        ])

    def _store_data(self, name, features, price):
        #Store data point in buffer
        self.data_buffer[name].append({
            'date': self.Time,
            'features': features,
            'price': price
        })
        
        # Keeps only recent data
        if len(self.data_buffer[name]) > self.lookback_days * 2:
            self.data_buffer[name] = self.data_buffer[name][-self.lookback_days * 2:]

    def _should_retrain(self, name):
        #Check if we should retrain the model
        if len(self.data_buffer[name]) < self.lookback_days:
            return False
        
        if name not in self.last_train_date:
            return True
            
        days_since_train = (self.Time - self.last_train_date[name]).days
        return days_since_train >= self.retrain_frequency

    def _train_model(self, name):
        #Train and validate the ML model
        try:
            data = self.data_buffer[name]
            if len(data) < self.lookback_days:
                return

            # Split data into 80% train, 20% validate
            split_idx = int(len(data) * 0.8)
            train_data = data[:split_idx]
            val_data = data[split_idx:]

            # Prepare training data
            X_train, y_train = self._prepare_features_labels(train_data)
            if len(X_train) < 50:
                return

            # Initialize model and scaler
            self.models[name] = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_split=10,
                random_state=42
            )
            self.scalers[name] = StandardScaler()

            # Train model
            X_train_scaled = self.scalers[name].fit_transform(X_train)
            self.models[name].fit(X_train_scaled, y_train)

            # Validate model
            X_val, y_val = self._prepare_features_labels(val_data)
            if len(X_val) > 0:
                X_val_scaled = self.scalers[name].transform(X_val)
                accuracy = self.models[name].score(X_val_scaled, y_val)
                
                # Enable trading only if model meets accuracy threshold
                self.model_ready[name] = accuracy >= self.min_accuracy
                self.last_train_date[name] = self.Time
                
                self.Log(f"{name}: Model trained - Accuracy: {accuracy:.3f}, Trading: {self.model_ready[name]}")
            
        except Exception as e:
            self.Log(f"Training error for {name}: {str(e)}")
            self.model_ready[name] = False

    def _prepare_features_labels(self, data):
        #Prepare features and labels for ML training
        X, y = [], []
        lookforward = 5
        
        for i in range(len(data) - lookforward):
            current = data[i]
            future = data[i + lookforward]
            
            # Calculate forward return
            forward_return = (future['price'] - current['price']) / current['price']
            
            # Binary classification: 1 if return > 0.05%, 0 otherwise
            label = 1 if forward_return > 0.0005 else 0
            
            X.append(current['features'])
            y.append(label)
        
        return np.array(X), np.array(y)

    def _make_trade_decision(self, name, symbol, features):
        #Make trading decision based on model prediction
        try:
            # Get model prediction
            X_scaled = self.scalers[name].transform([features])
            probabilities = self.models[name].predict_proba(X_scaled)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]
            
            # Only trade if confidence is high enough
            if confidence < self.confidence_threshold:
                return
            
            # Calculate position size based on confidence
            position_size = self.max_position * (confidence - 0.5) / 0.5
            
            # Determine target position
            target_position = position_size if prediction == 1 else -position_size
            
            # Execute trade
            current_position = self.Portfolio[symbol].HoldingsValue / self.Portfolio.TotalPortfolioValue
            
            if abs(target_position - current_position) > 0.02:  # 2% threshold for rebalancing
                self.SetHoldings(symbol, target_position)
                action = "BUY" if prediction == 1 else "SELL"
                
                # Track entry price for risk management
                if target_position != 0:
                    self.entry_prices[name] = features[0]  
                    self.highest_prices[name] = features[0]
                
                self.Log(f"{action} {name} - Confidence: {confidence:.3f}, Size: {target_position:.3f}")
        
        except Exception as e:
            self.Log(f"Trading error for {name}: {str(e)}")

    def OnOrderEvent(self, orderEvent):
        #Track actual fill prices for stop loss calculations
        if orderEvent.Status == OrderStatus.Filled:
            symbol = orderEvent.Symbol
            # Find currency name
            for name, sym in self.symbols.items():
                if sym == symbol:
                    if self.Portfolio[symbol].Invested:
                        self.entry_prices[name] = orderEvent.FillPrice
                        self.highest_prices[name] = orderEvent.FillPrice
                    break

    def OnEndOfAlgorithm(self):
        #Summary at end of backtest
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - 100000) / 100000 * 100
        
        self.Log("BACKTEST RESULTS")
        self.Log(f"Final Portfolio Value: ${final_value:,.2f}")
        self.Log(f"Total Return: {total_return:.2f}%")
        
        # Model status summary
        for name in self.symbols:
            status = "ACTIVE" if self.model_ready[name] else "INACTIVE"
            self.Log(f"{name}: {status}")