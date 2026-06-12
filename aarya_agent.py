import numpy as np
import pandas as pd
import warnings

# Force-silence any environment warnings for the online server
warnings.filterwarnings("ignore")

class InstitutionalAlphaEngine:
    @staticmethod
    def calculate_hurst_exponent(close_prices: np.ndarray, max_lags: int =
     10) -> float:
        try:
            if len(close_prices) < max_lags * 2: 
                return 0.50
            lags = np.arange(2, max_lags)
            variances = []
            for lag in lags:
                diffs = close_prices[lag:] - close_prices[:-lag]
                std_dev = np.std(diffs)
                variances.append(std_dev if std_dev > 0 else 1e-6)
            poly = np.polyfit(np.log(lags), np.log(variances), 1)
            return float(np.clip(poly[0] * 2.0, 0.0, 1.0))
        except Exception: 
            return 0.50

    @classmethod
    def evaluate_asset(cls, df: pd.DataFrame) -> dict:
        metrics = {"signal": "HOLD", "alpha_score": 0.0, "atr_pct": 0.01, "price": 0.0}
        try:
            if df is None or len(df) < 30:
                return metrics
            
            col_map = {str(c).lower().strip(): c for c in df.columns}
            close_key = col_map.get('close', col_map.get('price', df.columns[-1]))
            high_key = col_map.get('high', close_key)
            low_key = col_map.get('low', close_key)
            
            closes = df[close_key].to_numpy(dtype=float)
            highs = df[high_key].to_numpy(dtype=float)
            lows = df[low_key].to_numpy(dtype=float)
            
            current_price = closes[-1]
            metrics["price"] = current_price
            prices_series = pd.Series(closes)

            ema_9 = prices_series.ewm(span=9, adjust=False).mean().to_numpy()[-1]
            ema_50 = prices_series.ewm(span=50, adjust=False).mean().to_numpy()[-1]
            ema_100 = prices_series.ewm(span=100, adjust=False).mean().to_numpy()[-1] if len(closes) >= 100 else ema_50

            hl = highs - lows
            hc = np.abs(highs - np.roll(closes, 1))
            lc = np.abs(lows - np.roll(closes, 1))
            hc[0], lc[0] = 0, 0
            atr = pd.Series(np.maximum(hl, np.maximum(hc, lc))).rolling(window=14).mean().to_numpy()[-1]
            if np.isnan(atr) or atr <= 0: atr = current_price * 0.01
            metrics["atr_pct"] = float(atr / current_price)

            momentum = prices_series.pct_change().tail(5).mean()
            if np.isnan(momentum): momentum = 0.0

            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + (gain / (loss + 1e-6))))
            rsi_val = rsi.to_numpy()[-1]
            if np.isnan(rsi_val): rsi_val = 50.0

            hurst_val = cls.calculate_hurst_exponent(closes[-30:])

            if hurst_val > 0.55:  # Trend
                if current_price > ema_100 and momentum > 0 and rsi_val < 75:
                    if current_price > ema_9:
                        metrics["signal"] = "BUY"
                        metrics["alpha_score"] = float(momentum * 100.0)
                elif current_price < ema_50 or rsi_val > 80:
                    metrics["signal"] = "SELL"

            elif hurst_val < 0.45:  # Mean-Reversion
                rolling_mean = prices_series.rolling(window=20).mean().to_numpy()[-1]
                lower_floor = rolling_mean - (1.5 * atr)
                upper_ceiling = rolling_mean + (1.5 * atr)

                if current_price <= lower_floor or rsi_val < 32:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(100.0 - rsi_val)
                elif current_price >= upper_ceiling or rsi_val > 68:
                    metrics["signal"] = "SELL"
            
            else:  # Pivot
                if rsi_val < 26:
                    metrics["signal"] = "BUY"
                    metrics["alpha_score"] = float(50.0 - rsi_val)
                elif rsi_val > 74:
                    metrics["signal"] = "SELL"

            return metrics
        except Exception:
            return metrics

def decide(market_state: dict, portfolio_state: dict, cash: float) -> list:
    try:
        orders = []
        if not market_state:
            return orders

        # 1. Total Portfolio Accounting & Real-Time Exposure Tracking
        total_portfolio_value = float(cash)
        current_gross_exposure = 0.0
        current_positions = {}
        active_prices = {}

        for ticker, p_val in portfolio_state.items():
            qty_held = p_val.get('quantity', p_val.get('qty', 0)) if isinstance(p_val, dict) else p_val
            if qty_held and float(qty_held) > 0:
                current_positions[ticker] = float(qty_held)

        for ticker, bars in market_state.items():
            if bars:
                try:
                    last_bar = bars[-1]
                    price = last_bar.get("close", last_bar.get("price", last_bar.get("open", 0)))
                    active_prices[ticker] = float(price)
                except Exception:
                    continue

        # Calculate exact Net Asset Value (NAV) and starting Gross Exposure
        for ticker, qty in current_positions.items():
            if ticker in active_prices:
                pos_value = qty * active_prices[ticker]
                total_portfolio_value += pos_value
                current_gross_exposure += pos_value

        available_cash = float(cash)
        buy_candidates = []

        # 2. Extract Immediate Closures to Free Up Leverage Space First
        for ticker, bars in market_state.items():
            if not bars or len(bars) < 30:
                continue
            
            try:
                df = pd.DataFrame(bars)
                analysis = InstitutionalAlphaEngine.evaluate_asset(df)
                current_price = active_prices.get(ticker, analysis["price"])
                
                if current_price <= 0:
                    continue

                qty_held = current_positions.get(ticker, 0.0)

                if analysis["signal"] == "SELL" and qty_held > 0:
                    orders.append({"ticker": str(ticker), "side": "sell", "quantity": int(qty_held)})
                    # Credit exposure and cash pools back immediately for the current calculation turn
                    current_gross_exposure -= (qty_held * current_price)
                    available_cash += (qty_held * current_price)
                
                elif analysis["signal"] == "BUY":
                    buy_candidates.append({
                        "ticker": str(ticker),
                        "price": float(current_price),
                        "score": float(analysis["alpha_score"]),
                        "atr_pct": float(analysis["atr_pct"]),
                        "qty_held": float(qty_held)
                    })
            except Exception:
                continue

        # 3. Dynamic Leverage Allocation Layer
        # Hard ceiling: Maximum total exposure allowed across the whole portfolio is 1.42x NAV
        max_absolute_exposure = total_portfolio_value * 1.42
        
        # Sort opportunities by highest calculated Alpha score
        buy_candidates = sorted(buy_candidates, key=lambda x: x["score"], reverse=True)[:5]

        for candidate in buy_candidates:
            try:
                # Continuous real-time check of remaining room below the leverage ceiling
                remaining_leverage_room = max_absolute_exposure - current_gross_exposure
                if remaining_leverage_room <= 0:
                    break

                ticker = candidate["ticker"]
                price = candidate["price"]
                atr_pct = candidate["atr_pct"]
                qty_held = candidate["qty_held"]

                # Inverse-Volatility sizing base metric
                base_allocation = total_portfolio_value * (0.02 / (atr_pct + 1e-5))
                
                # Interlocking Guardrails: Protect cash, individual concentration limits, and overall leverage room
                target_spend = min(base_allocation, total_portfolio_value * 0.15, available_cash * 0.35, remaining_leverage_room)
                max_allowed_spend = (total_portfolio_value * 0.24) - (qty_held * price)
                
                final_spend = min(target_spend, max_allowed_spend)
                if final_spend > 0:
                    quantity = int(final_spend / price)
                    if quantity > 0:
                        orders.append({"ticker": ticker, "side": "buy", "quantity": quantity})
                        available_cash -= (quantity * price)
                        current_gross_exposure += (quantity * price)
            except Exception:
                continue

        return orders

    except Exception:
        return []