"""PBA: トレンド調整アプローチ - 構造的減少への対応"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    RobustBaselineForecaster
)
from scipy import stats, optimize

sns.set_style('whitegrid')


class PBATrendAdjustment:
    """PBA トレンド調整クラス"""

    def __init__(self):
        self.prep = SalesForecastDataPreparation()
        self.trend_params = {}

    def extract_trend(self, train_data: pd.DataFrame, method='linear') -> dict:
        """トレンド抽出"""

        x = np.arange(len(train_data))
        y = train_data['y'].values

        if method == 'linear':
            # 線形トレンド
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            trend_values = slope * x + intercept
            detrended = y - trend_values

            return {
                'method': 'linear',
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_values': trend_values,
                'detrended': detrended
            }

        elif method == 'exponential_decay':
            # 指数減衰トレンド: y = a * exp(b * x) + c
            def exp_func(x, a, b, c):
                return a * np.exp(b * x) + c

            try:
                # 初期パラメータ推定
                p0 = [y[0], -0.01, y.mean()]
                params, _ = optimize.curve_fit(exp_func, x, y, p0=p0, maxfev=10000)

                trend_values = exp_func(x, *params)
                detrended = y - trend_values

                # R²計算
                ss_res = np.sum((y - trend_values) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                return {
                    'method': 'exponential_decay',
                    'params': params,
                    'r_squared': r_squared,
                    'trend_values': trend_values,
                    'detrended': detrended
                }
            except:
                # フォールバック: 線形
                return self.extract_trend(train_data, method='linear')

        elif method == 'polynomial':
            # 2次多項式トレンド
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)

            trend_values = p(x)
            detrended = y - trend_values

            # R²計算
            ss_res = np.sum((y - trend_values) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                'method': 'polynomial',
                'poly_coeffs': z,
                'r_squared': r_squared,
                'trend_values': trend_values,
                'detrended': detrended
            }

    def forecast_with_trend_adjustment(self, train_data: pd.DataFrame, periods: int,
                                       trend_method='linear', forecast_method='ensemble') -> dict:
        """トレンド調整付き予測"""

        print(f"\n{'='*80}")
        print(f"Trend Adjustment Forecast")
        print(f"  Trend method: {trend_method}")
        print(f"  Forecast method: {forecast_method}")
        print(f"{'='*80}")

        # 1. トレンド抽出
        trend_result = self.extract_trend(train_data, method=trend_method)

        print(f"\nTrend Extraction ({trend_method}):")
        if trend_method == 'linear':
            print(f"  Slope: {trend_result['slope']:.2f} units/month ({trend_result['slope']/train_data['y'].mean()*100:.3f}%/month)")
            print(f"  R²: {trend_result['r_squared']:.4f}")
            print(f"  P-value: {trend_result['p_value']:.4e}")
        elif trend_method == 'exponential_decay':
            print(f"  Parameters: a={trend_result['params'][0]:.2f}, b={trend_result['params'][1]:.6f}, c={trend_result['params'][2]:.2f}")
            print(f"  R²: {trend_result['r_squared']:.4f}")
        elif trend_method == 'polynomial':
            print(f"  Coefficients: {trend_result['poly_coeffs']}")
            print(f"  R²: {trend_result['r_squared']:.4f}")

        # 2. 非トレンド成分で予測
        detrended_df = train_data.copy()
        detrended_df['y'] = trend_result['detrended']

        print(f"\nDetrended Data Statistics:")
        print(f"  Mean: {detrended_df['y'].mean():.2f}")
        print(f"  Std: {detrended_df['y'].std():.2f}")
        print(f"  CV: {detrended_df['y'].std() / abs(detrended_df['y'].mean()) * 100:.1f}%")

        # 3. 非トレンド成分の予測
        if forecast_method == 'ensemble':
            # アンサンブル予測
            models = [
                ('MA3', MovingAverageForecaster(window=3)),
                ('MA6', MovingAverageForecaster(window=6)),
                ('MA12', MovingAverageForecaster(window=12)),
                ('RobustMA6', RobustBaselineForecaster(window=6)),
                ('SeasonalNaive', SeasonalNaiveForecaster())
            ]

            # 検証分割
            val_size = min(6, len(detrended_df) // 4)
            train_cv = detrended_df[:-val_size].copy()
            val_cv = detrended_df[-val_size:].copy()

            predictions = []
            model_names = []

            for name, model in models:
                try:
                    model.fit(train_cv)
                    pred = model.predict(len(val_cv))
                    merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')

                    if len(merged) > 0:
                        predictions.append(merged['yhat'].values)
                        model_names.append(name)
                except:
                    pass

            if len(predictions) > 0:
                # 重み最適化（MAE最小化）
                X = np.column_stack(predictions)
                y_true = val_cv['y'].values[:len(predictions[0])]

                def objective(weights):
                    weights = weights / weights.sum()
                    ensemble_pred = X @ weights
                    mae = np.mean(np.abs(y_true - ensemble_pred))
                    return mae

                n_models = len(model_names)
                initial_weights = np.ones(n_models) / n_models
                bounds = [(0, 1) for _ in range(n_models)]
                constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

                result = optimize.minimize(objective, initial_weights, method='SLSQP',
                                          bounds=bounds, constraints=constraints)

                optimal_weights = result.x / result.x.sum()
                weights_dict = {name: weight for name, weight in zip(model_names, optimal_weights)}

                print(f"\nEnsemble Weights:")
                for name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
                    if weight > 0.01:
                        print(f"  {name}: {weight:.3f}")

                # 全データで予測
                ensemble_predictions = []
                for name, model in models:
                    if name in weights_dict and weights_dict[name] > 0.001:
                        try:
                            model.fit(detrended_df)
                            pred = model.predict(periods)
                            ensemble_predictions.append((pred, weights_dict[name]))
                        except:
                            pass

                detrended_forecast = ensemble_predictions[0][0][['ds']].copy()
                detrended_forecast['yhat'] = 0

                for pred, weight in ensemble_predictions:
                    detrended_forecast['yhat'] += pred['yhat'] * weight
            else:
                # フォールバック: 平均
                mean_val = detrended_df['y'].mean()
                detrended_forecast = pd.DataFrame({
                    'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                                       periods=periods, freq='MS'),
                    'yhat': [mean_val] * periods
                })

        else:
            # 単純な手法（RobustMA6）
            model = RobustBaselineForecaster(window=6)
            model.fit(detrended_df)
            detrended_forecast = model.predict(periods)

        # 4. トレンドを戻す
        train_len = len(train_data)
        future_x = np.arange(train_len, train_len + periods)

        if trend_method == 'linear':
            future_trend = trend_result['slope'] * future_x + trend_result['intercept']
        elif trend_method == 'exponential_decay':
            params = trend_result['params']
            future_trend = params[0] * np.exp(params[1] * future_x) + params[2]
        elif trend_method == 'polynomial':
            p = np.poly1d(trend_result['poly_coeffs'])
            future_trend = p(future_x)

        final_forecast = detrended_forecast.copy()
        final_forecast['yhat'] = detrended_forecast['yhat'] + future_trend

        # 負の値を0に制限
        final_forecast['yhat'] = final_forecast['yhat'].clip(lower=0)

        print(f"\nForecast Summary:")
        print(f"  Mean detrended forecast: {detrended_forecast['yhat'].mean():.2f}")
        print(f"  Mean trend component: {future_trend.mean():.2f}")
        print(f"  Mean final forecast: {final_forecast['yhat'].mean():.2f}")

        return {
            'forecast': final_forecast,
            'trend_result': trend_result,
            'detrended_forecast': detrended_forecast
        }


def evaluate_trend_approaches():
    """複数のトレンド調整アプローチを評価"""

    print("\n" + "="*80)
    print("PBA: Trend Adjustment Approaches Evaluation")
    print("="*80)

    prep = SalesForecastDataPreparation()
    train, test = prep.create_category_aggregation('PBA')

    pta = PBATrendAdjustment()

    approaches = [
        ('linear', 'ensemble'),
        ('exponential_decay', 'ensemble'),
        ('polynomial', 'ensemble'),
        ('linear', 'robust'),
    ]

    results = {}

    for trend_method, forecast_method in approaches:
        approach_name = f"{trend_method}_{forecast_method}"

        print(f"\n{'='*80}")
        print(f"Approach: {approach_name}")
        print(f"{'='*80}")

        try:
            result = pta.forecast_with_trend_adjustment(train, len(test),
                                                        trend_method=trend_method,
                                                        forecast_method=forecast_method)

            forecast = result['forecast']

            # 評価
            merged = pd.merge(test[['ds', 'y']], forecast[['ds', 'yhat']],
                             on='ds', how='inner')

            y_true = merged['y'].values
            y_pred = merged['yhat'].values
            mask = y_true > 0

            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

                results[approach_name] = {
                    'mape': mape,
                    'mae': mae,
                    'rmse': rmse,
                    'forecast': forecast,
                    'trend_result': result['trend_result']
                }

                print(f"\nResults:")
                print(f"  MAPE: {mape:.2f}%")
                print(f"  MAE:  {mae:.2f}")
                print(f"  RMSE: {rmse:.2f}")

        except Exception as e:
            print(f"\nERROR: {e}")
            continue

    # 比較
    print(f"\n{'='*80}")
    print(f"Comparison of All Approaches")
    print(f"{'='*80}")

    print(f"\nBaseline: 26.71% MAPE (RobustMA6)")
    print(f"Series-level ensemble: 32.11% MAPE")
    print(f"\nTrend Adjustment Approaches:")

    for name, result in sorted(results.items(), key=lambda x: x[1]['mape']):
        mape = result['mape']
        improvement = 26.71 - mape
        print(f"  {name}: {mape:.2f}% MAPE ({improvement:+.2f}% points)")

    # 最良の手法
    if results:
        best_name = min(results, key=lambda x: results[x]['mape'])
        best_result = results[best_name]

        print(f"\nBest Approach: {best_name}")
        print(f"  MAPE: {best_result['mape']:.2f}%")
        print(f"  Improvement: {26.71 - best_result['mape']:+.2f}% points")

        # 可視化
        visualize_best_approach(train, test, best_name, best_result)

        return best_result['mape'], best_name

    return None, None


def visualize_best_approach(train, test, approach_name, result):
    """最良アプローチの可視化"""

    print(f"\nCreating visualization for: {approach_name}")

    forecast = result['forecast']
    trend_result = result['trend_result']

    merged = pd.merge(test[['ds', 'y']], forecast[['ds', 'yhat']],
                     on='ds', how='inner')

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # グラフ1: 予測 vs 実績
    ax = axes[0]
    ax.plot(train['ds'], train['y'], 'o-', label='Training Data',
           linewidth=2, markersize=3, color='blue')
    ax.plot(test['ds'], test['y'], 'o-', label='Actual Test',
           linewidth=2, markersize=6, color='red', zorder=5)
    ax.plot(merged['ds'], merged['yhat'], '^--', label=f'Forecast ({result["mape"]:.2f}% MAPE)',
           linewidth=2, markersize=6, color='green', alpha=0.8)

    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle='--',
              linewidth=2, label='Train/Test Split')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'PBA: Trend Adjustment Forecast - {approach_name}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: トレンド成分
    ax = axes[1]
    full_data = pd.concat([train, test], ignore_index=True)
    x_all = np.arange(len(full_data))

    if trend_result['method'] == 'linear':
        trend_all = trend_result['slope'] * x_all + trend_result['intercept']
    elif trend_result['method'] == 'exponential_decay':
        params = trend_result['params']
        trend_all = params[0] * np.exp(params[1] * x_all) + params[2]
    elif trend_result['method'] == 'polynomial':
        p = np.poly1d(trend_result['poly_coeffs'])
        trend_all = p(x_all)

    ax.plot(full_data['ds'], full_data['y'].values, 'o-', label='Actual Data',
           linewidth=2, markersize=3, color='blue', alpha=0.5)
    ax.plot(full_data['ds'], trend_all, '--', label=f'Trend Component (R²={trend_result["r_squared"]:.3f})',
           linewidth=3, color='red', alpha=0.8)

    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle='--',
              linewidth=2)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'Trend Component ({trend_result["method"]})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ3: 誤差率
    ax = axes[2]
    merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100

    colors = ['green' if abs(e) < 20 else 'orange' if abs(e) < 30 else 'red'
              for e in merged['pct_error']]
    ax.bar(merged['ds'], merged['pct_error'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.5, label='±20%')
    ax.axhline(y=-20, color='green', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Error by Month', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/pba_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'pba_trend_adjustment.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir / 'pba_trend_adjustment.png'}")
    plt.close()


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PBA: Trend Adjustment Evaluation")
    print("="*80)

    mape, best_approach = evaluate_trend_approaches()

    if mape:
        print(f"\n{'='*80}")
        print(f"Final Result: {best_approach}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Improvement vs baseline: {26.71 - mape:+.2f}% points")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
