"""PBA: シリーズレベルトレンド調整 - 最終最適化"""

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


class PBASeriesTrendAdjustment:
    """PBA シリーズレベルトレンド調整クラス"""

    def __init__(self):
        self.prep = SalesForecastDataPreparation()
        self.series_forecasts = {}
        self.series_trend_params = {}
        self.series_results = {}

    def get_series_data(self, df: pd.DataFrame, series_name: str, train_end: str = '2024-05-31'):
        """シリーズデータを取得"""

        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        train = monthly[monthly['ds'] <= train_end][['ds', 'y']].copy()
        test = monthly[monthly['ds'] > train_end][['ds', 'y']].copy()

        return train, test

    def forecast_series_with_trend(self, train_data: pd.DataFrame, periods: int, series_name: str) -> dict:
        """シリーズごとのトレンド調整予測"""

        print(f"\n[{series_name}]", end=" ")

        if len(train_data) < 6:
            print("Insufficient data, using mean")
            mean_val = max(train_data['y'].mean(), 0)
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                                   periods=periods, freq='MS'),
                'yhat': [mean_val] * periods
            })
            return {
                'forecast': forecast_df,
                'trend_slope': 0,
                'r_squared': 0
            }

        # 1. トレンド抽出
        x = np.arange(len(train_data))
        y = train_data['y'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        trend_values = slope * x + intercept
        detrended = y - trend_values

        # 2. 非トレンド成分で予測（シンプルなアプローチ）
        detrended_df = train_data.copy()
        detrended_df['y'] = detrended

        # モデル選択: MA3/6/12, RobustMA6 の中から最良を選択
        models = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('RobustMA6', RobustBaselineForecaster(window=6)),
        ]

        best_model = None
        best_mae = float('inf')

        # 簡易検証（最後3か月）
        if len(detrended_df) >= 6:
            val_size = min(3, len(detrended_df) // 4)
            train_cv = detrended_df[:-val_size].copy()
            val_cv = detrended_df[-val_size:].copy()

            for name, model in models:
                try:
                    if hasattr(model, 'window') and len(train_cv) < model.window:
                        continue

                    model.fit(train_cv)
                    pred = model.predict(len(val_cv))
                    merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')

                    if len(merged) > 0:
                        mae = np.mean(np.abs(merged['y'] - merged['yhat']))
                        if mae < best_mae:
                            best_mae = mae
                            best_model = (name, model)
                except:
                    pass

        if best_model is None:
            best_model = ('RobustMA6', RobustBaselineForecaster(window=6))

        # 全データで予測
        try:
            model_name, model = best_model
            model.fit(detrended_df)
            detrended_forecast = model.predict(periods)
        except:
            # フォールバック: 平均
            mean_val = detrended_df['y'].mean()
            detrended_forecast = pd.DataFrame({
                'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                                   periods=periods, freq='MS'),
                'yhat': [mean_val] * periods
            })
            model_name = 'Mean'

        # 3. トレンドを戻す
        train_len = len(train_data)
        future_x = np.arange(train_len, train_len + periods)
        future_trend = slope * future_x + intercept

        final_forecast = detrended_forecast.copy()
        final_forecast['yhat'] = detrended_forecast['yhat'] + future_trend

        # 負の値を0に制限
        final_forecast['yhat'] = final_forecast['yhat'].clip(lower=0)

        print(f"(R²={r_squared:.3f}, slope={slope:.1f}/mo, model={model_name})")

        return {
            'forecast': final_forecast,
            'trend_slope': slope,
            'r_squared': r_squared,
            'model': model_name
        }

    def hierarchical_forecast(self, sheet_name: str = 'PBA', periods: int = 12) -> pd.DataFrame:
        """シリーズレベル階層的予測（トレンド調整付き）"""

        print(f"\n{'='*80}")
        print(f"PBA: Series-Level Hierarchical Forecast with Trend Adjustment")
        print(f"{'='*80}")

        df = self.prep.load_data(sheet_name)
        series_list = sorted(df['モデル名'].unique())

        print(f"\nForecasting {len(series_list)} series...")

        all_forecasts = []

        for series_name in series_list:
            train, test = self.get_series_data(df, series_name)

            if len(train) < 3:
                print(f"\n[{series_name}] SKIP: Insufficient data")
                continue

            result = self.forecast_series_with_trend(train, periods, series_name)

            self.series_forecasts[series_name] = result['forecast']
            self.series_trend_params[series_name] = {
                'slope': result['trend_slope'],
                'r_squared': result['r_squared'],
                'model': result['model']
            }
            self.series_results[series_name] = {
                'train_mean': train['y'].mean(),
                'test_mean': test['y'].mean() if len(test) > 0 else 0,
                'trend_slope': result['trend_slope'],
                'r_squared': result['r_squared']
            }

            all_forecasts.append(result['forecast'])

        # カテゴリレベルに集約
        if len(all_forecasts) > 0:
            category_forecast = all_forecasts[0][['ds']].copy()
            category_forecast['yhat'] = 0

            for forecast_df in all_forecasts:
                category_forecast['yhat'] += forecast_df['yhat']
        else:
            category_forecast = pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"Category-Level Forecast Complete")
        print(f"{'='*80}")

        return category_forecast


def evaluate_series_trend_adjustment():
    """シリーズレベルトレンド調整の評価"""

    print(f"\n{'='*80}")
    print(f"PBA: Series-Level Trend Adjustment Evaluation")
    print(f"{'='*80}")

    psta = PBASeriesTrendAdjustment()
    forecast = psta.hierarchical_forecast('PBA', periods=12)

    if len(forecast) == 0:
        print("ERROR: No forecast generated")
        return None

    # 実績データ取得
    prep = SalesForecastDataPreparation()
    _, test_monthly = prep.create_category_aggregation('PBA')

    # 評価
    merged = pd.merge(test_monthly[['ds', 'y']], forecast[['ds', 'yhat']],
                     on='ds', how='inner')

    y_true = merged['y'].values
    y_pred = merged['yhat'].values
    mask = y_true > 0

    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
        rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

        print(f"\n{'='*80}")
        print(f"Evaluation Results")
        print(f"{'='*80}")
        print(f"MAPE: {mape:.2f}%")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

        print(f"\nComparison:")
        print(f"  Baseline (RobustMA6): 26.71% MAPE")
        print(f"  Series-level ensemble: 32.11% MAPE")
        print(f"  Category-level trend adjustment: 17.64% MAPE")
        print(f"  Series-level trend adjustment: {mape:.2f}% MAPE")

        if mape < 17.64:
            improvement = 17.64 - mape
            print(f"  Additional improvement: {improvement:+.2f}% points ✓")
        else:
            print(f"  Change: {17.64 - mape:+.2f}% points")

        total_improvement = 26.71 - mape
        print(f"  Total improvement from baseline: {total_improvement:+.2f}% points")

        # 月別詳細
        merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100
        merged['abs_pct_error'] = np.abs(merged['pct_error'])

        print(f"\n{'='*80}")
        print(f"Month-by-Month Performance")
        print(f"{'='*80}")
        print(merged[['ds', 'y', 'yhat', 'pct_error']].to_string(index=False))

        # シリーズ別サマリー
        print(f"\n{'='*80}")
        print(f"Series Trend Summary")
        print(f"{'='*80}")

        series_summary = []
        for series_name, result in psta.series_results.items():
            series_summary.append({
                'Series': series_name,
                'Train_Mean': result['train_mean'],
                'Test_Mean': result['test_mean'],
                'Trend_Slope': result['trend_slope'],
                'R²': result['r_squared'],
                'Change%': ((result['test_mean'] - result['train_mean']) / result['train_mean'] * 100) if result['train_mean'] > 0 else 0
            })

        summary_df = pd.DataFrame(series_summary)
        summary_df = summary_df.sort_values('Train_Mean', ascending=False)
        print(summary_df.to_string(index=False))

        # 可視化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # グラフ1: 予測 vs 実績
        ax = axes[0]
        ax.plot(merged['ds'], merged['y'], 'o-', label='Actual',
               linewidth=2, markersize=8, color='black', zorder=5)
        ax.plot(merged['ds'], merged['yhat'], '^--', label=f'Series-Level Trend Adj. (MAPE: {mape:.2f}%)',
               linewidth=2, markersize=6, color='darkgreen', alpha=0.8)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
        ax.set_title(f'PBA: Series-Level Trend Adjustment Forecast (MAPE: {mape:.2f}%)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # グラフ2: 誤差率
        ax = axes[1]
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
        plt.savefig(save_dir / 'pba_series_trend_adjustment.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_dir / 'pba_series_trend_adjustment.png'}")
        plt.close()

        return mape

    return None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PBA: Series-Level Trend Adjustment")
    print("="*80)

    mape = evaluate_series_trend_adjustment()

    if mape:
        print(f"\n{'='*80}")
        print(f"Final Result: {mape:.2f}% MAPE")
        print(f"Total improvement: {26.71 - mape:+.2f}% points")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
