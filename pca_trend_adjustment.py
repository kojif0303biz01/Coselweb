"""PCAベースライン予測へのトレンド調整実装"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import MovingAverageForecaster, RobustBaselineForecaster

sns.set_style('whitegrid')


class TrendAdjustedForecaster:
    """トレンド調整付きベースライン予測クラス"""

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name
        self.prep = SalesForecastDataPreparation()

    def detect_series_spikes(self, series_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """シリーズごとにスパイクを検出"""
        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        if len(monthly) >= 12:
            is_spike = self.prep.detect_spikes(monthly['y'], method='combined',
                                               iqr_multiplier=2.0, z_threshold=2.5)
            monthly['is_spike'] = is_spike
            monthly['y_baseline'] = monthly['y'].copy()
            monthly.loc[is_spike, 'y_baseline'] = np.nan
        else:
            monthly['is_spike'] = False
            monthly['y_baseline'] = monthly['y']

        return monthly

    def extract_trend(self, train_monthly: pd.DataFrame) -> dict:
        """ベースラインデータから線形トレンドを抽出"""

        # スパイク除外データ
        baseline_data = train_monthly[~train_monthly['is_spike']].copy()

        if len(baseline_data) < 6:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}

        # 時間インデックス（0, 1, 2, ...）
        x = np.arange(len(baseline_data))
        y = baseline_data['y_baseline'].values

        # 欠損値を除外
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            return {'slope': 0, 'intercept': 0, 'r_squared': 0}

        x_clean = x[mask]
        y_clean = y[mask]

        # 線形回帰
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'last_x': len(baseline_data) - 1
        }

    def forecast_with_trend(self, series_name: str, train_monthly: pd.DataFrame,
                           periods: int, use_trend: bool = True) -> tuple:
        """トレンド調整付き予測"""

        # スパイク除外データ
        baseline_data = train_monthly[~train_monthly['is_spike']].copy()

        if len(baseline_data) < 6:
            return None, None, None

        # ベースライン時系列データ
        baseline_ts = pd.DataFrame({
            'ds': baseline_data['ds'],
            'y': baseline_data['y_baseline']
        }).dropna()

        if len(baseline_ts) < 3:
            return None, None, None

        # トレンド抽出
        trend_info = self.extract_trend(train_monthly)

        # ベースモデルで予測（トレンド除去データ）
        if use_trend and trend_info['r_squared'] > 0.1:  # R² > 0.1の場合のみトレンド調整
            # トレンド除去
            x = np.arange(len(baseline_ts))
            trend_line = trend_info['slope'] * x + trend_info['intercept']
            detrended_y = baseline_ts['y'].values - trend_line

            detrended_ts = pd.DataFrame({
                'ds': baseline_ts['ds'],
                'y': detrended_y
            })

            # モデル選択
            best_model = self._select_best_model(detrended_ts)
            best_model.fit(detrended_ts)
            forecast = best_model.predict(periods)

            # トレンド成分を追加
            last_x = trend_info['last_x']
            future_x = np.arange(last_x + 1, last_x + 1 + periods)
            future_trend = trend_info['slope'] * future_x + trend_info['intercept']

            forecast['yhat'] = forecast['yhat'] + future_trend
            forecast['yhat'] = forecast['yhat'].clip(lower=0)  # 負の値を0にクリップ

            model_name = f"{best_model.__class__.__name__}+Trend"
        else:
            # トレンド調整なし
            best_model = self._select_best_model(baseline_ts)
            best_model.fit(baseline_ts)
            forecast = best_model.predict(periods)
            model_name = best_model.__class__.__name__

        return forecast, model_name, trend_info

    def _select_best_model(self, data: pd.DataFrame) -> object:
        """最適モデルを選択"""

        models_to_test = [
            MovingAverageForecaster(window=3),
            MovingAverageForecaster(window=6),
            RobustBaselineForecaster(window=6)
        ]

        val_size = min(6, len(data) // 4)
        if val_size == 0:
            return RobustBaselineForecaster(window=min(6, len(data) // 2))

        train_cv = data[:-val_size].copy()
        val_cv = data[-val_size:].copy()

        best_model = None
        best_mape = float('inf')

        for model in models_to_test:
            try:
                model.fit(train_cv)
                pred = model.predict(len(val_cv))
                merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')

                if len(merged) > 0:
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    mask = (y_true != 0) & (~np.isnan(y_true))

                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        if mape < best_mape:
                            best_mape = mape
                            best_model = model
            except:
                continue

        return best_model if best_model is not None else RobustBaselineForecaster(window=6)

    def hierarchical_trend_adjusted_forecast(self, periods: int = 12, use_trend: bool = True) -> dict:
        """階層的トレンド調整予測"""

        print(f"\n{'='*80}")
        print(f"Trend-Adjusted Baseline Forecast: {self.sheet_name}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(self.sheet_name)
        train_df = df[df['年月'] <= '2024-05-31'].copy()
        test_df = df[df['年月'] > '2024-05-31'].copy()

        # 上位シリーズ
        ranking_path = Path('data/processed') / f'{self.sheet_name}_series_ranking.csv'
        series_ranking = pd.read_csv(ranking_path)
        top_series = series_ranking[series_ranking['cumulative_ratio'] <= 80]['モデル名'].tolist()

        print(f"\nForecasting {len(top_series)} top series with trend adjustment...")

        test_dates = sorted(test_df['年月'].unique())[:periods]

        series_forecasts = []
        for i, series in enumerate(top_series, 1):
            print(f"\n  [{i}/{len(top_series)}] {series}")

            # スパイク検出
            series_monthly = self.detect_series_spikes(series, train_df)
            train_series = series_monthly[series_monthly['ds'] <= '2024-05-31']

            # トレンド調整予測
            forecast, model_name, trend_info = self.forecast_with_trend(
                series, train_series, periods, use_trend=use_trend
            )

            if forecast is not None and trend_info is not None:
                print(f"    Model: {model_name}")
                print(f"    Trend: slope={trend_info['slope']:.2f}/month, R²={trend_info['r_squared']:.3f}")

                series_forecasts.append({
                    'series': series,
                    'forecast': forecast,
                    'model': model_name,
                    'trend_info': trend_info
                })
            else:
                print(f"    SKIP")

        # 集計
        aggregated = self._aggregate_forecasts(series_forecasts, test_dates)

        # Long Tail
        longtail_series = [s for s in df['モデル名'].unique() if s not in top_series]
        longtail_df = train_df[train_df['モデル名'].isin(longtail_series)]
        longtail_monthly = longtail_df.groupby('年月')['台数'].sum().reset_index()
        longtail_monthly.columns = ['ds', 'y']

        if len(longtail_monthly) > 0:
            longtail_spikes = self.prep.detect_spikes(longtail_monthly['y'])
            longtail_baseline = longtail_monthly.loc[~longtail_spikes, 'y']
            longtail_avg = longtail_baseline.mean() if len(longtail_baseline) > 0 else longtail_monthly['y'].mean()

            aggregated['yhat'] = aggregated['yhat'] + longtail_avg
            print(f"\nLong Tail contribution: {longtail_avg:.0f} units/month")

        return {
            'forecast': aggregated,
            'series_details': series_forecasts
        }

    def _aggregate_forecasts(self, series_forecasts: list, test_dates: list) -> pd.DataFrame:
        """予測を集計"""
        aggregated = []
        for date in test_dates:
            total = 0
            for sf in series_forecasts:
                date_pred = sf['forecast'][sf['forecast']['ds'] == date]['yhat'].values
                if len(date_pred) > 0:
                    total += date_pred[0]
            aggregated.append({'ds': date, 'yhat': total})
        return pd.DataFrame(aggregated)


def compare_trend_adjustment(sheet_name: str = 'PCA'):
    """トレンド調整あり/なしを比較"""

    print(f"\n{'='*80}")
    print(f"Comparing: Baseline vs Trend-Adjusted Forecast")
    print(f"{'='*80}")

    # トレンド調整なし
    print("\n[1] Without Trend Adjustment")
    taf = TrendAdjustedForecaster(sheet_name)
    result_no_trend = taf.hierarchical_trend_adjusted_forecast(periods=12, use_trend=False)

    # トレンド調整あり
    print("\n[2] With Trend Adjustment")
    result_with_trend = taf.hierarchical_trend_adjusted_forecast(periods=12, use_trend=True)

    # 実績データ
    prep = SalesForecastDataPreparation()
    df = prep.load_data(sheet_name)
    test_df = df[df['年月'] > '2024-05-31'].copy()
    test_monthly = test_df.groupby('年月')['台数'].sum().reset_index()
    test_monthly.columns = ['ds', 'y']
    test_monthly = test_monthly.sort_values('ds').head(12).reset_index(drop=True)

    # スパイク検出
    test_spikes = prep.detect_spikes(test_monthly['y'])
    test_monthly['is_spike'] = test_spikes

    # 評価
    results = {}
    for name, result in [('No Trend', result_no_trend), ('With Trend', result_with_trend)]:
        merged = pd.merge(test_monthly[['ds', 'y', 'is_spike']],
                         result['forecast'][['ds', 'yhat']], on='ds', how='inner')

        # ベースラインのみ（スパイク除外）
        baseline_only = merged[~merged['is_spike']]
        if len(baseline_only) > 0:
            y_true = baseline_only['y'].values
            y_pred = baseline_only['yhat'].values
            mask = y_true > 0

            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
                results[name] = {'mape': mape, 'mae': mae, 'rmse': rmse, 'merged': merged}

    # 結果表示
    print(f"\n{'='*80}")
    print("Evaluation Results (Baseline-only, Spike-excluded)")
    print(f"{'='*80}")

    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  MAPE: {res['mape']:.2f}%")
        print(f"  MAE:  {res['mae']:.2f}")
        print(f"  RMSE: {res['rmse']:.2f}")

    if 'No Trend' in results and 'With Trend' in results:
        improvement = results['No Trend']['mape'] - results['With Trend']['mape']
        print(f"\nImprovement from trend adjustment: {improvement:+.2f}% points")

    # 可視化
    fig, ax = plt.subplots(figsize=(14, 7))

    merged_no_trend = results['No Trend']['merged']
    merged_with_trend = results['With Trend']['merged']

    ax.plot(merged_no_trend['ds'], merged_no_trend['y'], 'o-',
           label='Actual', linewidth=2, markersize=8, color='black', zorder=5)

    ax.plot(merged_no_trend['ds'], merged_no_trend['yhat'], 's--',
           label='Forecast (No Trend)', linewidth=2, markersize=6, color='blue', alpha=0.7)

    ax.plot(merged_with_trend['ds'], merged_with_trend['yhat'], '^--',
           label='Forecast (With Trend)', linewidth=2, markersize=6, color='green', alpha=0.7)

    # スパイク月を強調
    spike_months = merged_no_trend[merged_no_trend['is_spike']]
    if len(spike_months) > 0:
        ax.scatter(spike_months['ds'], spike_months['y'], s=200, facecolors='none',
                  edgecolors='red', linewidths=3, label='Spike Months', zorder=7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'{sheet_name}: Baseline vs Trend-Adjusted Forecast',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_dir = Path('visualizations/pca_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / f'{sheet_name}_trend_adjustment_comparison.png',
               dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / f'{sheet_name}_trend_adjustment_comparison.png'}")
    plt.close()

    return results


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PCA: Trend-Adjusted Forecast Evaluation")
    print("="*80)

    results = compare_trend_adjustment('PCA')

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"Previous baseline (spike-separated): 43.37% MAPE")

    if 'With Trend' in results:
        print(f"Trend-adjusted forecast: {results['With Trend']['mape']:.2f}% MAPE")
        improvement = 43.37 - results['With Trend']['mape']
        print(f"Total improvement: {improvement:+.2f}% points")


if __name__ == '__main__':
    main()
