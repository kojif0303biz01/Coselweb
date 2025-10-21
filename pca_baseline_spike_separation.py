"""PCAシリーズレベルのスパイク除外ベースライン予測"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import RobustBaselineForecaster, MovingAverageForecaster

sns.set_style('whitegrid')


class BaselineSpikeSeparationForecaster:
    """ベースライン需要とスパイクを分離して予測するクラス"""

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name
        self.prep = SalesForecastDataPreparation()
        self.series_baselines = {}
        self.series_spikes = {}

    def detect_series_spikes(self, series_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """シリーズごとにスパイクを検出"""

        # シリーズデータ抽出
        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        # スパイク検出
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

    def forecast_baseline(self, series_name: str, train_monthly: pd.DataFrame, periods: int) -> tuple:
        """ベースライン需要のみを予測"""

        # スパイク除外データで予測
        baseline_data = train_monthly[~train_monthly['is_spike']].copy()

        if len(baseline_data) < 6:
            print(f"    {series_name}: Insufficient baseline data")
            return None, None, 0

        # ベースラインデータで時系列作成
        baseline_ts = pd.DataFrame({
            'ds': baseline_data['ds'],
            'y': baseline_data['y_baseline']
        })

        # 欠損値を前方補完
        baseline_ts = baseline_ts.sort_values('ds').reset_index(drop=True)

        # 最適モデル選択（簡易CV）
        best_model = None
        best_mape = float('inf')
        best_name = None

        models_to_test = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('RobustMA6', RobustBaselineForecaster(window=6))
        ]

        # 簡易CV（最後の6か月を検証）
        val_size = min(6, len(baseline_ts) // 4)
        if val_size > 0:
            train_cv = baseline_ts[:-val_size].copy()
            val_cv = baseline_ts[-val_size:].copy()

            for name, model in models_to_test:
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
                                best_model = model.__class__
                                best_name = name
                except:
                    continue

        # 最適モデルで全訓練データを学習
        if best_model is not None:
            if best_name == 'MA3':
                final_model = MovingAverageForecaster(window=3)
            elif best_name == 'MA6':
                final_model = MovingAverageForecaster(window=6)
            else:
                final_model = RobustBaselineForecaster(window=6)
                best_name = 'RobustMA6'
        else:
            final_model = RobustBaselineForecaster(window=6)
            best_name = 'RobustMA6'

        final_model.fit(baseline_ts)
        forecast = final_model.predict(periods)

        spike_count = train_monthly['is_spike'].sum()

        return forecast, best_name, spike_count

    def hierarchical_baseline_forecast(self, periods: int = 12) -> dict:
        """階層的ベースライン予測（スパイク除外）"""

        print(f"\n{'='*80}")
        print(f"Baseline Forecast with Spike Separation: {self.sheet_name}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(self.sheet_name)
        train_df = df[df['年月'] <= '2024-05-31'].copy()
        test_df = df[df['年月'] > '2024-05-31'].copy()

        # 上位シリーズリスト
        ranking_path = Path('data/processed') / f'{self.sheet_name}_series_ranking.csv'
        series_ranking = pd.read_csv(ranking_path)
        top_series = series_ranking[series_ranking['cumulative_ratio'] <= 80]['モデル名'].tolist()

        print(f"\nForecasting {len(top_series)} top series with spike separation...")

        test_dates = sorted(test_df['年月'].unique())[:periods]

        # 各シリーズの予測
        series_forecasts = []
        for i, series in enumerate(top_series, 1):
            print(f"\n  [{i}/{len(top_series)}] {series}")

            # スパイク検出
            series_monthly = self.detect_series_spikes(series, train_df)
            train_series = series_monthly[series_monthly['ds'] <= '2024-05-31']
            test_series = self.detect_series_spikes(series, test_df)

            spike_count = train_series['is_spike'].sum()
            spike_ratio = (spike_count / len(train_series)) * 100 if len(train_series) > 0 else 0

            print(f"    Spikes detected: {spike_count}/{len(train_series)} ({spike_ratio:.1f}%)")

            # ベースライン予測
            forecast, model_name, _ = self.forecast_baseline(series, train_series, periods)

            if forecast is not None:
                print(f"    Model selected: {model_name}")
                series_forecasts.append({
                    'series': series,
                    'forecast': forecast,
                    'model': model_name,
                    'train_data': train_series,
                    'test_data': test_series,
                    'spike_count': spike_count,
                    'spike_ratio': spike_ratio
                })
            else:
                print(f"    SKIP")

        # 予測を集計
        aggregated = self._aggregate_baseline_forecasts(series_forecasts, test_dates)

        # Long Tail追加
        longtail_series = [s for s in df['モデル名'].unique() if s not in top_series]
        longtail_df = train_df[train_df['モデル名'].isin(longtail_series)]

        # Long TailもスパイクGET除外して平均
        longtail_monthly = longtail_df.groupby('年月')['台数'].sum().reset_index()
        longtail_monthly.columns = ['ds', 'y']
        if len(longtail_monthly) > 0:
            longtail_spikes = self.prep.detect_spikes(longtail_monthly['y'])
            longtail_baseline = longtail_monthly.loc[~longtail_spikes, 'y']
            longtail_avg = longtail_baseline.mean() if len(longtail_baseline) > 0 else longtail_monthly['y'].mean()

            aggregated['yhat'] = aggregated['yhat'] + longtail_avg
            print(f"\nLong Tail contribution (baseline avg): {longtail_avg:.0f} units/month")

        return {
            'forecast': aggregated,
            'series_details': series_forecasts
        }

    def _aggregate_baseline_forecasts(self, series_forecasts: list, test_dates: list) -> pd.DataFrame:
        """ベースライン予測を集計"""

        aggregated = []
        for date in test_dates:
            total = 0
            for sf in series_forecasts:
                date_pred = sf['forecast'][sf['forecast']['ds'] == date]['yhat'].values
                if len(date_pred) > 0:
                    total += date_pred[0]

            aggregated.append({'ds': date, 'yhat': total})

        return pd.DataFrame(aggregated)


def evaluate_baseline_accuracy(sheet_name: str):
    """ベースライン同士の精度評価"""

    print(f"\n{'='*80}")
    print(f"Baseline vs Baseline Accuracy Evaluation: {sheet_name}")
    print(f"{'='*80}")

    bsf = BaselineSpikeSeparationForecaster(sheet_name)
    result = bsf.hierarchical_baseline_forecast(periods=12)

    # 実績データ（スパイク除外）取得
    prep = SalesForecastDataPreparation()
    df = prep.load_data(sheet_name)
    test_df = df[df['年月'] > '2024-05-31'].copy()

    # テスト期間の月次集計とスパイク検出
    test_monthly = test_df.groupby('年月')['台数'].sum().reset_index()
    test_monthly.columns = ['ds', 'y']
    test_monthly = test_monthly.sort_values('ds').head(12).reset_index(drop=True)

    # テスト期間のスパイク検出
    test_spikes = prep.detect_spikes(test_monthly['y'])
    test_monthly['is_spike'] = test_spikes
    test_monthly['y_baseline'] = test_monthly['y'].copy()
    test_monthly.loc[test_spikes, 'y_baseline'] = np.nan

    # 予測とマージ
    forecast_df = result['forecast']
    merged = pd.merge(test_monthly[['ds', 'y', 'y_baseline', 'is_spike']],
                     forecast_df[['ds', 'yhat']], on='ds', how='inner')

    # ベースライン同士の比較（スパイク除外）
    baseline_only = merged[~merged['is_spike']].copy()

    if len(baseline_only) > 0:
        y_true = baseline_only['y_baseline'].values
        y_pred = baseline_only['yhat'].values

        mask = (y_true > 0) & (~np.isnan(y_true))
        if mask.sum() > 0:
            mape_baseline = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            rmse_baseline = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
            mae_baseline = np.mean(np.abs(y_true[mask] - y_pred[mask]))

            print(f"\n【Baseline vs Baseline (Spike-excluded)】")
            print(f"Non-spike months: {mask.sum()}/{len(merged)}")
            print(f"MAPE: {mape_baseline:.2f}%")
            print(f"MAE:  {mae_baseline:.2f}")
            print(f"RMSE: {rmse_baseline:.2f}")

    # 全期間比較（スパイク含む）
    y_true_all = merged['y'].values
    y_pred_all = merged['yhat'].values
    mask_all = y_true_all > 0

    if mask_all.sum() > 0:
        mape_all = np.mean(np.abs((y_true_all[mask_all] - y_pred_all[mask_all]) / y_true_all[mask_all])) * 100
        rmse_all = np.sqrt(np.mean((y_true_all[mask_all] - y_pred_all[mask_all]) ** 2))
        mae_all = np.mean(np.abs(y_true_all[mask_all] - y_pred_all[mask_all]))

        print(f"\n【All Data (Spike-included)】")
        print(f"MAPE: {mape_all:.2f}%")
        print(f"MAE:  {mae_all:.2f}")
        print(f"RMSE: {rmse_all:.2f}")

    # 詳細表示
    print(f"\n【Month-by-Month Detail】")
    print(merged[['ds', 'y', 'y_baseline', 'yhat', 'is_spike']].to_string(index=False))

    # 可視化
    fig, ax = plt.subplots(figsize=(14, 7))

    # 実績（全データ）
    ax.plot(merged['ds'], merged['y'], 'o-', label='Actual (All)',
           linewidth=2, markersize=8, color='black', zorder=5)

    # 実績（ベースラインのみ）
    baseline_only_plot = merged[~merged['is_spike']]
    ax.plot(baseline_only_plot['ds'], baseline_only_plot['y_baseline'], 's',
           label='Actual (Baseline Only)', markersize=10, color='blue', zorder=6)

    # 予測
    ax.plot(merged['ds'], merged['yhat'], '^--', label='Forecast (Baseline)',
           linewidth=2, markersize=6, color='green', alpha=0.8)

    # スパイク月を強調
    spike_months = merged[merged['is_spike']]
    ax.scatter(spike_months['ds'], spike_months['y'], s=200, facecolors='none',
              edgecolors='red', linewidths=3, label='Spike Months', zorder=7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'{sheet_name}: Baseline Forecast (Spike-Separated)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_dir = Path('visualizations/pca_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / f'{sheet_name}_baseline_spike_separated.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / f'{sheet_name}_baseline_spike_separated.png'}")
    plt.close()

    return {
        'baseline_mape': mape_baseline if len(baseline_only) > 0 else None,
        'all_mape': mape_all if mask_all.sum() > 0 else None,
        'merged': merged
    }


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PCA: Baseline Forecast with Spike Separation")
    print("="*80)

    result = evaluate_baseline_accuracy('PCA')

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    if result['baseline_mape'] is not None:
        print(f"Baseline vs Baseline MAPE: {result['baseline_mape']:.2f}%")
    if result['all_mape'] is not None:
        print(f"All Data MAPE (with spikes): {result['all_mape']:.2f}%")

    print(f"\nPrevious Series-Level MAPE (with spikes): 45.06%")

    if result['baseline_mape'] is not None:
        improvement = 45.06 - result['baseline_mape']
        print(f"Improvement from spike separation: {improvement:.2f}% points")


if __name__ == '__main__':
    main()
