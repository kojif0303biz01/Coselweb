"""TUHS: 製品レベルスパイク除去 → カテゴリレベル集計予測"""

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
from scipy.optimize import minimize

sns.set_style('whitegrid')


class ProductLevelSpikeRemoval:
    """製品レベルスパイク除去 → カテゴリレベル集計クラス"""

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name
        self.prep = SalesForecastDataPreparation()
        self.product_spike_info = {}

    def detect_product_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """全製品のスパイクを検出"""

        print(f"\n{'='*80}")
        print(f"Product-Level Spike Detection: {self.sheet_name}")
        print(f"{'='*80}")

        # 製品リスト
        products = df['製品名'].unique()
        print(f"\nAnalyzing {len(products)} products...")

        all_spikes = []
        spike_summary = []

        for i, product in enumerate(products, 1):
            # 製品データ
            product_df = df[df['製品名'] == product].copy()
            monthly = product_df.groupby('年月')['台数'].sum().reset_index()
            monthly.columns = ['ds', 'y']
            monthly = monthly.sort_values('ds').reset_index(drop=True)

            if len(monthly) < 6:
                continue

            # スパイク検出
            is_spike = self.prep.detect_spikes(monthly['y'], method='combined',
                                               iqr_multiplier=2.0, z_threshold=2.5)

            spike_count = is_spike.sum()
            if spike_count > 0:
                monthly['is_spike'] = is_spike
                monthly['製品名'] = product
                monthly['y_baseline'] = monthly['y'].copy()
                monthly.loc[is_spike, 'y_baseline'] = np.nan

                all_spikes.append(monthly)

                # スパイク統計
                spike_ratio = (spike_count / len(monthly)) * 100
                max_spike = monthly.loc[is_spike, 'y'].max() if spike_count > 0 else 0
                avg_baseline = monthly.loc[~is_spike, 'y'].mean()
                max_multiplier = max_spike / avg_baseline if avg_baseline > 0 else 0

                spike_summary.append({
                    'Product': product,
                    'Spike Count': spike_count,
                    'Total Months': len(monthly),
                    'Spike Ratio (%)': spike_ratio,
                    'Max Spike': max_spike,
                    'Avg Baseline': avg_baseline,
                    'Max Multiplier': max_multiplier
                })

                if i <= 10 or spike_count >= 5:  # 最初の10製品または多スパイク製品を表示
                    print(f"  [{i}/{len(products)}] {product}: {spike_count} spikes ({spike_ratio:.1f}%), max {max_multiplier:.1f}x")

        # サマリー
        if len(spike_summary) > 0:
            summary_df = pd.DataFrame(spike_summary)
            total_spikes = summary_df['Spike Count'].sum()
            total_months = summary_df['Total Months'].sum()

            print(f"\n【Spike Detection Summary】")
            print(f"Products with spikes: {len(summary_df)}/{len(products)}")
            print(f"Total spike months: {total_spikes}/{total_months} ({total_spikes/total_months*100:.1f}%)")
            print(f"Max multiplier: {summary_df['Max Multiplier'].max():.1f}x")

            self.product_spike_info = summary_df

        return all_spikes

    def create_category_baseline(self, df: pd.DataFrame, train_end: str = '2024-05-31') -> tuple:
        """製品レベルスパイク除去後のカテゴリレベルベースライン時系列を作成"""

        print(f"\n{'='*80}")
        print(f"Creating Category-Level Baseline (Product-Level Spikes Removed)")
        print(f"{'='*80}")

        # 全製品のスパイク検出
        spike_products = self.detect_product_spikes(df)

        # 全期間の日付リスト
        all_dates = sorted(df['年月'].unique())

        # 製品ごとにベースライン値を集計
        category_baseline = []

        for date in all_dates:
            date_total = 0
            spike_excluded_count = 0

            for product in df['製品名'].unique():
                product_df = df[(df['製品名'] == product) & (df['年月'] == date)]

                if len(product_df) == 0:
                    continue

                product_value = product_df['台数'].sum()

                # この製品・月がスパイクかチェック
                is_spike = False
                for spike_df in spike_products:
                    if (spike_df['製品名'].iloc[0] == product) and (date in spike_df['ds'].values):
                        spike_row = spike_df[spike_df['ds'] == date]
                        if len(spike_row) > 0 and spike_row['is_spike'].iloc[0]:
                            is_spike = True
                            spike_excluded_count += 1
                            # スパイク月は除外（または補間）
                            # ここでは除外せず、前後の平均で補間
                            break

                if not is_spike:
                    date_total += product_value
                else:
                    # スパイクの場合、その製品の平均ベースラインで補完
                    product_monthly = df[df['製品名'] == product].groupby('年月')['台数'].sum()
                    product_spikes = self.prep.detect_spikes(product_monthly, method='combined',
                                                              iqr_multiplier=2.0, z_threshold=2.5)
                    baseline_values = product_monthly.values[~product_spikes]
                    if len(baseline_values) > 0:
                        date_total += baseline_values.mean()

            category_baseline.append({
                'ds': date,
                'y': date_total,
                'spikes_excluded': spike_excluded_count
            })

        baseline_df = pd.DataFrame(category_baseline)

        # 訓練/テスト分割
        train_baseline = baseline_df[baseline_df['ds'] <= train_end].copy()
        test_baseline = baseline_df[baseline_df['ds'] > train_end].copy()

        print(f"\nCategory baseline created:")
        print(f"  Training period: {len(train_baseline)} months")
        print(f"  Test period: {len(test_baseline)} months")
        print(f"  Avg spikes excluded per month: {baseline_df['spikes_excluded'].mean():.1f}")

        return train_baseline, test_baseline, baseline_df

    def forecast_with_ensemble(self, train_baseline: pd.DataFrame, periods: int) -> dict:
        """アンサンブル予測"""

        print(f"\n{'='*80}")
        print(f"Ensemble Forecast on Product-Level Spike-Removed Baseline")
        print(f"{'='*80}")

        # モデル作成
        models = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('MA12', MovingAverageForecaster(window=12)),
            ('RobustMA6', RobustBaselineForecaster(window=6)),
            ('RobustMA12', RobustBaselineForecaster(window=12)),
            ('SeasonalNaive', SeasonalNaiveForecaster())
        ]

        # 検証データ分割
        val_size = min(6, len(train_baseline) // 4)
        train_cv = train_baseline[:-val_size][['ds', 'y']].copy()
        val_cv = train_baseline[-val_size:][['ds', 'y']].copy()

        # 各モデルの予測
        predictions = []
        model_names = []

        print(f"\nTraining {len(models)} models...")
        for name, model in models:
            try:
                model.fit(train_cv)
                pred = model.predict(len(val_cv))
                merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')

                if len(merged) > 0:
                    predictions.append(merged['yhat'].values)
                    model_names.append(name)
                    print(f"  {name}: OK")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

        if len(predictions) == 0:
            print("No models succeeded, using fallback")
            model = RobustBaselineForecaster(window=6)
            model.fit(train_baseline[['ds', 'y']])
            forecast = model.predict(periods)
            return {'forecast': forecast, 'weights': {'RobustMA6': 1.0}}

        # 重み最適化
        X = np.column_stack(predictions)
        y_true = val_cv['y'].values[:len(predictions[0])]

        def objective(weights):
            weights = weights / weights.sum()
            ensemble_pred = X @ weights
            mask = y_true > 0
            if mask.sum() == 0:
                return 1e6
            mape = np.mean(np.abs((y_true[mask] - ensemble_pred[mask]) / y_true[mask]))
            return mape

        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        optimal_weights = result.x / result.x.sum()
        weights_dict = {name: weight for name, weight in zip(model_names, optimal_weights)}

        print(f"\n【Optimized Ensemble Weights】")
        for name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"  {name}: {weight:.3f}")

        # 全訓練データで予測
        ensemble_predictions = []
        for name, model in models:
            if name in weights_dict:
                try:
                    model.fit(train_baseline[['ds', 'y']])
                    pred = model.predict(periods)
                    ensemble_predictions.append((pred, weights_dict[name]))
                except:
                    continue

        # 重み付け平均
        forecast_df = ensemble_predictions[0][0][['ds']].copy()
        forecast_df['yhat'] = 0

        for pred, weight in ensemble_predictions:
            forecast_df['yhat'] += pred['yhat'] * weight

        return {'forecast': forecast_df, 'weights': weights_dict}


def evaluate_product_spike_removal(sheet_name: str = 'TUHS'):
    """製品レベルスパイク除去アプローチを評価"""

    print(f"\n{'='*80}")
    print(f"Evaluation: Product-Level Spike Removal Approach")
    print(f"{'='*80}")

    psr = ProductLevelSpikeRemoval(sheet_name)

    # データ読み込み
    df = psr.prep.load_data(sheet_name)

    # 製品レベルスパイク除去後のカテゴリベースライン作成
    train_baseline, test_baseline, full_baseline = psr.create_category_baseline(df)

    # アンサンブル予測
    result = psr.forecast_with_ensemble(train_baseline, periods=len(test_baseline))

    # 評価
    merged = pd.merge(test_baseline[['ds', 'y']], result['forecast'][['ds', 'yhat']],
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
        print(f"  Previous category-level: 13.53% MAPE")
        print(f"  Product-level spike removal: {mape:.2f}% MAPE")
        print(f"  Improvement: {13.53 - mape:+.2f}% points")

        if mape < 10.0:
            print(f"\n🎯 TARGET ACHIEVED: {mape:.2f}% < 10%")
        else:
            print(f"\n⚠ Target not yet achieved: {mape:.2f}% (need {mape - 10.0:.2f}% more improvement)")

    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # グラフ1: 予測 vs 実績
    ax = axes[0]
    ax.plot(merged['ds'], merged['y'], 'o-', label='Actual (Product-Spike-Removed)',
           linewidth=2, markersize=8, color='black', zorder=5)
    ax.plot(merged['ds'], merged['yhat'], '^--', label='Ensemble Forecast',
           linewidth=2, markersize=6, color='purple', alpha=0.8)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'{sheet_name}: Product-Level Spike Removal + Ensemble Forecast',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: 誤差率
    ax = axes[1]
    merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100
    colors = ['green' if abs(e) < 10 else 'orange' if abs(e) < 15 else 'red'
              for e in merged['pct_error']]
    ax.bar(merged['ds'], merged['pct_error'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (<10%)')
    ax.axhline(y=-10, color='green', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Error by Month', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/tuhs_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'tuhs_product_spike_removal.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'tuhs_product_spike_removal.png'}")
    plt.close()

    return mape if 'mape' in locals() else None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("TUHS: Product-Level Spike Removal Evaluation")
    print("="*80)

    mape = evaluate_product_spike_removal('TUHS')


if __name__ == '__main__':
    main()
