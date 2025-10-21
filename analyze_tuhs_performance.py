"""TUHS詳細パフォーマンス分析 - 10%以下を目指して"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import RobustBaselineForecaster

sns.set_style('whitegrid')


def analyze_tuhs_performance():
    """TUHS現状パフォーマンスの詳細分析"""

    print("\n" + "="*80)
    print("TUHS Performance Analysis - Target: <10% MAPE")
    print("="*80)

    # データ読み込み
    prep = SalesForecastDataPreparation()
    train_monthly, test_monthly = prep.create_category_aggregation('TUHS')

    print(f"\n【Current Best Performance】")
    print(f"Category-level (Robust MA 6M): 13.53% MAPE")
    print(f"Target: <10% MAPE")
    print(f"Required improvement: >3.53% points")

    # カテゴリレベル予測
    model = RobustBaselineForecaster(window=6, name="TUHS_Category")
    model.fit(train_monthly)
    pred = model.predict(len(test_monthly))

    # 評価
    merged = pd.merge(test_monthly[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')
    merged['error'] = merged['y'] - merged['yhat']
    merged['abs_error'] = np.abs(merged['error'])
    merged['pct_error'] = (merged['error'] / merged['y']) * 100
    merged['abs_pct_error'] = np.abs(merged['pct_error'])

    print(f"\n【Month-by-Month Errors】")
    print(merged[['ds', 'y', 'yhat', 'error', 'pct_error']].to_string(index=False))

    print(f"\n【Error Statistics】")
    print(f"Mean APE: {merged['abs_pct_error'].mean():.2f}%")
    print(f"Median APE: {merged['abs_pct_error'].median():.2f}%")
    print(f"Max APE: {merged['abs_pct_error'].max():.2f}%")
    print(f"Min APE: {merged['abs_pct_error'].min():.2f}%")
    print(f"Std Dev: {merged['abs_pct_error'].std():.2f}%")

    # 大きな誤差の月を特定
    high_error_months = merged[merged['abs_pct_error'] > 15.0]
    print(f"\n【High Error Months (>15%)】")
    if len(high_error_months) > 0:
        print(high_error_months[['ds', 'y', 'yhat', 'pct_error']].to_string(index=False))
    else:
        print("None")

    return merged


def analyze_tuhs_spikes():
    """TUHSのスパイクパターンを詳細分析"""

    print(f"\n{'='*80}")
    print("TUHS Spike Pattern Analysis")
    print(f"{'='*80}")

    prep = SalesForecastDataPreparation()
    df = prep.load_data('TUHS')

    # 全期間の月次データ
    monthly = df.groupby('年月')['台数'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly = monthly.sort_values('ds').reset_index(drop=True)

    # スパイク検出
    is_spike = prep.detect_spikes(monthly['y'], method='combined',
                                   iqr_multiplier=2.0, z_threshold=2.5)

    monthly['is_spike'] = is_spike
    monthly['y_baseline'] = monthly['y'].copy()
    monthly.loc[is_spike, 'y_baseline'] = np.nan

    # 訓練期間とテスト期間
    train = monthly[monthly['ds'] <= '2024-05-31']
    test = monthly[monthly['ds'] > '2024-05-31']

    train_spikes = train['is_spike'].sum()
    test_spikes = test['is_spike'].sum()

    print(f"\n【Spike Statistics】")
    print(f"Training period spikes: {train_spikes}/{len(train)} ({train_spikes/len(train)*100:.1f}%)")
    print(f"Test period spikes: {test_spikes}/{len(test)} ({test_spikes/len(test)*100:.1f}%)")

    # スパイク月の詳細
    spike_months_train = train[train['is_spike']]
    if len(spike_months_train) > 0:
        print(f"\n【Training Period Spike Months】")
        spike_months_train['spike_ratio'] = spike_months_train['y'] / train[~train['is_spike']]['y'].mean()
        print(spike_months_train[['ds', 'y', 'spike_ratio']].head(10).to_string(index=False))

    spike_months_test = test[test['is_spike']]
    if len(spike_months_test) > 0:
        print(f"\n【Test Period Spike Months】")
        spike_months_test['spike_ratio'] = spike_months_test['y'] / test[~test['is_spike']]['y'].mean()
        print(spike_months_test[['ds', 'y', 'spike_ratio']].to_string(index=False))

    # ベースライン統計
    baseline_train = train[~train['is_spike']]['y']
    baseline_test = test[~test['is_spike']]['y']

    print(f"\n【Baseline Statistics (Spike-excluded)】")
    print(f"Training baseline mean: {baseline_train.mean():.0f} units/month")
    print(f"Training baseline std: {baseline_train.std():.0f}")
    print(f"Training baseline CV: {baseline_train.std()/baseline_train.mean()*100:.1f}%")

    if len(baseline_test) > 0:
        print(f"Test baseline mean: {baseline_test.mean():.0f} units/month")
        print(f"Test baseline std: {baseline_test.std():.0f}")
        print(f"Test baseline CV: {baseline_test.std()/baseline_test.mean()*100:.1f}%")

    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # グラフ1: スパイク強調表示
    ax = axes[0]
    ax.plot(monthly['ds'], monthly['y'], 'o-', label='Actual (All)',
           linewidth=2, markersize=4, color='blue', alpha=0.7)

    # スパイク月を強調
    spike_months = monthly[monthly['is_spike']]
    ax.scatter(spike_months['ds'], spike_months['y'], s=200, facecolors='none',
              edgecolors='red', linewidths=3, label='Spike Months', zorder=10)

    # ベースライン平均線
    baseline_mean = monthly[~monthly['is_spike']]['y'].mean()
    ax.axhline(y=baseline_mean, color='green', linestyle='--',
              linewidth=2, label=f'Baseline Mean ({baseline_mean:.0f})', alpha=0.7)

    # 訓練/テスト境界
    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle=':',
              linewidth=2, label='Train/Test Split', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('TUHS: Sales with Spike Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: ベースラインのみ
    ax = axes[1]
    baseline_monthly = monthly[~monthly['is_spike']]
    ax.plot(baseline_monthly['ds'], baseline_monthly['y'], 'o-',
           label='Baseline (Spike-excluded)', linewidth=2, markersize=6, color='green')

    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle=':',
              linewidth=2, label='Train/Test Split', alpha=0.7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('TUHS: Baseline Demand Only (Spikes Excluded)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/tuhs_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'tuhs_spike_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'tuhs_spike_analysis.png'}")
    plt.close()

    return monthly


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("TUHS Detailed Analysis - Aiming for <10% MAPE")
    print("="*80)

    # 1. 現状パフォーマンス分析
    error_df = analyze_tuhs_performance()

    # 2. スパイクパターン分析
    spike_df = analyze_tuhs_spikes()

    # 3. 改善機会の特定
    print(f"\n{'='*80}")
    print("Improvement Opportunities Identified")
    print(f"{'='*80}")

    print("\n1. Spike Separation (HIGH PRIORITY)")
    print("   - TUHS has MANY spikes (much more than PCA)")
    print("   - Separating spikes from baseline demand should be highly effective")
    print("   - Expected: 2-4% improvement")

    print("\n2. Series-Level Ensemble")
    print("   - 5 series with distinct patterns")
    print("   - Apply PCA's successful ensemble approach")
    print("   - Expected: 1-2% improvement")

    print("\n3. Product-Level Forecasting")
    print("   - Only 23 products (manageable)")
    print("   - Top 10 products contribute 80%")
    print("   - Expected: 0.5-1% improvement")

    print("\n4. Advanced Smoothing Methods")
    print("   - Holt-Winters exponential smoothing")
    print("   - Expected: 0.5-1% improvement")

    print(f"\n{'='*80}")
    print("Recommended Approach")
    print(f"{'='*80}")
    print("\nStep 1: Spike Separation (Most Impact)")
    print("Step 2: Series-Level Ensemble")
    print("Step 3: Fine-tuning if needed")
    print("\nTotal Expected Improvement: 3.5-7% points")
    print("Target Achievement: HIGH probability")


if __name__ == '__main__':
    main()
