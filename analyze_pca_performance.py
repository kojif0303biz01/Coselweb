"""PCAカテゴリの詳細パフォーマンス分析"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from hierarchical_forecasting_series import HierarchicalSeriesForecaster

sns.set_style('whitegrid')


def analyze_pca_series_trends():
    """PCAの各シリーズのトレンドを分析"""

    print("\n" + "="*80)
    print("PCA Category: Detailed Series Trend Analysis")
    print("="*80)

    # データ読み込み
    prep = SalesForecastDataPreparation()
    df = prep.load_data('PCA')

    # 各シリーズのトレンド分析
    series_list = ['PCA600F シリーズ', 'PCA1000F シリーズ', 'PCA300F シリーズ', 'PCA1500F シリーズ']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    series_stats = []

    for idx, series_name in enumerate(series_list):
        ax = axes[idx]

        # シリーズデータ抽出
        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        # 訓練期間とテスト期間に分割
        train = monthly[monthly['ds'] <= '2024-05-31']
        test = monthly[monthly['ds'] > '2024-05-31']

        # トレンド分析（線形回帰）
        if len(train) > 0:
            x = np.arange(len(train))
            y = train['y'].values

            # 線形トレンド
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            trend_line = p(x)

            # トレンドの傾き（月あたりの変化）
            slope = z[0]
            slope_percent = (slope / y.mean()) * 100 if y.mean() != 0 else 0

            # 変動係数（CV: Coefficient of Variation）
            cv = (y.std() / y.mean() * 100) if y.mean() != 0 else 0

            # テスト期間の平均とトレンド予測の比較
            if len(test) > 0:
                test_actual_mean = test['y'].mean()
                # 訓練期間の最後の値から線形トレンドで予測
                last_train_x = len(train) - 1
                test_x = np.arange(last_train_x + 1, last_train_x + 1 + len(test))
                test_trend_pred = p(test_x).mean()

                trend_error = ((test_actual_mean - test_trend_pred) / test_actual_mean * 100) if test_actual_mean != 0 else 0
            else:
                test_actual_mean = np.nan
                test_trend_pred = np.nan
                trend_error = np.nan

            series_stats.append({
                'Series': series_name,
                'Train Mean': f"{y.mean():.1f}",
                'Train Std': f"{y.std():.1f}",
                'CV (%)': f"{cv:.1f}",
                'Trend Slope': f"{slope:.2f}",
                'Trend %/month': f"{slope_percent:.2f}",
                'Test Mean': f"{test_actual_mean:.1f}" if not np.isnan(test_actual_mean) else "N/A",
                'Trend Prediction Error (%)': f"{trend_error:.1f}" if not np.isnan(trend_error) else "N/A"
            })

            # グラフ描画
            ax.plot(monthly['ds'], monthly['y'], 'o-', label='Actual', linewidth=2, markersize=4)
            ax.plot(train['ds'], trend_line, '--', label='Linear Trend (Train)',
                   linewidth=2, color='red', alpha=0.7)

            # 訓練/テスト境界
            ax.axvline(x=pd.Timestamp('2024-05-31'), color='green',
                      linestyle=':', linewidth=2, label='Train/Test Split', alpha=0.7)

            ax.set_title(f'{series_name}\nTrend: {slope_percent:.2f}%/month, CV: {cv:.1f}%',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel('Sales Volume', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/pca_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'pca_series_trends.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'pca_series_trends.png'}")
    plt.close()

    # 統計サマリーテーブル
    print("\n" + "="*80)
    print("PCA Series Statistics Summary")
    print("="*80)
    stats_df = pd.DataFrame(series_stats)
    print(stats_df.to_string(index=False))

    return stats_df


def analyze_forecast_errors():
    """現在のシリーズレベル予測の誤差を詳細分析"""

    print("\n" + "="*80)
    print("PCA: Current Series-Level Forecast Error Analysis")
    print("="*80)

    # シリーズレベル予測を実行
    hf = HierarchicalSeriesForecaster('PCA')
    series_pred = hf.hierarchical_forecast(periods=12)

    # 実績データ取得
    prep = SalesForecastDataPreparation()
    _, test_monthly = prep.create_category_aggregation('PCA')

    # 予測と実績をマージ
    merged = pd.merge(test_monthly[['ds', 'y']], series_pred[['ds', 'yhat']],
                     on='ds', how='inner')

    # 誤差分析
    merged['error'] = merged['y'] - merged['yhat']
    merged['abs_error'] = np.abs(merged['error'])
    merged['pct_error'] = (merged['error'] / merged['y']) * 100
    merged['abs_pct_error'] = np.abs(merged['pct_error'])

    print("\nMonth-by-Month Forecast Errors:")
    print(merged[['ds', 'y', 'yhat', 'error', 'pct_error']].to_string(index=False))

    print(f"\nMean Absolute Percentage Error: {merged['abs_pct_error'].mean():.2f}%")
    print(f"Median Absolute Percentage Error: {merged['abs_pct_error'].median():.2f}%")
    print(f"Max Absolute Percentage Error: {merged['abs_pct_error'].max():.2f}%")
    print(f"Min Absolute Percentage Error: {merged['abs_pct_error'].min():.2f}%")

    # 誤差の可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # グラフ1: 実績 vs 予測
    ax = axes[0]
    ax.plot(merged['ds'], merged['y'], 'o-', label='Actual',
           linewidth=2, markersize=8, color='black')
    ax.plot(merged['ds'], merged['yhat'], 's--', label='Series-Level Forecast',
           linewidth=2, markersize=6, color='red')
    ax.fill_between(merged['ds'], merged['y'], merged['yhat'], alpha=0.3)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Actual vs Series-Level Forecast', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: 誤差率
    ax = axes[1]
    colors = ['green' if e >= 0 else 'red' for e in merged['pct_error']]
    ax.bar(merged['ds'], merged['pct_error'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Forecast Percentage Error by Month', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/pca_analysis')
    plt.savefig(save_dir / 'pca_forecast_errors.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'pca_forecast_errors.png'}")
    plt.close()

    return merged


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PCA Performance Analysis for Improvement")
    print("="*80)

    # 1. シリーズトレンド分析
    stats_df = analyze_pca_series_trends()

    # 2. 予測誤差分析
    error_df = analyze_forecast_errors()

    # 3. 改善提案
    print("\n" + "="*80)
    print("Improvement Opportunities Identified")
    print("="*80)

    print("\n1. Trend Adjustment")
    print("   - Some series show clear linear trends that current MA models may not capture well")
    print("   - Consider adding linear trend component to forecasts")

    print("\n2. Long Tail Prediction")
    print("   - Currently using simple average for long-tail series (28.6% contribution)")
    print("   - Could improve by applying category-level trend to long-tail")

    print("\n3. Ensemble Methods")
    print("   - Combine multiple forecasting methods with optimized weights")
    print("   - Could blend MA, trend-adjusted MA, and exponential smoothing")

    print("\n4. Series-Specific Model Selection")
    print("   - Currently auto-selecting from limited set (MA3, MA6, SeasonalNaive)")
    print("   - Could expand to include exponential smoothing, damped trend methods")

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
