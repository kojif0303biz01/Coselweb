"""階層的予測の比較可視化スクリプト"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import RobustBaselineForecaster
from hierarchical_forecasting_series import HierarchicalSeriesForecaster

sns.set_style('whitegrid')


def create_comparison_graphs(sheet_name: str, save_dir: Path):
    """カテゴリごとの比較グラフを作成"""

    print(f"\n{'='*80}")
    print(f"Creating Comparison Graphs: {sheet_name}")
    print(f"{'='*80}")

    # データ準備
    prep = SalesForecastDataPreparation()
    train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)
    periods = len(test_monthly)

    # 1. カテゴリレベル予測（ロバスト移動平均6か月）
    category_model = RobustBaselineForecaster(window=6, name="Category-Level")
    category_model.fit(train_monthly)
    category_pred = category_model.predict(periods)

    # 2. シリーズレベル階層的予測
    hf = HierarchicalSeriesForecaster(sheet_name)
    series_pred = hf.hierarchical_forecast(periods)

    # ===================================================================
    # グラフ1: 時系列比較（カテゴリレベル vs シリーズレベル vs 実績）
    # ===================================================================
    fig, ax = plt.subplots(figsize=(16, 8))

    # 訓練期間の実績
    ax.plot(train_monthly['ds'], train_monthly['y'],
            'o-', linewidth=2, markersize=4, label='Training Data',
            color='blue', alpha=0.7)

    # テスト期間の実績
    ax.plot(test_monthly['ds'], test_monthly['y'],
            'o-', linewidth=3, markersize=8, label='Actual (Test)',
            color='black', zorder=10)

    # カテゴリレベル予測
    ax.plot(category_pred['ds'], category_pred['yhat'],
            's--', linewidth=2, markersize=6, label='Category-Level Forecast',
            color='red', alpha=0.8)

    # シリーズレベル予測
    if series_pred is not None:
        ax.plot(series_pred['ds'], series_pred['yhat'],
                '^--', linewidth=2, markersize=6, label='Series-Level Hierarchical',
                color='green', alpha=0.8)

    # 訓練/テスト境界線
    train_test_boundary = train_monthly['ds'].max()
    ax.axvline(x=train_test_boundary, color='red', linestyle=':',
               linewidth=2, label='Train/Test Split', alpha=0.7)

    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=14, fontweight='bold')
    ax.set_title(f'{sheet_name} Category: Hierarchical vs Category-Level Forecast',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = save_dir / f'{sheet_name}_hierarchical_comparison_timeseries.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ2: テスト期間詳細比較
    # ===================================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    # 実績
    ax.plot(test_monthly['ds'], test_monthly['y'],
            'o-', linewidth=3, markersize=8, label='Actual',
            color='black', zorder=10)

    # カテゴリレベル予測
    ax.plot(category_pred['ds'], category_pred['yhat'],
            's--', linewidth=2, markersize=6, label='Category-Level',
            color='red', alpha=0.8)

    # シリーズレベル予測
    if series_pred is not None:
        ax.plot(series_pred['ds'], series_pred['yhat'],
                '^--', linewidth=2, markersize=6, label='Series-Level Hierarchical',
                color='green', alpha=0.8)

    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=14, fontweight='bold')
    ax.set_title(f'{sheet_name} Category: Test Period Detail (2024-06 to 2025-05)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_path = save_dir / f'{sheet_name}_hierarchical_comparison_testperiod.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ3: メトリクス比較（MAPE, MAE, RMSE）
    # ===================================================================
    metrics_data = []

    # カテゴリレベルのメトリクス
    merged_cat = pd.merge(test_monthly[['ds', 'y']], category_pred[['ds', 'yhat']],
                          on='ds', how='inner')
    if len(merged_cat) > 0:
        y_true = merged_cat['y'].values
        y_pred = merged_cat['yhat'].values

        rmse_cat = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae_cat = np.mean(np.abs(y_true - y_pred))
        mask = y_true != 0
        mape_cat = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

        metrics_data.append({
            'Method': 'Category-Level',
            'RMSE': rmse_cat,
            'MAE': mae_cat,
            'MAPE': mape_cat
        })

    # シリーズレベルのメトリクス
    if series_pred is not None:
        merged_ser = pd.merge(test_monthly[['ds', 'y']], series_pred[['ds', 'yhat']],
                              on='ds', how='inner')
        if len(merged_ser) > 0:
            y_true = merged_ser['y'].values
            y_pred = merged_ser['yhat'].values

            rmse_ser = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae_ser = np.mean(np.abs(y_true - y_pred))
            mask = y_true != 0
            mape_ser = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan

            metrics_data.append({
                'Method': 'Series-Level',
                'RMSE': rmse_ser,
                'MAE': mae_ser,
                'MAPE': mape_ser
            })

    metrics_df = pd.DataFrame(metrics_data)

    # メトリクス比較グラフ
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['red', 'green']

    # MAPE
    ax = axes[0]
    bars = ax.bar(metrics_df['Method'], metrics_df['MAPE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # MAE
    ax = axes[1]
    bars = ax.bar(metrics_df['Method'], metrics_df['MAE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    # RMSE
    ax = axes[2]
    bars = ax.bar(metrics_df['Method'], metrics_df['RMSE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle(f'{sheet_name} Category: Category-Level vs Series-Level Hierarchical',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_dir / f'{sheet_name}_hierarchical_comparison_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()

    return metrics_df


def create_summary_comparison():
    """全カテゴリのサマリー比較グラフ"""

    print(f"\n{'='*80}")
    print("Creating Summary Comparison Graph")
    print(f"{'='*80}")

    # 各カテゴリの結果を収集
    summary_data = []

    # カテゴリレベル結果（既存の評価結果から）
    category_results = {
        'TUHS': 13.53,
        'PCA': 51.13,
        'PBA': 26.71
    }

    # シリーズレベル結果
    series_results = {
        'TUHS': 13.92,
        'PCA': 45.06,
        'PBA': 56.49
    }

    for category in ['TUHS', 'PCA', 'PBA']:
        summary_data.append({
            'Category': category,
            'Category-Level': category_results[category],
            'Series-Level': series_results[category]
        })

    summary_df = pd.DataFrame(summary_data)

    # サマリー比較グラフ
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(summary_df))
    width = 0.35

    bars1 = ax.bar(x - width/2, summary_df['Category-Level'], width,
                   label='Category-Level', color='red', alpha=0.8)
    bars2 = ax.bar(x + width/2, summary_df['Series-Level'], width,
                   label='Series-Level Hierarchical', color='green', alpha=0.8)

    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
    ax.set_title('Forecast Performance Comparison: Category-Level vs Series-Level Hierarchical',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Category'])
    ax.legend(fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)

    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    # 改善を示す矢印を追加
    for i, row in summary_df.iterrows():
        cat_val = row['Category-Level']
        ser_val = row['Series-Level']
        if ser_val < cat_val:  # 改善
            ax.annotate('', xy=(i + width/2, ser_val + 1),
                       xytext=(i - width/2, cat_val - 1),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            improvement = cat_val - ser_val
            ax.text(i, (cat_val + ser_val) / 2, f'-{improvement:.1f}%',
                   ha='center', fontsize=9, color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    save_dir = Path('visualizations/hierarchical_series')
    save_path = save_dir / 'overall_comparison_summary.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("Hierarchical Forecast Visualization")
    print("="*80)

    # 保存ディレクトリの作成
    save_dir = Path('visualizations/hierarchical_series')
    save_dir.mkdir(exist_ok=True, parents=True)

    # カテゴリごとのグラフ作成
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        metrics_df = create_comparison_graphs(sheet_name, save_dir)

    # サマリーグラフ作成
    create_summary_comparison()

    print(f"\n{'='*80}")
    print("All Visualization Graphs Created!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
