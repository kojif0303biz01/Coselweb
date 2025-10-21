"""予測結果の可視化スクリプト"""

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
from src.lightgbm_models import LightGBMForecaster
import warnings
warnings.filterwarnings('ignore')

# スタイル設定
sns.set_style('whitegrid')
sns.set_palette('husl')
sns.set_context("notebook", font_scale=1.0)


def plot_category_comparison(sheet_name: str, save_dir: Path):
    """カテゴリごとの予測vs実績グラフを作成"""

    print(f"\n{'='*80}")
    print(f"グラフ作成中: {sheet_name}")
    print(f"{'='*80}")

    # データ準備
    prep = SalesForecastDataPreparation()
    train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

    periods = len(test_monthly)

    # モデルの予測を取得
    models_predictions = {}

    # 1. 移動平均(3か月)
    ma3 = MovingAverageForecaster(window=3, name="MA3")
    ma3.fit(train_monthly)
    models_predictions['Moving Avg (3M)'] = ma3.predict(periods)

    # 2. 移動平均(6か月)
    ma6 = MovingAverageForecaster(window=6, name="MA6")
    ma6.fit(train_monthly)
    models_predictions['Moving Avg (6M)'] = ma6.predict(periods)

    # 3. 前年同月
    sn = SeasonalNaiveForecaster(name="SeasonalNaive")
    sn.fit(train_monthly)
    models_predictions['Seasonal Naive'] = sn.predict(periods)

    # 4. ロバスト移動平均(6か月)
    rma6 = RobustBaselineForecaster(window=6, name="RobustMA6")
    rma6.fit(train_monthly)
    models_predictions['Robust MA (6M)'] = rma6.predict(periods)

    # 5. LightGBM
    try:
        lgb_model = LightGBMForecaster(name="LightGBM")
        lgb_model.fit(train_monthly)
        models_predictions['LightGBM'] = lgb_model.predict(train_monthly, periods)
    except Exception as e:
        print(f"  LightGBM予測エラー: {e}")

    # ===================================================================
    # グラフ1: 全期間の時系列 + 各モデルの予測
    # ===================================================================
    fig, ax = plt.subplots(figsize=(16, 8))

    # Training period actual
    ax.plot(train_monthly['ds'], train_monthly['y'],
            'o-', linewidth=2, markersize=4, label='Training Data', color='blue', alpha=0.7)

    # Test period actual
    ax.plot(test_monthly['ds'], test_monthly['y'],
            'o-', linewidth=3, markersize=6, label='Actual (Test)', color='black', zorder=10)

    # Model predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, pred_df) in enumerate(models_predictions.items()):
        ax.plot(pred_df['ds'], pred_df['yhat'],
                's--', linewidth=2, markersize=5, label=model_name,
                color=colors[i % len(colors)], alpha=0.8)

    # Train/test boundary
    train_test_boundary = train_monthly['ds'].max()
    ax.axvline(x=train_test_boundary, color='red', linestyle=':', linewidth=2,
               label='Train/Test Split', alpha=0.7)

    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=14, fontweight='bold')
    ax.set_title(f'{sheet_name} Category: Predictions vs Actual',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存
    save_path = save_dir / f'{sheet_name}_timeseries_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ2: テスト期間のみの詳細比較
    # ===================================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    # Actual values
    ax.plot(test_monthly['ds'], test_monthly['y'],
            'o-', linewidth=3, markersize=8, label='Actual', color='black', zorder=10)

    # Model predictions
    for i, (model_name, pred_df) in enumerate(models_predictions.items()):
        ax.plot(pred_df['ds'], pred_df['yhat'],
                's--', linewidth=2, markersize=6, label=model_name,
                color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=14, fontweight='bold')
    ax.set_title(f'{sheet_name} Category: Test Period Detail (2024-06 to 2025-05)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存
    save_path = save_dir / f'{sheet_name}_testperiod_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ3: モデル性能比較（棒グラフ）
    # ===================================================================
    metrics_data = []

    for model_name, pred_df in models_predictions.items():
        # 実際のデータとマージ
        merged = pd.merge(test_monthly[['ds', 'y']], pred_df[['ds', 'yhat']],
                         on='ds', how='inner')

        if len(merged) > 0:
            y_true = merged['y'].values
            y_pred = merged['yhat'].values

            # メトリクス計算
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            mae = np.mean(np.abs(y_true - y_pred))

            mask = y_true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan

            metrics_data.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            })

    metrics_df = pd.DataFrame(metrics_data)

    # 3つのメトリクスを横並びで表示
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MAPE
    ax = axes[0]
    bars = ax.bar(metrics_df['Model'], metrics_df['MAPE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Percentage Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    # Display values on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

    # MAE
    ax = axes[1]
    bars = ax.bar(metrics_df['Model'], metrics_df['MAE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
    ax.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    # RMSE
    ax = axes[2]
    bars = ax.bar(metrics_df['Model'], metrics_df['RMSE'], color=colors[:len(metrics_df)])
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    ax.set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle(f'{sheet_name} Category: Model Performance Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # 保存
    save_path = save_dir / f'{sheet_name}_metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()

    return metrics_df


def plot_overall_summary(save_dir: Path):
    """全カテゴリの総合サマリーグラフ"""

    print(f"\n{'='*80}")
    print("総合サマリーグラフ作成中")
    print(f"{'='*80}")

    # 評価結果CSVを読み込み
    results_path = 'data/processed/model_evaluation_results.csv'
    if Path(results_path).exists():
        results_df = pd.read_csv(results_path, encoding='utf-8-sig')
    else:
        print("  評価結果CSVが見つかりません")
        return

    # ===================================================================
    # グラフ1: カテゴリ別・モデル別MAPE比較（ヒートマップ）
    # ===================================================================
    pivot_data = results_df.pivot_table(
        values='MAPE',
        index='モデル名',
        columns='カテゴリ',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'MAPE (%)'}, ax=ax, vmin=0, vmax=100,
                annot_kws={'fontsize': 11})
    ax.set_title('Model Performance Heatmap: MAPE by Category and Model',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / 'overall_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ2: カテゴリ別ベストモデルのMAPE比較
    # ===================================================================
    best_models = results_df.groupby('カテゴリ').apply(
        lambda x: x.loc[x['MAPE'].idxmin()]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {'TUHS': 'green', 'PBA': 'orange', 'PCA': 'red'}
    bar_colors = [colors_map.get(cat, 'blue') for cat in best_models['カテゴリ']]

    bars = ax.bar(range(len(best_models)), best_models['MAPE'], color=bar_colors)
    ax.set_xticks(range(len(best_models)))
    ax.set_xticklabels([f"{row['カテゴリ']}\n({row['モデル名']})"
                        for _, row in best_models.iterrows()], fontsize=11)
    ax.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
    ax.set_title('Best Model Performance by Category', fontsize=16, fontweight='bold', pad=20)
    ax.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Excellent (<20%)')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<40%)')
    ax.grid(True, axis='y', alpha=0.3)

    # 値をバーの上に表示
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    save_path = save_dir / 'best_models_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()

    # ===================================================================
    # グラフ3: モデル種別（ベースライン vs 機械学習）比較
    # ===================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, category in enumerate(['TUHS', 'PCA', 'PBA']):
        ax = axes[i]
        cat_data = results_df[results_df['カテゴリ'] == category]

        baseline_mape = cat_data[cat_data['モデル種別'] == 'ベースライン']['MAPE'].mean()
        ml_mape = cat_data[cat_data['モデル種別'] == '機械学習']['MAPE'].mean()

        bars = ax.bar(['Baseline', 'Machine Learning'], [baseline_mape, ml_mape],
                     color=['skyblue', 'salmon'])
        ax.set_ylabel('Average MAPE (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{category} Category', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(baseline_mape, ml_mape) * 1.2)

        # Display values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Baseline vs Machine Learning: Average MAPE Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = save_dir / 'baseline_vs_ml.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  保存: {save_path}")
    plt.close()


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("購買予測システム - 可視化レポート生成")
    print("="*80)

    # 保存ディレクトリの作成
    save_dir = Path('visualizations')
    save_dir.mkdir(exist_ok=True)

    # カテゴリごとのグラフ作成
    all_metrics = {}
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        metrics_df = plot_category_comparison(sheet_name, save_dir)
        all_metrics[sheet_name] = metrics_df

    # 総合サマリーグラフ
    plot_overall_summary(save_dir)

    print(f"\n{'='*80}")
    print(f"全てのグラフを保存しました: {save_dir}/")
    print(f"{'='*80}\n")

    # 生成されたファイル一覧
    print("生成されたファイル:")
    for file in sorted(save_dir.glob('*.png')):
        print(f"  - {file.name}")

    return all_metrics


if __name__ == "__main__":
    metrics = main()
