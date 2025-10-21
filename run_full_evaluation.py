"""全モデルの総合評価スクリプト"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    RobustBaselineForecaster
)
from src.lightgbm_models import LightGBMForecaster
import warnings
warnings.filterwarnings('ignore')


def evaluate_all_models(sheet_name: str):
    """全モデルの評価"""
    print(f"\n{'#'*80}")
    print(f"# {sheet_name} カテゴリの総合評価")
    print(f"{'#'*80}\n")

    # データ準備
    prep = SalesForecastDataPreparation()
    train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

    results = []
    periods = len(test_monthly)

    # ================================================================================
    # ベースラインモデル
    # ================================================================================
    print(f"\n{'='*80}")
    print("ベースラインモデルの評価")
    print(f"{'='*80}\n")

    baseline_models = [
        ('移動平均(3か月)', MovingAverageForecaster(window=3)),
        ('移動平均(6か月)', MovingAverageForecaster(window=6)),
        ('移動平均(12か月)', MovingAverageForecaster(window=12)),
        ('前年同月', SeasonalNaiveForecaster()),
        ('ロバスト移動平均(6か月)', RobustBaselineForecaster(window=6)),
    ]

    for model_name, model in baseline_models:
        print(f"実行中: {model_name}")
        model.fit(train_monthly)
        predictions = model.predict(periods)
        metrics = model.evaluate(test_monthly)

        results.append({
            'カテゴリ': sheet_name,
            'モデル種別': 'ベースライン',
            'モデル名': model_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'MAPE': metrics['MAPE']
        })

        print(f"  MAPE: {metrics['MAPE']:.2f}%")

    # ================================================================================
    # LightGBMモデル
    # ================================================================================
    print(f"\n{'='*80}")
    print("LightGBMモデルの評価")
    print(f"{'='*80}\n")

    lgb_models = [
        ('LightGBM (標準)', LightGBMForecaster(
            name="LightGBM_Standard",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5
        )),
        ('LightGBM (深層)', LightGBMForecaster(
            name="LightGBM_Deep",
            n_estimators=150,
            learning_rate=0.03,
            max_depth=7
        )),
    ]

    for model_name, model in lgb_models:
        try:
            print(f"実行中: {model_name}")
            model.fit(train_monthly)
            predictions = model.predict(train_monthly, periods)
            metrics = model.evaluate(test_monthly, predictions)

            results.append({
                'カテゴリ': sheet_name,
                'モデル種別': '機械学習',
                'モデル名': model_name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE']
            })

            print(f"  MAPE: {metrics['MAPE']:.2f}%")

        except Exception as e:
            print(f"  エラー: {e}")

    # ================================================================================
    # 結果サマリー
    # ================================================================================
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE')

    print(f"\n{'='*80}")
    print(f"{sheet_name} カテゴリ - 全モデル評価結果（MAPE順）")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    # ベストモデル
    best = results_df.iloc[0]
    print(f"\n【{sheet_name} ベストモデル】")
    print(f"  モデル: {best['モデル名']}")
    print(f"  MAPE: {best['MAPE']:.2f}%")
    print(f"  RMSE: {best['RMSE']:.2f}")
    print(f"  MAE: {best['MAE']:.2f}")

    return results_df


def main():
    """メイン処理"""
    print(f"\n{'#'*80}")
    print("# 購買予測システム - 全カテゴリ総合評価")
    print(f"{'#'*80}\n")

    all_results = []

    # 全カテゴリで評価
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        category_results = evaluate_all_models(sheet_name)
        all_results.append(category_results)

        print(f"\n{'-'*80}\n")

    # 全カテゴリの結果を結合
    final_results = pd.concat(all_results, ignore_index=True)

    # 結果をCSVに保存
    output_path = 'data/processed/model_evaluation_results.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_results.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n{'#'*80}")
    print("# 全カテゴリ統合結果")
    print(f"{'#'*80}\n")

    # カテゴリ別ベストモデル
    print("【カテゴリ別ベストモデル】\n")
    for category in ['TUHS', 'PCA', 'PBA']:
        category_best = final_results[final_results['カテゴリ'] == category].sort_values('MAPE').iloc[0]
        print(f"{category:5s}: {category_best['モデル名']:30s} (MAPE: {category_best['MAPE']:6.2f}%)")

    print(f"\n{'='*80}")
    print(f"結果を保存しました: {output_path}")
    print(f"{'='*80}\n")

    # カテゴリ別・モデル種別でのまとめ
    print("\n【カテゴリ別・モデル種別サマリー】\n")
    summary = final_results.groupby(['カテゴリ', 'モデル種別'])['MAPE'].agg(['mean', 'min', 'max'])
    print(summary)

    return final_results


if __name__ == "__main__":
    results = main()
