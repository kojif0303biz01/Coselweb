"""TUHS: Series-Level Ensemble Forecast for <10% MAPE Target"""

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


class TUHSSeriesEnsemble:
    """TUHS シリーズレベル アンサンブル予測クラス"""

    def __init__(self):
        self.prep = SalesForecastDataPreparation()
        self.series_forecasts = {}
        self.series_weights = {}

    def get_series_data(self, df: pd.DataFrame, series_name: str, train_end: str = '2024-05-31'):
        """シリーズデータを取得して訓練/テストに分割"""

        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        train = monthly[monthly['ds'] <= train_end][['ds', 'y']].copy()
        test = monthly[monthly['ds'] > train_end][['ds', 'y']].copy()

        return train, test, monthly

    def forecast_series_with_ensemble(self, train_data: pd.DataFrame, periods: int, series_name: str) -> dict:
        """シリーズごとのアンサンブル予測"""

        print(f"\n{'='*80}")
        print(f"Ensemble Forecast: {series_name}")
        print(f"{'='*80}")

        # モデルセット（スパイクに強いモデルを含む）
        models = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('MA12', MovingAverageForecaster(window=12)),
            ('RobustMA3', RobustBaselineForecaster(window=3)),
            ('RobustMA6', RobustBaselineForecaster(window=6)),
            ('RobustMA12', RobustBaselineForecaster(window=12)),
            ('SeasonalNaive', SeasonalNaiveForecaster())
        ]

        # 検証分割（最後の6か月）
        val_size = min(6, len(train_data) // 4)
        train_cv = train_data[:-val_size].copy()
        val_cv = train_data[-val_size:].copy()

        print(f"\nTraining models on {len(train_cv)} months, validating on {val_size} months...")

        # 各モデルの予測を収集
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

                    # 各モデルのMAPE
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    mask = y_true > 0
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
                    print(f"  {name}: {mape:.2f}% MAPE")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

        if len(predictions) == 0:
            print("No models succeeded, using RobustMA6 fallback")
            model = RobustBaselineForecaster(window=6)
            model.fit(train_data)
            forecast = model.predict(periods)
            return {'forecast': forecast, 'weights': {'RobustMA6': 1.0}, 'model_names': ['RobustMA6']}

        # 重み最適化
        X = np.column_stack(predictions)
        y_true = val_cv['y'].values[:len(predictions[0])]

        def objective(weights):
            """MAPE最小化目的関数"""
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

        print(f"\n【Optimized Weights for {series_name}】")
        for name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"  {name}: {weight:.3f}")

        # 検証期間のアンサンブルMAPE
        ensemble_val_pred = X @ optimal_weights
        mask = y_true > 0
        ensemble_mape = np.mean(np.abs((y_true[mask] - ensemble_val_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf
        print(f"  Ensemble Validation MAPE: {ensemble_mape:.2f}%")

        # 全訓練データで再学習して予測
        ensemble_predictions = []
        for name, model in models:
            if name in weights_dict and weights_dict[name] > 0.001:
                try:
                    model.fit(train_data)
                    pred = model.predict(periods)
                    ensemble_predictions.append((pred, weights_dict[name]))
                except Exception as e:
                    print(f"  {name} final training failed: {e}")
                    continue

        # 重み付け平均
        forecast_df = ensemble_predictions[0][0][['ds']].copy()
        forecast_df['yhat'] = 0

        for pred, weight in ensemble_predictions:
            forecast_df['yhat'] += pred['yhat'] * weight

        return {
            'forecast': forecast_df,
            'weights': weights_dict,
            'model_names': model_names,
            'val_mape': ensemble_mape
        }

    def hierarchical_ensemble_forecast(self, sheet_name: str = 'TUHS', periods: int = 12) -> pd.DataFrame:
        """シリーズレベル階層的アンサンブル予測"""

        print(f"\n{'='*80}")
        print(f"TUHS: Series-Level Hierarchical Ensemble Forecast")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(sheet_name)

        # シリーズリスト（TUHSは5シリーズ）
        series_list = sorted(df['モデル名'].unique())
        print(f"\nTotal series: {len(series_list)}")
        print(f"Series: {', '.join(series_list)}")

        # 各シリーズの予測
        all_forecasts = []

        for series_name in series_list:
            train, test, monthly = self.get_series_data(df, series_name)

            if len(train) < 12:
                print(f"\n[SKIP] {series_name}: Insufficient training data ({len(train)} months)")
                continue

            # アンサンブル予測
            result = self.forecast_series_with_ensemble(train, periods, series_name)

            self.series_forecasts[series_name] = result['forecast']
            self.series_weights[series_name] = result['weights']

            all_forecasts.append(result['forecast'])

        # カテゴリレベルに集約
        category_forecast = all_forecasts[0][['ds']].copy()
        category_forecast['yhat'] = 0

        for forecast_df in all_forecasts:
            category_forecast['yhat'] += forecast_df['yhat']

        print(f"\n{'='*80}")
        print(f"Category-Level Forecast Complete")
        print(f"{'='*80}")

        return category_forecast


def evaluate_tuhs_ensemble():
    """TUHS アンサンブル予測を評価"""

    print(f"\n{'='*80}")
    print(f"TUHS: Series-Level Ensemble Evaluation")
    print(f"{'='*80}")

    tse = TUHSSeriesEnsemble()

    # 予測実行
    forecast = tse.hierarchical_ensemble_forecast('TUHS', periods=12)

    # 実績データ取得
    prep = SalesForecastDataPreparation()
    _, test_monthly = prep.create_category_aggregation('TUHS')

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
        print(f"  Category-level baseline: 13.53% MAPE")
        print(f"  Product-level spike removal: 14.57% MAPE")
        print(f"  Series-level ensemble: {mape:.2f}% MAPE")

        if mape < 13.53:
            print(f"  Improvement: {13.53 - mape:+.2f}% points ✓")
        else:
            print(f"  Change: {13.53 - mape:+.2f}% points")

        if mape < 10.0:
            print(f"\n🎯 TARGET ACHIEVED: {mape:.2f}% < 10%")
        else:
            print(f"\n⚠ Target not yet achieved: {mape:.2f}% (need {mape - 10.0:.2f}% more improvement)")

        # 月別詳細
        merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100
        merged['abs_pct_error'] = np.abs(merged['pct_error'])

        print(f"\n{'='*80}")
        print(f"Month-by-Month Performance")
        print(f"{'='*80}")
        print(merged[['ds', 'y', 'yhat', 'pct_error']].to_string(index=False))

        # 可視化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # グラフ1: 予測 vs 実績
        ax = axes[0]
        ax.plot(merged['ds'], merged['y'], 'o-', label='Actual',
               linewidth=2, markersize=8, color='black', zorder=5)
        ax.plot(merged['ds'], merged['yhat'], '^--', label='Series-Level Ensemble',
               linewidth=2, markersize=6, color='purple', alpha=0.8)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
        ax.set_title(f'TUHS: Series-Level Ensemble Forecast (MAPE: {mape:.2f}%)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # グラフ2: 誤差率
        ax = axes[1]
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
        plt.savefig(save_dir / 'tuhs_series_ensemble.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_dir / 'tuhs_series_ensemble.png'}")
        plt.close()

        return mape

    return None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("TUHS: Series-Level Ensemble Forecast Evaluation")
    print("Target: MAPE < 10%")
    print("="*80)

    mape = evaluate_tuhs_ensemble()


if __name__ == '__main__':
    main()
