"""PBA: シリーズレベル階層的アンサンブル予測"""

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


class PBASeriesEnsemble:
    """PBA シリーズレベル アンサンブル予測クラス"""

    def __init__(self):
        self.prep = SalesForecastDataPreparation()
        self.series_forecasts = {}
        self.series_weights = {}
        self.series_results = {}

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

        # データチェック
        if len(train_data) < 6:
            print(f"  Insufficient data ({len(train_data)} months), using mean forecast")
            mean_val = train_data['y'].mean()
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                                   periods=periods, freq='MS'),
                'yhat': [mean_val] * periods
            })
            return {
                'forecast': forecast_df,
                'weights': {'Mean': 1.0},
                'model_names': ['Mean'],
                'val_mape': 0
            }

        # モデルセット
        models = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('MA12', MovingAverageForecaster(window=12)),
            ('RobustMA3', RobustBaselineForecaster(window=3)),
            ('RobustMA6', RobustBaselineForecaster(window=6)),
            ('RobustMA12', RobustBaselineForecaster(window=12)),
            ('SeasonalNaive', SeasonalNaiveForecaster())
        ]

        # 検証分割（最後の6か月、または25%）
        val_size = min(6, max(3, len(train_data) // 4))
        train_cv = train_data[:-val_size].copy()
        val_cv = train_data[-val_size:].copy()

        print(f"  Training: {len(train_cv)} months, Validation: {val_size} months")

        # 各モデルの予測を収集
        predictions = []
        model_names = []

        for name, model in models:
            try:
                # ウィンドウサイズチェック
                if hasattr(model, 'window') and len(train_cv) < model.window:
                    continue

                model.fit(train_cv)
                pred = model.predict(len(val_cv))
                merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')

                if len(merged) > 0 and not merged['yhat'].isna().any():
                    predictions.append(merged['yhat'].values)
                    model_names.append(name)

                    # 各モデルのMAPE
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    mask = y_true > 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        print(f"    {name}: {mape:.2f}% MAPE")
            except Exception as e:
                print(f"    {name}: FAILED - {e}")

        if len(predictions) == 0:
            print("  No models succeeded, using RobustMA3 fallback")
            model = RobustBaselineForecaster(window=3)
            model.fit(train_data)
            forecast = model.predict(periods)
            return {
                'forecast': forecast,
                'weights': {'RobustMA3': 1.0},
                'model_names': ['RobustMA3'],
                'val_mape': 0
            }

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

        print(f"  【Optimized Weights】")
        for name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"    {name}: {weight:.3f}")

        # 検証期間のアンサンブルMAPE
        ensemble_val_pred = X @ optimal_weights
        mask = y_true > 0
        ensemble_mape = np.mean(np.abs((y_true[mask] - ensemble_val_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0
        print(f"    Ensemble Val MAPE: {ensemble_mape:.2f}%")

        # 全訓練データで再学習して予測
        ensemble_predictions = []
        for name, model in models:
            if name in weights_dict and weights_dict[name] > 0.001:
                try:
                    model.fit(train_data)
                    pred = model.predict(periods)
                    ensemble_predictions.append((pred, weights_dict[name]))
                except Exception as e:
                    print(f"    {name} final training failed: {e}")
                    continue

        # 重み付け平均
        if len(ensemble_predictions) > 0:
            forecast_df = ensemble_predictions[0][0][['ds']].copy()
            forecast_df['yhat'] = 0

            for pred, weight in ensemble_predictions:
                forecast_df['yhat'] += pred['yhat'] * weight
        else:
            # フォールバック
            mean_val = train_data['y'].mean()
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                                   periods=periods, freq='MS'),
                'yhat': [mean_val] * periods
            })

        return {
            'forecast': forecast_df,
            'weights': weights_dict,
            'model_names': model_names,
            'val_mape': ensemble_mape
        }

    def hierarchical_ensemble_forecast(self, sheet_name: str = 'PBA', periods: int = 12) -> pd.DataFrame:
        """シリーズレベル階層的アンサンブル予測"""

        print(f"\n{'='*80}")
        print(f"PBA: Series-Level Hierarchical Ensemble Forecast")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(sheet_name)

        # シリーズリスト
        series_list = sorted(df['モデル名'].unique())
        print(f"\nTotal series: {len(series_list)}")

        # 各シリーズの予測
        all_forecasts = []

        for series_name in series_list:
            train, test, monthly = self.get_series_data(df, series_name)

            if len(train) < 3:
                print(f"\n[SKIP] {series_name}: Insufficient data ({len(train)} months)")
                continue

            # アンサンブル予測
            result = self.forecast_series_with_ensemble(train, periods, series_name)

            self.series_forecasts[series_name] = result['forecast']
            self.series_weights[series_name] = result['weights']
            self.series_results[series_name] = {
                'val_mape': result['val_mape'],
                'train_mean': train['y'].mean(),
                'test_mean': test['y'].mean() if len(test) > 0 else 0
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


def evaluate_pba_ensemble():
    """PBA シリーズレベルアンサンブル予測を評価"""

    print(f"\n{'='*80}")
    print(f"PBA: Series-Level Ensemble Evaluation")
    print(f"{'='*80}")

    pse = PBASeriesEnsemble()

    # 予測実行
    forecast = pse.hierarchical_ensemble_forecast('PBA', periods=12)

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
        print(f"  Category-level baseline: 26.71% MAPE")
        print(f"  Series-level ensemble: {mape:.2f}% MAPE")

        if mape < 26.71:
            improvement = 26.71 - mape
            print(f"  Improvement: {improvement:+.2f}% points ✓")
        else:
            print(f"  Change: {26.71 - mape:+.2f}% points")

        # 月別詳細
        merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100
        merged['abs_pct_error'] = np.abs(merged['pct_error'])

        print(f"\n{'='*80}")
        print(f"Month-by-Month Performance")
        print(f"{'='*80}")
        print(merged[['ds', 'y', 'yhat', 'pct_error']].to_string(index=False))

        # シリーズ別サマリー
        print(f"\n{'='*80}")
        print(f"Series Summary (sorted by training volume)")
        print(f"{'='*80}")

        series_summary = []
        for series_name, result in pse.series_results.items():
            series_summary.append({
                'Series': series_name,
                'Train_Mean': result['train_mean'],
                'Test_Mean': result['test_mean'],
                'Val_MAPE': result['val_mape'],
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
        ax.plot(merged['ds'], merged['yhat'], '^--', label=f'Series-Level Ensemble (MAPE: {mape:.2f}%)',
               linewidth=2, markersize=6, color='purple', alpha=0.8)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
        ax.set_title(f'PBA: Series-Level Ensemble Forecast (MAPE: {mape:.2f}%)',
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
        plt.savefig(save_dir / 'pba_series_ensemble.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_dir / 'pba_series_ensemble.png'}")
        plt.close()

        return mape

    return None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PBA: Series-Level Ensemble Forecast Evaluation")
    print("="*80)

    mape = evaluate_pba_ensemble()

    if mape:
        print(f"\n{'='*80}")
        print(f"Final Result: {mape:.2f}% MAPE")
        if mape < 26.71:
            print(f"Improvement: {26.71 - mape:+.2f}% points")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
