"""PCAアンサンブル予測の実装 - 複数モデルの最適重み付け組み合わせ"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    RobustBaselineForecaster
)

sns.set_style('whitegrid')


class EnsembleForecaster:
    """アンサンブル予測クラス"""

    def __init__(self, sheet_name: str):
        self.sheet_name = sheet_name
        self.prep = SalesForecastDataPreparation()
        self.ensemble_weights = {}

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

    def create_model_ensemble(self, train_data: pd.DataFrame) -> list:
        """複数のベースラインモデルを作成"""

        models = []

        # 移動平均モデル（複数窓サイズ）
        for window in [3, 6, 12]:
            if len(train_data) >= window:
                models.append(('MA' + str(window), MovingAverageForecaster(window=window)))

        # ロバスト移動平均
        for window in [6, 12]:
            if len(train_data) >= window:
                models.append(('RobustMA' + str(window), RobustBaselineForecaster(window=window)))

        # 前年同月（季節性）
        if len(train_data) >= 12:
            models.append(('SeasonalNaive', SeasonalNaiveForecaster()))

        return models

    def optimize_ensemble_weights(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                                  periods: int) -> dict:
        """検証データで最適な重みを学習"""

        models = self.create_model_ensemble(train_data)

        if len(models) == 0:
            return {}

        # 各モデルの予測を取得
        predictions = []
        model_names = []

        for name, model in models:
            try:
                model.fit(train_data)
                pred = model.predict(periods)
                merged = pd.merge(val_data[['ds', 'y']], pred[['ds', 'yhat']],
                                on='ds', how='inner')

                if len(merged) > 0:
                    predictions.append(merged['yhat'].values)
                    model_names.append(name)
            except Exception as e:
                print(f"      {name} failed: {e}")
                continue

        if len(predictions) == 0:
            return {}

        # 予測行列（各列が1つのモデルの予測）
        X = np.column_stack(predictions)
        y_true = val_data['y'].values[:len(predictions[0])]

        # 最適化: MAPE最小化
        def objective(weights):
            weights = weights / weights.sum()  # 正規化
            ensemble_pred = X @ weights
            mask = y_true > 0
            if mask.sum() == 0:
                return 1e6
            mape = np.mean(np.abs((y_true[mask] - ensemble_pred[mask]) / y_true[mask]))
            return mape

        # 制約: 重みの合計=1、各重み≥0
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}

        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        optimal_weights = result.x / result.x.sum()

        weights_dict = {name: weight for name, weight in zip(model_names, optimal_weights)}

        return weights_dict

    def forecast_series_ensemble(self, series_name: str, train_monthly: pd.DataFrame,
                                 periods: int) -> tuple:
        """シリーズのアンサンブル予測"""

        # スパイク除外データ
        baseline_data = train_monthly[~train_monthly['is_spike']].copy()

        if len(baseline_data) < 6:
            return None, None

        # ベースライン時系列
        baseline_ts = pd.DataFrame({
            'ds': baseline_data['ds'],
            'y': baseline_data['y_baseline']
        }).dropna()

        if len(baseline_ts) < 12:
            return None, None

        # 検証データ分割（最後の6か月）
        val_size = min(6, len(baseline_ts) // 4)
        train_cv = baseline_ts[:-val_size].copy()
        val_cv = baseline_ts[-val_size:].copy()

        # 重み最適化
        weights = self.optimize_ensemble_weights(train_cv, val_cv, len(val_cv))

        if len(weights) == 0:
            # フォールバック: 単一モデル
            model = RobustBaselineForecaster(window=6)
            model.fit(baseline_ts)
            forecast = model.predict(periods)
            return forecast, {'RobustMA6': 1.0}

        # 全訓練データで各モデルを学習し、アンサンブル予測
        models = self.create_model_ensemble(baseline_ts)
        ensemble_predictions = []

        for name, model in models:
            if name in weights:
                try:
                    model.fit(baseline_ts)
                    pred = model.predict(periods)
                    ensemble_predictions.append((pred, weights[name]))
                except:
                    continue

        if len(ensemble_predictions) == 0:
            return None, None

        # 重み付け平均
        forecast_df = ensemble_predictions[0][0][['ds']].copy()
        forecast_df['yhat'] = 0

        for pred, weight in ensemble_predictions:
            forecast_df['yhat'] += pred['yhat'] * weight

        return forecast_df, weights

    def hierarchical_ensemble_forecast(self, periods: int = 12) -> dict:
        """階層的アンサンブル予測"""

        print(f"\n{'='*80}")
        print(f"Ensemble Forecast: {self.sheet_name}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(self.sheet_name)
        train_df = df[df['年月'] <= '2024-05-31'].copy()
        test_df = df[df['年月'] > '2024-05-31'].copy()

        # 上位シリーズ
        ranking_path = Path('data/processed') / f'{self.sheet_name}_series_ranking.csv'
        series_ranking = pd.read_csv(ranking_path)
        top_series = series_ranking[series_ranking['cumulative_ratio'] <= 80]['モデル名'].tolist()

        print(f"\nForecasting {len(top_series)} top series with ensemble...")

        test_dates = sorted(test_df['年月'].unique())[:periods]

        series_forecasts = []
        for i, series in enumerate(top_series, 1):
            print(f"\n  [{i}/{len(top_series)}] {series}")

            # スパイク検出
            series_monthly = self.detect_series_spikes(series, train_df)
            train_series = series_monthly[series_monthly['ds'] <= '2024-05-31']

            # アンサンブル予測
            forecast, weights = self.forecast_series_ensemble(series, train_series, periods)

            if forecast is not None and weights is not None:
                print(f"    Ensemble weights:")
                for model, weight in sorted(weights.items(), key=lambda x: -x[1]):
                    if weight > 0.01:  # 1%以上の重みのみ表示
                        print(f"      {model}: {weight:.3f}")

                series_forecasts.append({
                    'series': series,
                    'forecast': forecast,
                    'weights': weights
                })

                self.ensemble_weights[series] = weights
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


def evaluate_ensemble(sheet_name: str = 'PCA'):
    """アンサンブル予測を評価"""

    print(f"\n{'='*80}")
    print(f"Ensemble Forecast Evaluation: {sheet_name}")
    print(f"{'='*80}")

    # アンサンブル予測
    ef = EnsembleForecaster(sheet_name)
    result = ef.hierarchical_ensemble_forecast(periods=12)

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

            print(f"\n{'='*80}")
            print(f"Evaluation Results (Baseline-only, Spike-excluded)")
            print(f"{'='*80}")
            print(f"MAPE: {mape:.2f}%")
            print(f"MAE:  {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")

    # 可視化
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(merged['ds'], merged['y'], 'o-',
           label='Actual', linewidth=2, markersize=8, color='black', zorder=5)

    ax.plot(merged['ds'], merged['yhat'], '^--',
           label='Ensemble Forecast', linewidth=2, markersize=6, color='purple', alpha=0.8)

    # スパイク月を強調
    spike_months = merged[merged['is_spike']]
    if len(spike_months) > 0:
        ax.scatter(spike_months['ds'], spike_months['y'], s=200, facecolors='none',
                  edgecolors='red', linewidths=3, label='Spike Months', zorder=7)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'{sheet_name}: Ensemble Forecast (Optimized Weights)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_dir = Path('visualizations/pca_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / f'{sheet_name}_ensemble_forecast.png',
               dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / f'{sheet_name}_ensemble_forecast.png'}")
    plt.close()

    # アンサンブル重みの可視化
    if len(ef.ensemble_weights) > 0:
        fig, axes = plt.subplots(1, len(ef.ensemble_weights), figsize=(12, 5))
        if len(ef.ensemble_weights) == 1:
            axes = [axes]

        for idx, (series, weights) in enumerate(ef.ensemble_weights.items()):
            ax = axes[idx]
            models = list(weights.keys())
            weights_values = list(weights.values())

            ax.bar(range(len(models)), weights_values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Weight', fontsize=10, fontweight='bold')
            ax.set_title(f'{series}\nEnsemble Weights', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')

            # 値を表示
            for i, v in enumerate(weights_values):
                if v > 0.01:
                    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_dir / f'{sheet_name}_ensemble_weights.png',
                   dpi=300, bbox_inches='tight')
        print(f"Saved: {save_dir / f'{sheet_name}_ensemble_weights.png'}")
        plt.close()

    return mape if 'mape' in locals() else None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PCA: Ensemble Forecast Evaluation")
    print("="*80)

    mape = evaluate_ensemble('PCA')

    if mape is not None:
        print(f"\n{'='*80}")
        print("Summary")
        print(f"{'='*80}")
        print(f"Previous approaches:")
        print(f"  Initial series-level (with spikes): 45.06% MAPE")
        print(f"  Spike-separated baseline: 43.37% MAPE")
        print(f"  Trend-adjusted: 52.48% MAPE (worse)")
        print(f"\nEnsemble forecast: {mape:.2f}% MAPE")

        improvement_from_spike = 43.37 - mape
        improvement_from_initial = 45.06 - mape

        print(f"\nImprovement from spike-separated: {improvement_from_spike:+.2f}% points")
        print(f"Total improvement from initial: {improvement_from_initial:+.2f}% points")


if __name__ == '__main__':
    main()
