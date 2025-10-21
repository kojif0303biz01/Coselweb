"""PCA: 製品レベルスパイク除去 → シリーズ集約 → 予測"""

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


class PCAProductSpikeRemoval:
    """PCA 製品レベルスパイク除去クラス"""

    def __init__(self):
        self.prep = SalesForecastDataPreparation()

    def detect_product_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """各製品のスパイクを検出"""

        products = df['製品名'].unique()
        all_spike_info = []

        print(f"\nDetecting spikes for {len(products)} products...")

        spike_count = 0
        products_with_spikes = 0

        for i, product in enumerate(products, 1):
            product_df = df[df['製品名'] == product].copy()
            monthly = product_df.groupby('年月')['台数'].sum().reset_index()
            monthly.columns = ['ds', 'y']
            monthly = monthly.sort_values('ds').reset_index(drop=True)

            if len(monthly) < 6:
                continue

            # スパイク検出（結合方式）
            is_spike = self.prep.detect_spikes(monthly['y'], method='combined',
                                              iqr_multiplier=2.0, z_threshold=2.5)

            n_spikes = is_spike.sum()
            if n_spikes > 0:
                products_with_spikes += 1
                spike_count += n_spikes

                # スパイクの最大倍率
                non_spike_median = monthly.loc[~is_spike, 'y'].median()
                if non_spike_median > 0:
                    spike_values = monthly.loc[is_spike, 'y']
                    max_multiplier = spike_values.max() / non_spike_median if len(spike_values) > 0 else 0
                else:
                    max_multiplier = 0

                if i <= 10 or n_spikes >= 3:  # 上位10製品または多スパイク製品を表示
                    print(f"  [{i}/{len(products)}] {product}: {n_spikes} spikes ({n_spikes/len(monthly)*100:.1f}%), max {max_multiplier:.1f}x")

            # スパイク情報を保存
            for idx, row in monthly.iterrows():
                all_spike_info.append({
                    '製品名': product,
                    'ds': row['ds'],
                    'y': row['y'],
                    'is_spike': is_spike.iloc[idx] if idx < len(is_spike) else False
                })

        spike_df = pd.DataFrame(all_spike_info)

        print(f"\n【Product-Level Spike Detection Summary】")
        print(f"  Products with spikes: {products_with_spikes}/{len(products)} ({products_with_spikes/len(products)*100:.1f}%)")
        print(f"  Total spike months: {spike_count}/{len(spike_df)} ({spike_count/len(spike_df)*100:.1f}%)")

        return spike_df

    def create_series_baseline(self, df: pd.DataFrame, spike_df: pd.DataFrame, train_end='2024-05-31') -> dict:
        """製品レベルスパイク除去後のシリーズレベルベースライン時系列を作成"""

        print(f"\n{'='*80}")
        print(f"Creating Series-Level Baselines (Product Spikes Removed)")
        print(f"{'='*80}")

        series_list = sorted(df['モデル名'].unique())
        series_baselines = {}

        for series_name in series_list:
            print(f"\nProcessing: {series_name}")

            series_products = df[df['モデル名'] == series_name]['製品名'].unique()
            print(f"  Products: {len(series_products)}")

            # 全ての年月を取得
            all_dates = sorted(df['年月'].unique())

            series_baseline = []

            for date in all_dates:
                date_total = 0
                n_products = 0
                n_spikes_replaced = 0

                for product in series_products:
                    # この製品のこの月のデータを取得
                    product_month_data = df[(df['製品名'] == product) & (df['年月'] == date)]

                    if len(product_month_data) == 0:
                        continue

                    product_value = product_month_data['台数'].sum()

                    # スパイク情報を確認
                    spike_info = spike_df[(spike_df['製品名'] == product) & (spike_df['ds'] == date)]

                    if len(spike_info) == 0:
                        # スパイク情報がない場合は実値を使用
                        date_total += product_value
                        n_products += 1
                    else:
                        is_spike = spike_info['is_spike'].values[0]

                        if not is_spike:
                            # スパイクでない場合は実値を使用
                            date_total += product_value
                            n_products += 1
                        else:
                            # スパイクの場合、その製品の非スパイク月の中央値で置換
                            product_spike_df = spike_df[spike_df['製品名'] == product]
                            non_spike_values = product_spike_df[~product_spike_df['is_spike']]['y'].values

                            if len(non_spike_values) > 0:
                                replacement_value = np.median(non_spike_values)
                                date_total += replacement_value
                                n_products += 1
                                n_spikes_replaced += 1
                            else:
                                # 非スパイク値がない場合は実値を使用
                                date_total += product_value
                                n_products += 1

                series_baseline.append({
                    'ds': date,
                    'y': date_total,
                    'n_products': n_products,
                    'n_spikes_replaced': n_spikes_replaced
                })

            baseline_df = pd.DataFrame(series_baseline)

            # 訓練/テスト分割
            train_baseline = baseline_df[baseline_df['ds'] <= train_end].copy()
            test_baseline = baseline_df[baseline_df['ds'] > train_end].copy()

            avg_spikes = train_baseline['n_spikes_replaced'].mean()

            print(f"  Training months: {len(train_baseline)}")
            print(f"  Test months: {len(test_baseline)}")
            print(f"  Avg spikes replaced per month: {avg_spikes:.1f}")

            series_baselines[series_name] = {
                'train': train_baseline[['ds', 'y']],
                'test': test_baseline[['ds', 'y']],
                'full': baseline_df
            }

        return series_baselines

    def forecast_series_with_ensemble(self, train_baseline: pd.DataFrame, periods: int, series_name: str) -> dict:
        """シリーズごとのアンサンブル予測"""

        print(f"\nForecasting: {series_name}")

        # モデルセット
        models = [
            ('MA3', MovingAverageForecaster(window=3)),
            ('MA6', MovingAverageForecaster(window=6)),
            ('MA12', MovingAverageForecaster(window=12)),
            ('RobustMA6', RobustBaselineForecaster(window=6)),
            ('RobustMA12', RobustBaselineForecaster(window=12)),
            ('SeasonalNaive', SeasonalNaiveForecaster())
        ]

        # 検証分割
        val_size = min(6, len(train_baseline) // 4)
        train_cv = train_baseline[:-val_size].copy()
        val_cv = train_baseline[-val_size:].copy()

        print(f"  Training: {len(train_cv)} months, Validation: {val_size} months")

        # 各モデルの予測
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

                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    mask = y_true > 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        print(f"    {name}: {mape:.2f}% MAPE")
            except Exception as e:
                print(f"    {name}: FAILED")

        if len(predictions) == 0:
            print(f"  No models succeeded, using fallback")
            mean_val = train_baseline['y'].mean()
            forecast_df = pd.DataFrame({
                'ds': pd.date_range(start=train_baseline['ds'].max() + pd.DateOffset(months=1),
                                   periods=periods, freq='MS'),
                'yhat': [mean_val] * periods
            })
            return {'forecast': forecast_df, 'weights': {'Mean': 1.0}}

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

        print(f"  【Optimized Weights】")
        for name, weight in sorted(weights_dict.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                print(f"    {name}: {weight:.3f}")

        # 全訓練データで予測
        ensemble_predictions = []
        for name, model in models:
            if name in weights_dict and weights_dict[name] > 0.001:
                try:
                    model.fit(train_baseline)
                    pred = model.predict(periods)
                    ensemble_predictions.append((pred, weights_dict[name]))
                except:
                    pass

        forecast_df = ensemble_predictions[0][0][['ds']].copy()
        forecast_df['yhat'] = 0

        for pred, weight in ensemble_predictions:
            forecast_df['yhat'] += pred['yhat'] * weight

        return {'forecast': forecast_df, 'weights': weights_dict}


def evaluate_pca_product_spike_removal():
    """PCA製品レベルスパイク除去の評価"""

    print("\n" + "="*80)
    print("PCA: Product-Level Spike Removal → Series Aggregation Evaluation")
    print("="*80)

    prep = SalesForecastDataPreparation()
    df = prep.load_data('PCA')

    psr = PCAProductSpikeRemoval()

    # 1. 製品レベルスパイク検出
    spike_df = psr.detect_product_spikes(df)

    # 2. シリーズレベルベースライン作成
    series_baselines = psr.create_series_baseline(df, spike_df)

    # 3. 各シリーズで予測
    print(f"\n{'='*80}")
    print(f"Series-Level Ensemble Forecasting")
    print(f"{'='*80}")

    all_forecasts = []

    for series_name, baseline_data in series_baselines.items():
        train_baseline = baseline_data['train']
        result = psr.forecast_series_with_ensemble(train_baseline, 12, series_name)
        all_forecasts.append(result['forecast'])

    # 4. カテゴリレベルに集約
    category_forecast = all_forecasts[0][['ds']].copy()
    category_forecast['yhat'] = 0

    for forecast_df in all_forecasts:
        category_forecast['yhat'] += forecast_df['yhat']

    # 5. 評価
    _, test_monthly = prep.create_category_aggregation('PCA')

    merged = pd.merge(test_monthly[['ds', 'y']], category_forecast[['ds', 'yhat']],
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
        print(f"  Category-level baseline: 51.13% MAPE")
        print(f"  Series-level hierarchical: 45.06% MAPE")
        print(f"  Series-level ensemble: 37.87% MAPE (current best)")
        print(f"  Product spike removal → Series: {mape:.2f}% MAPE")

        if mape < 37.87:
            improvement = 37.87 - mape
            print(f"  🎉 NEW BEST! Improvement: {improvement:+.2f}% points")
        else:
            print(f"  Change: {37.87 - mape:+.2f}% points")

        # 月別詳細
        merged['pct_error'] = ((merged['y'] - merged['yhat']) / merged['y']) * 100

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
        ax.plot(merged['ds'], merged['yhat'], '^--', label=f'Product Spike Removal (MAPE: {mape:.2f}%)',
               linewidth=2, markersize=6, color='darkblue', alpha=0.8)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
        ax.set_title(f'PCA: Product-Level Spike Removal → Series Aggregation (MAPE: {mape:.2f}%)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # グラフ2: 誤差率
        ax = axes[1]
        colors = ['green' if abs(e) < 30 else 'orange' if abs(e) < 50 else 'red'
                  for e in merged['pct_error']]
        ax.bar(merged['ds'], merged['pct_error'], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.5, label='±30%')
        ax.axhline(y=-30, color='green', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
        ax.set_title('Forecast Error by Month', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        save_dir = Path('visualizations/pca_analysis')
        save_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_dir / 'pca_product_spike_removal.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_dir / 'pca_product_spike_removal.png'}")
        plt.close()

        return mape

    return None


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PCA: Product-Level Spike Removal Evaluation")
    print("="*80)

    mape = evaluate_pca_product_spike_removal()

    if mape:
        print(f"\n{'='*80}")
        print(f"Final Result: {mape:.2f}% MAPE")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
