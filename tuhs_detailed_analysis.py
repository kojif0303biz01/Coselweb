"""TUHS: 詳細分析 - 10%以下を達成するために"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from scipy import stats

sns.set_style('whitegrid')


def analyze_structural_changes():
    """構造的変化を分析"""

    print("\n" + "="*80)
    print("TUHS: Structural Change Analysis")
    print("="*80)

    prep = SalesForecastDataPreparation()
    train, test = prep.create_category_aggregation('TUHS')

    # 全データ
    full = pd.concat([train, test], ignore_index=True)

    # トレンド分析（訓練期間）
    train_months = len(train)
    x_train = np.arange(train_months)
    y_train = train['y'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)

    print(f"\nTraining Period Trend Analysis:")
    print(f"  Period: {train['ds'].min()} to {train['ds'].max()} ({train_months} months)")
    print(f"  Mean: {y_train.mean():,.0f} units/month")
    print(f"  Std: {y_train.std():,.0f} units/month")
    print(f"  Trend slope: {slope:.2f} units/month")
    print(f"  Trend %/month: {(slope / y_train.mean() * 100):.3f}%")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.4e}")

    # テスト期間の変化
    test_months = len(test)
    test_mean = test['y'].mean()
    train_mean = train['y'].mean()
    change_pct = ((test_mean - train_mean) / train_mean) * 100

    print(f"\nTest Period Changes:")
    print(f"  Period: {test['ds'].min()} to {test['ds'].max()} ({test_months} months)")
    print(f"  Mean: {test_mean:,.0f} units/month")
    print(f"  Change from training: {change_pct:+.2f}%")

    # 6月の分析（特に2024-06の大幅減少）
    june_data = full[full['ds'].dt.month == 6].copy()
    june_data['year'] = june_data['ds'].dt.year

    print(f"\nJune Performance by Year:")
    for _, row in june_data.iterrows():
        year = row['ds'].year
        value = row['y']
        avg = full[full['ds'].dt.year == year]['y'].mean()
        vs_avg = ((value - avg) / avg * 100) if avg > 0 else 0
        print(f"  {year} June: {value:,.0f} units (vs year avg: {vs_avg:+.1f}%)")

    # 月別季節性分析
    full['month'] = full['ds'].dt.month
    monthly_avg = full.groupby('month')['y'].agg(['mean', 'std', 'count']).reset_index()
    monthly_avg.columns = ['Month', 'Mean', 'Std', 'Count']
    monthly_avg['CV%'] = (monthly_avg['Std'] / monthly_avg['Mean']) * 100

    print(f"\nMonthly Seasonality:")
    print(monthly_avg.to_string(index=False))

    # テスト期間の異常値検出
    print(f"\nTest Period Anomaly Detection:")
    test_mean = test['y'].mean()
    test_std = test['y'].std()

    for _, row in test.iterrows():
        z_score = (row['y'] - test_mean) / test_std
        if abs(z_score) > 2:
            status = "ANOMALY" if abs(z_score) > 2.5 else "outlier"
            print(f"  {row['ds']}: {row['y']:,.0f} units (Z-score: {z_score:+.2f}) [{status}]")

    # 可視化
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # グラフ1: 全期間時系列 + トレンド線
    ax = axes[0]
    ax.plot(train['ds'], train['y'], 'o-', label='Training Data',
           linewidth=2, markersize=4, color='blue')
    ax.plot(test['ds'], test['y'], 'o-', label='Test Data',
           linewidth=2, markersize=6, color='red')

    # 訓練期間のトレンド線
    trend_line = slope * x_train + intercept
    ax.plot(train['ds'], trend_line, '--', label=f'Linear Trend (R²={r_value**2:.3f})',
           linewidth=2, color='green', alpha=0.7)

    # テスト期間の平均線
    ax.axhline(y=test_mean, color='red', linestyle=':', linewidth=2,
              alpha=0.5, label=f'Test Mean: {test_mean:,.0f}')

    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle='--',
              linewidth=2, label='Train/Test Split')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('TUHS: Full Time Series with Trend Analysis',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: 月別季節性
    ax = axes[1]
    ax.bar(monthly_avg['Month'], monthly_avg['Mean'], alpha=0.7, color='steelblue')
    ax.errorbar(monthly_avg['Month'], monthly_avg['Mean'], yerr=monthly_avg['Std'],
               fmt='none', color='black', capsize=5, alpha=0.5)
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('TUHS: Monthly Seasonality (with Std Dev)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # グラフ3: テスト期間詳細
    ax = axes[2]
    test_sorted = test.sort_values('ds')
    colors = []
    for _, row in test_sorted.iterrows():
        z_score = (row['y'] - test_mean) / test_std
        if abs(z_score) > 2.5:
            colors.append('red')
        elif abs(z_score) > 2:
            colors.append('orange')
        else:
            colors.append('green')

    ax.bar(test_sorted['ds'], test_sorted['y'], color=colors, alpha=0.7)
    ax.axhline(y=test_mean, color='black', linestyle='--', linewidth=2,
              label=f'Test Mean: {test_mean:,.0f}')
    ax.axhline(y=test_mean + 2*test_std, color='red', linestyle=':',
              linewidth=1, alpha=0.5, label='±2σ')
    ax.axhline(y=test_mean - 2*test_std, color='red', linestyle=':',
              linewidth=1, alpha=0.5)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title('TUHS: Test Period Detail (with Anomaly Detection)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    save_dir = Path('visualizations/tuhs_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'tuhs_structural_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'tuhs_structural_analysis.png'}")
    plt.close()

    return full


def try_exponential_smoothing():
    """指数平滑法を試す（トレンドと季節性を考慮）"""

    print("\n" + "="*80)
    print("TUHS: Exponential Smoothing with Trend")
    print("="*80)

    prep = SalesForecastDataPreparation()
    train, test = prep.create_category_aggregation('TUHS')

    # シンプルな指数平滑（alpha最適化）
    def simple_exp_smoothing(train_data, periods, alpha=None):
        """シンプル指数平滑法"""

        y = train_data['y'].values

        if alpha is None:
            # alpha最適化（グリッドサーチ）
            best_alpha = 0.3
            best_mse = float('inf')

            for a in np.linspace(0.1, 0.9, 17):
                # バックテスト
                predictions = []
                for i in range(6, len(y)):  # 最後の6か月で検証
                    train_subset = y[:i]
                    s = train_subset[0]
                    for val in train_subset[1:]:
                        s = a * val + (1 - a) * s
                    predictions.append(s)

                if len(predictions) >= 6:
                    actual = y[-len(predictions):]
                    mse = np.mean((np.array(predictions[-6:]) - actual[-6:]) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_alpha = a

            alpha = best_alpha
            print(f"  Optimized alpha: {alpha:.3f}")

        # 予測
        s = y[0]
        for val in y[1:]:
            s = alpha * val + (1 - alpha) * s

        # 将来予測（最後の平滑値を使用）
        forecast_values = [s] * periods

        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                               periods=periods, freq='MS'),
            'yhat': forecast_values
        })

        return forecast_df, alpha

    # Holt's Linear Trend (Double Exponential Smoothing)
    def holt_linear_trend(train_data, periods, alpha=None, beta=None):
        """Holtの線形トレンド法"""

        y = train_data['y'].values

        if alpha is None or beta is None:
            # パラメータ最適化
            best_params = (0.3, 0.1)
            best_mse = float('inf')

            for a in np.linspace(0.1, 0.9, 9):
                for b in np.linspace(0.05, 0.5, 10):
                    # バックテスト
                    if len(y) < 12:
                        continue

                    level = y[0]
                    trend = y[1] - y[0]
                    predictions = []

                    for i in range(1, len(y)):
                        # 1期先予測
                        forecast = level + trend
                        predictions.append(forecast)

                        # 更新
                        last_level = level
                        level = a * y[i] + (1 - a) * (level + trend)
                        trend = b * (level - last_level) + (1 - b) * trend

                    if len(predictions) >= 6:
                        actual = y[1:]
                        mse = np.mean((np.array(predictions[-6:]) - actual[-6:]) ** 2)
                        if mse < best_mse:
                            best_mse = mse
                            best_params = (a, b)

            alpha, beta = best_params
            print(f"  Optimized alpha: {alpha:.3f}, beta: {beta:.3f}")

        # 全データで学習
        level = y[0]
        trend = y[1] - y[0] if len(y) > 1 else 0

        for val in y[1:]:
            last_level = level
            level = alpha * val + (1 - alpha) * (level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend

        # 将来予測
        forecast_values = [level + (h + 1) * trend for h in range(periods)]

        forecast_df = pd.DataFrame({
            'ds': pd.date_range(start=train_data['ds'].max() + pd.DateOffset(months=1),
                               periods=periods, freq='MS'),
            'yhat': forecast_values
        })

        return forecast_df, alpha, beta

    print("\n1. Simple Exponential Smoothing:")
    ses_forecast, ses_alpha = simple_exp_smoothing(train, len(test))

    merged_ses = pd.merge(test[['ds', 'y']], ses_forecast, on='ds', how='inner')
    y_true = merged_ses['y'].values
    y_pred = merged_ses['yhat'].values
    mask = y_true > 0
    ses_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

    print(f"  MAPE: {ses_mape:.2f}%")

    print("\n2. Holt's Linear Trend:")
    holt_forecast, holt_alpha, holt_beta = holt_linear_trend(train, len(test))

    merged_holt = pd.merge(test[['ds', 'y']], holt_forecast, on='ds', how='inner')
    y_true = merged_holt['y'].values
    y_pred = merged_holt['yhat'].values
    mask = y_true > 0
    holt_mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.inf

    print(f"  MAPE: {holt_mape:.2f}%")

    print(f"\nComparison:")
    print(f"  Category-level (RobustMA6): 13.53% MAPE")
    print(f"  Series-level ensemble: 13.76% MAPE")
    print(f"  Simple Exponential Smoothing: {ses_mape:.2f}% MAPE")
    print(f"  Holt's Linear Trend: {holt_mape:.2f}% MAPE")

    best_method = "RobustMA6"
    best_mape = 13.53

    if ses_mape < best_mape:
        best_method = "Simple Exponential Smoothing"
        best_mape = ses_mape
        best_forecast = ses_forecast
    if holt_mape < best_mape:
        best_method = "Holt's Linear Trend"
        best_mape = holt_mape
        best_forecast = holt_forecast

    print(f"\nBest Method: {best_method} ({best_mape:.2f}% MAPE)")

    if best_mape < 10.0:
        print(f"🎯 TARGET ACHIEVED: {best_mape:.2f}% < 10%")
    else:
        print(f"⚠ Target not yet achieved: {best_mape:.2f}% (need {best_mape - 10.0:.2f}% more improvement)")


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("TUHS: Detailed Analysis for <10% MAPE Target")
    print("="*80)

    # 1. 構造的変化の分析
    full = analyze_structural_changes()

    # 2. 指数平滑法の試行
    try_exponential_smoothing()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
