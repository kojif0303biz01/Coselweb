"""TUHS: June特有補正アプローチ"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import RobustBaselineForecaster
from scipy import stats

sns.set_style('whitegrid')


def analyze_june_trend():
    """6月のトレンドを分析"""

    print("\n" + "="*80)
    print("TUHS: June-Specific Trend Analysis")
    print("="*80)

    prep = SalesForecastDataPreparation()
    df = prep.load_data('TUHS')

    # カテゴリレベルの月次データ
    monthly = df.groupby('年月')['台数'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    monthly = monthly.sort_values('ds').reset_index(drop=True)

    # 6月のデータを抽出
    june_data = monthly[monthly['ds'].dt.month == 6].copy()
    june_data['year'] = june_data['ds'].dt.year
    june_data = june_data.sort_values('year').reset_index(drop=True)

    print(f"\nHistorical June Performance:")
    print(june_data[['year', 'y']].to_string(index=False))

    # 6月のトレンド分析（訓練期間のみ：2020-2023）
    june_train = june_data[june_data['year'] <= 2023].copy()

    if len(june_train) >= 3:
        x = june_train['year'].values
        y = june_train['y'].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        print(f"\nJune Trend (2020-2023):")
        print(f"  Slope: {slope:.2f} units/year")
        print(f"  Intercept: {intercept:.2f}")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  P-value: {p_value:.4e}")

        # 2024年6月の予測
        june_2024_pred = slope * 2024 + intercept
        june_2024_actual = june_data[june_data['year'] == 2024]['y'].values[0]

        print(f"\nJune 2024 Prediction using Linear Trend:")
        print(f"  Predicted: {june_2024_pred:,.0f} units")
        print(f"  Actual: {june_2024_actual:,.0f} units")
        print(f"  Error: {((june_2024_actual - june_2024_pred) / june_2024_actual * 100):+.1f}%")

        return slope, intercept, r_value**2
    else:
        print("Insufficient June data for trend analysis")
        return None, None, None


def forecast_with_june_correction():
    """6月補正付き予測"""

    print("\n" + "="*80)
    print("TUHS: Forecast with June-Specific Correction")
    print("="*80)

    prep = SalesForecastDataPreparation()
    train, test = prep.create_category_aggregation('TUHS')

    # ベースライン予測（RobustMA6）
    base_model = RobustBaselineForecaster(window=6, name="RobustMA6")
    base_model.fit(train)
    base_forecast = base_model.predict(len(test))

    print("\nBase Model: RobustMA6")

    # 6月のトレンドを取得
    df = prep.load_data('TUHS')
    monthly = df.groupby('年月')['台数'].sum().reset_index()
    monthly.columns = ['ds', 'y']
    june_data = monthly[monthly['ds'].dt.month == 6].copy()
    june_data['year'] = june_data['ds'].dt.year

    june_train = june_data[june_data['year'] <= 2023]
    x = june_train['year'].values
    y = june_train['y'].values

    if len(june_train) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 修正後の予測
        corrected_forecast = base_forecast.copy()

        # 6月の値を線形トレンドベースの値に置き換え
        june_months = corrected_forecast[corrected_forecast['ds'].dt.month == 6]

        for idx, row in june_months.iterrows():
            year = row['ds'].year
            june_pred = slope * year + intercept

            # 負の値にならないように下限を設定
            june_pred = max(june_pred, 20000)

            corrected_forecast.loc[idx, 'yhat'] = june_pred
            print(f"  June {year}: {row['yhat']:,.0f} → {june_pred:,.0f} (trend-based correction)")

    else:
        print("Skipping June correction due to insufficient data")
        corrected_forecast = base_forecast

    # 評価: ベースライン
    merged_base = pd.merge(test[['ds', 'y']], base_forecast[['ds', 'yhat']],
                           on='ds', how='inner')
    y_true = merged_base['y'].values
    y_pred_base = merged_base['yhat'].values
    mask = y_true > 0
    mape_base = np.mean(np.abs((y_true[mask] - y_pred_base[mask]) / y_true[mask])) * 100

    # 評価: 6月補正後
    merged_corr = pd.merge(test[['ds', 'y']], corrected_forecast[['ds', 'yhat']],
                           on='ds', how='inner')
    y_pred_corr = merged_corr['yhat'].values
    mape_corr = np.mean(np.abs((y_true[mask] - y_pred_corr[mask]) / y_true[mask])) * 100

    mae_corr = np.mean(np.abs(y_true[mask] - y_pred_corr[mask]))
    rmse_corr = np.sqrt(np.mean((y_true[mask] - y_pred_corr[mask]) ** 2))

    print(f"\n{'='*80}")
    print(f"Evaluation Results")
    print(f"{'='*80}")
    print(f"Baseline (RobustMA6): {mape_base:.2f}% MAPE")
    print(f"June-Corrected: {mape_corr:.2f}% MAPE")
    print(f"Improvement: {mape_base - mape_corr:+.2f}% points")
    print(f"\nJune-Corrected Metrics:")
    print(f"  MAPE: {mape_corr:.2f}%")
    print(f"  MAE:  {mae_corr:.2f}")
    print(f"  RMSE: {rmse_corr:.2f}")

    if mape_corr < 10.0:
        print(f"\n🎯 TARGET ACHIEVED: {mape_corr:.2f}% < 10%")
    else:
        print(f"\n⚠ Target not yet achieved: {mape_corr:.2f}% (need {mape_corr - 10.0:.2f}% more improvement)")

    # 6月を除いた場合のMAPE
    non_june_mask = (test['ds'].dt.month != 6).values
    if non_june_mask.sum() > 0:
        test_no_june = test[non_june_mask].copy()
        merged_no_june = pd.merge(test_no_june[['ds', 'y']], base_forecast[['ds', 'yhat']],
                                  on='ds', how='inner')
        y_true_nj = merged_no_june['y'].values
        y_pred_nj = merged_no_june['yhat'].values
        mask_nj = y_true_nj > 0
        mape_no_june = np.mean(np.abs((y_true_nj[mask_nj] - y_pred_nj[mask_nj]) / y_true_nj[mask_nj])) * 100

        print(f"\nFor comparison:")
        print(f"  MAPE excluding June 2024: {mape_no_june:.2f}%")
        print(f"  Impact of June 2024: {mape_base - mape_no_june:.2f}% points")

    # 月別詳細
    merged_corr['pct_error'] = ((merged_corr['y'] - merged_corr['yhat']) / merged_corr['y']) * 100

    print(f"\n{'='*80}")
    print(f"Month-by-Month Performance (June-Corrected)")
    print(f"{'='*80}")
    print(merged_corr[['ds', 'y', 'yhat', 'pct_error']].to_string(index=False))

    # 可視化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # グラフ1: 予測比較
    ax = axes[0]
    ax.plot(merged_base['ds'], merged_base['y'], 'o-', label='Actual',
           linewidth=2, markersize=8, color='black', zorder=5)
    ax.plot(merged_base['ds'], merged_base['yhat'], 's--', label=f'Baseline (RobustMA6): {mape_base:.2f}%',
           linewidth=2, markersize=6, color='red', alpha=0.7)
    ax.plot(merged_corr['ds'], merged_corr['yhat'], '^--', label=f'June-Corrected: {mape_corr:.2f}%',
           linewidth=2, markersize=6, color='green', alpha=0.8)

    # June 2024をハイライト
    june_2024 = merged_corr[merged_corr['ds'] == '2024-06-01']
    if len(june_2024) > 0:
        ax.scatter(june_2024['ds'], june_2024['y'], s=300, facecolors='none',
                  edgecolors='red', linewidth=3, label='June 2024 (outlier)', zorder=10)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=12, fontweight='bold')
    ax.set_title(f'TUHS: June-Corrected Forecast (MAPE: {mape_corr:.2f}%)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # グラフ2: 誤差率比較
    ax = axes[1]
    x = np.arange(len(merged_corr))
    width = 0.35

    merged_base_sorted = merged_base.sort_values('ds')
    merged_corr_sorted = merged_corr.sort_values('ds')

    pct_error_base = ((merged_base_sorted['y'] - merged_base_sorted['yhat']) / merged_base_sorted['y']) * 100
    pct_error_corr = ((merged_corr_sorted['y'] - merged_corr_sorted['yhat']) / merged_corr_sorted['y']) * 100

    ax.bar(x - width/2, pct_error_base, width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, pct_error_corr, width, label='June-Corrected', color='green', alpha=0.7)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=10, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-10, color='green', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Error Comparison: Baseline vs June-Corrected',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%Y-%m') for d in merged_corr_sorted['ds']], rotation=45)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_dir = Path('visualizations/tuhs_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'tuhs_june_correction.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'tuhs_june_correction.png'}")
    plt.close()

    return mape_corr


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("TUHS: June-Specific Correction Approach")
    print("Target: MAPE < 10%")
    print("="*80)

    # 1. 6月トレンド分析
    slope, intercept, r2 = analyze_june_trend()

    # 2. 6月補正付き予測
    mape = forecast_with_june_correction()


if __name__ == '__main__':
    main()
