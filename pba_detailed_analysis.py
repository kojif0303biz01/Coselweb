"""PBA: 詳細分析 - 構造的減少への対応"""

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


def analyze_pba_overview():
    """PBA概要分析"""

    print("\n" + "="*80)
    print("PBA: Overview Analysis")
    print("="*80)

    prep = SalesForecastDataPreparation()
    df = prep.load_data('PBA')
    train, test = prep.create_category_aggregation('PBA')

    print(f"\nData Overview:")
    print(f"  Total products: {df['製品名'].nunique()}")
    print(f"  Total series: {df['モデル名'].nunique()}")
    print(f"  Series list: {', '.join(sorted(df['モデル名'].unique()))}")
    print(f"  Period: {df['年月'].min()} to {df['年月'].max()}")
    print(f"  Total records: {len(df):,}")

    # カテゴリレベル統計
    print(f"\nCategory-Level Statistics:")
    print(f"  Training period: {train['ds'].min()} to {train['ds'].max()} ({len(train)} months)")
    print(f"  Test period: {test['ds'].min()} to {test['ds'].max()} ({len(test)} months)")
    print(f"  Training mean: {train['y'].mean():,.0f} units/month")
    print(f"  Training std: {train['y'].std():,.0f} units/month")
    print(f"  Training CV: {train['y'].std() / train['y'].mean() * 100:.1f}%")
    print(f"  Test mean: {test['y'].mean():,.0f} units/month")
    print(f"  Test std: {test['y'].std():,.0f} units/month")
    print(f"  Test CV: {test['y'].std() / test['y'].mean() * 100:.1f}%")

    change_pct = ((test['y'].mean() - train['y'].mean()) / train['y'].mean()) * 100
    print(f"  Change (train→test): {change_pct:+.2f}%")

    return df, train, test


def analyze_structural_decline(train, test):
    """構造的減少の詳細分析"""

    print("\n" + "="*80)
    print("PBA: Structural Decline Analysis")
    print("="*80)

    full = pd.concat([train, test], ignore_index=True)

    # 全期間のトレンド
    x = np.arange(len(full))
    y = full['y'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"\nFull Period Linear Trend:")
    print(f"  Slope: {slope:.2f} units/month")
    print(f"  Slope %/month: {(slope / full['y'].mean() * 100):.3f}%")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  P-value: {p_value:.4e}")

    if p_value < 0.05:
        total_decline_pct = (slope * len(full) / full['y'].values[0]) * 100
        print(f"  Statistically significant trend (p < 0.05)")
        print(f"  Total decline over {len(full)} months: {total_decline_pct:+.1f}%")

    # 訓練期間のトレンド
    x_train = np.arange(len(train))
    y_train = train['y'].values

    slope_train, intercept_train, r_value_train, p_value_train, std_err_train = stats.linregress(x_train, y_train)

    print(f"\nTraining Period Trend:")
    print(f"  Slope: {slope_train:.2f} units/month")
    print(f"  Slope %/month: {(slope_train / train['y'].mean() * 100):.3f}%")
    print(f"  R²: {r_value_train**2:.4f}")
    print(f"  P-value: {p_value_train:.4e}")

    # テスト期間への影響
    train_mean = train['y'].mean()
    test_mean = test['y'].mean()
    actual_decline = ((test_mean - train_mean) / train_mean) * 100

    print(f"\nTrain→Test Change:")
    print(f"  Training mean: {train_mean:,.0f} units/month")
    print(f"  Test mean: {test_mean:,.0f} units/month")
    print(f"  Actual decline: {actual_decline:+.2f}%")

    # 年別平均
    full['year'] = full['ds'].dt.year
    yearly = full.groupby('year')['y'].agg(['mean', 'sum', 'count']).reset_index()
    yearly.columns = ['Year', 'Mean', 'Sum', 'Months']

    print(f"\nYearly Performance:")
    print(yearly.to_string(index=False))

    # 年別変化率
    print(f"\nYear-over-Year Change:")
    for i in range(1, len(yearly)):
        prev = yearly.loc[i-1, 'Mean']
        curr = yearly.loc[i, 'Mean']
        change = ((curr - prev) / prev) * 100
        print(f"  {int(yearly.loc[i, 'Year'])}: {change:+.1f}%")

    return slope, r_value**2, actual_decline


def analyze_series_characteristics(df):
    """シリーズレベル特性分析"""

    print("\n" + "="*80)
    print("PBA: Series-Level Characteristics")
    print("="*80)

    series_list = sorted(df['モデル名'].unique())

    series_stats = []

    for series_name in series_list:
        series_df = df[df['モデル名'] == series_name].copy()
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']

        if len(monthly) < 12:
            continue

        # 訓練/テスト分割
        train_series = monthly[monthly['ds'] <= '2024-05-31']
        test_series = monthly[monthly['ds'] > '2024-05-31']

        # 統計
        products = series_df['製品名'].nunique()
        train_mean = train_series['y'].mean() if len(train_series) > 0 else 0
        test_mean = test_series['y'].mean() if len(test_series) > 0 else 0

        if train_mean > 0:
            cv = train_series['y'].std() / train_mean * 100
            change = ((test_mean - train_mean) / train_mean) * 100 if train_mean > 0 else 0
        else:
            cv = 0
            change = 0

        # トレンド
        if len(train_series) >= 12:
            x = np.arange(len(train_series))
            y = train_series['y'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend_pct_month = (slope / train_mean * 100) if train_mean > 0 else 0
            r_squared = r_value ** 2
        else:
            trend_pct_month = 0
            r_squared = 0

        series_stats.append({
            'Series': series_name,
            'Products': products,
            'Train_Mean': train_mean,
            'Test_Mean': test_mean,
            'CV%': cv,
            'Change%': change,
            'Trend%/mo': trend_pct_month,
            'R²': r_squared
        })

    stats_df = pd.DataFrame(series_stats)
    stats_df = stats_df.sort_values('Train_Mean', ascending=False)

    print(f"\nSeries Statistics (sorted by Training Mean):")
    print(stats_df.to_string(index=False))

    # シリーズの分類
    print(f"\nSeries Classification:")

    high_volume = stats_df[stats_df['Train_Mean'] > stats_df['Train_Mean'].median()]
    print(f"  High-volume series ({len(high_volume)}):")
    for _, row in high_volume.iterrows():
        print(f"    {row['Series']}: {row['Train_Mean']:,.0f} units/month (change: {row['Change%']:+.1f}%)")

    strong_decline = stats_df[stats_df['Change%'] < -20]
    print(f"\n  Strong decline (>20%) series ({len(strong_decline)}):")
    for _, row in strong_decline.iterrows():
        print(f"    {row['Series']}: {row['Change%']:+.1f}%")

    return stats_df


def analyze_seasonality(train, test):
    """季節性分析"""

    print("\n" + "="*80)
    print("PBA: Seasonality Analysis")
    print("="*80)

    full = pd.concat([train, test], ignore_index=True)
    full['month'] = full['ds'].dt.month

    monthly_stats = full.groupby('month')['y'].agg(['mean', 'std', 'count']).reset_index()
    monthly_stats.columns = ['Month', 'Mean', 'Std', 'Count']
    monthly_stats['CV%'] = (monthly_stats['Std'] / monthly_stats['Mean']) * 100

    print(f"\nMonthly Seasonality:")
    print(monthly_stats.to_string(index=False))

    # 季節性の強さ
    overall_mean = full['y'].mean()
    seasonal_variation = (monthly_stats['Mean'].max() - monthly_stats['Mean'].min()) / overall_mean * 100

    print(f"\nSeasonality Strength:")
    print(f"  Overall mean: {overall_mean:,.0f}")
    print(f"  Max month mean: {monthly_stats['Mean'].max():,.0f} (Month {monthly_stats.loc[monthly_stats['Mean'].idxmax(), 'Month']:.0f})")
    print(f"  Min month mean: {monthly_stats['Mean'].min():,.0f} (Month {monthly_stats.loc[monthly_stats['Mean'].idxmin(), 'Month']:.0f})")
    print(f"  Seasonal variation: {seasonal_variation:.1f}%")

    return monthly_stats


def visualize_pba_analysis(train, test, series_stats):
    """可視化"""

    print("\n" + "="*80)
    print("Creating Visualizations...")
    print("="*80)

    full = pd.concat([train, test], ignore_index=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    # グラフ1: 全期間時系列 + トレンド
    ax = axes[0, 0]
    ax.plot(train['ds'], train['y'], 'o-', label='Training Data',
           linewidth=2, markersize=3, color='blue')
    ax.plot(test['ds'], test['y'], 'o-', label='Test Data',
           linewidth=2, markersize=5, color='red')

    # トレンド線
    x = np.arange(len(full))
    y = full['y'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    trend_line = slope * x + intercept
    ax.plot(full['ds'], trend_line, '--', label=f'Linear Trend (R²={r_value**2:.3f})',
           linewidth=2, color='green', alpha=0.7)

    ax.axvline(x=pd.Timestamp('2024-05-31'), color='black', linestyle='--',
              linewidth=2, label='Train/Test Split')

    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sales Volume', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Full Time Series with Structural Decline',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=9)

    # グラフ2: 年別平均
    ax = axes[0, 1]
    full['year'] = full['ds'].dt.year
    yearly = full.groupby('year')['y'].mean().reset_index()
    yearly.columns = ['Year', 'Mean']

    colors = ['blue'] * (len(yearly) - 1) + ['red']
    ax.bar(yearly['Year'], yearly['Mean'], color=colors, alpha=0.7)

    # トレンド線
    x_year = yearly['Year'].values
    y_year = yearly['Mean'].values
    z = np.polyfit(x_year, y_year, 1)
    p = np.poly1d(z)
    ax.plot(x_year, p(x_year), "--", color='green', linewidth=2, label='Trend')

    ax.set_xlabel('Year', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Sales Volume', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Yearly Average Performance',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # グラフ3: 月別季節性
    ax = axes[1, 0]
    full['month'] = full['ds'].dt.month
    monthly_avg = full.groupby('month')['y'].mean().reset_index()
    monthly_avg.columns = ['Month', 'Mean']
    monthly_std = full.groupby('month')['y'].std().reset_index()

    ax.bar(monthly_avg['Month'], monthly_avg['Mean'], alpha=0.7, color='steelblue')
    ax.errorbar(monthly_avg['Month'], monthly_avg['Mean'], yerr=monthly_std['y'],
               fmt='none', color='black', capsize=5, alpha=0.5)

    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Sales Volume', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Monthly Seasonality (with Std Dev)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3, axis='y')

    # グラフ4: シリーズ別トレンド
    ax = axes[1, 1]
    top_series = series_stats.nlargest(8, 'Train_Mean')

    x_pos = np.arange(len(top_series))
    colors = ['red' if change < -20 else 'orange' if change < 0 else 'green'
              for change in top_series['Change%']]

    ax.barh(x_pos, top_series['Change%'], color=colors, alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(top_series['Series'], fontsize=9)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=-20, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Change % (Train→Test)', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Top 8 Series - Train→Test Change',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # グラフ5: シリーズ別ボリューム
    ax = axes[2, 0]
    top_series_sorted = top_series.sort_values('Train_Mean', ascending=True)

    x_pos = np.arange(len(top_series_sorted))
    ax.barh(x_pos, top_series_sorted['Train_Mean'], alpha=0.7, color='steelblue')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(top_series_sorted['Series'], fontsize=9)

    ax.set_xlabel('Average Sales Volume (Training)', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Top 8 Series - Volume',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # グラフ6: 変動係数 vs トレンド
    ax = axes[2, 1]
    scatter = ax.scatter(series_stats['CV%'], series_stats['Change%'],
                        s=series_stats['Train_Mean']/50, alpha=0.6,
                        c=series_stats['R²'], cmap='viridis')

    # 注釈（上位シリーズのみ）
    for _, row in top_series.head(5).iterrows():
        ax.annotate(row['Series'], (row['CV%'], row['Change%']),
                   fontsize=8, alpha=0.7)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-20, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Coefficient of Variation (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Train→Test Change (%)', fontsize=11, fontweight='bold')
    ax.set_title('PBA: Series Volatility vs Trend (size=volume, color=R²)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax, label='Trend R²')

    plt.tight_layout()
    save_dir = Path('visualizations/pba_analysis')
    save_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_dir / 'pba_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_dir / 'pba_detailed_analysis.png'}")
    plt.close()


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("PBA: Detailed Analysis - Understanding Structural Decline")
    print("="*80)

    # 1. 概要分析
    df, train, test = analyze_pba_overview()

    # 2. 構造的減少の分析
    slope, r_squared, decline_pct = analyze_structural_decline(train, test)

    # 3. シリーズレベル特性
    series_stats = analyze_series_characteristics(df)

    # 4. 季節性分析
    monthly_stats = analyze_seasonality(train, test)

    # 5. 可視化
    visualize_pba_analysis(train, test, series_stats)

    # 6. サマリーと推奨
    print("\n" + "="*80)
    print("Analysis Summary & Recommendations")
    print("="*80)

    print(f"\nKey Findings:")
    print(f"  1. Structural decline: {decline_pct:+.1f}% (train→test)")
    print(f"  2. Linear trend R²: {r_squared:.4f}")
    print(f"  3. Series count: {len(series_stats)}")
    print(f"  4. High-volume series: {len(series_stats[series_stats['Train_Mean'] > series_stats['Train_Mean'].median()])}")
    print(f"  5. Strong decline series (>20%): {len(series_stats[series_stats['Change%'] < -20])}")

    print(f"\nRecommended Approach:")

    if r_squared > 0.5:
        print(f"  ✓ Strong trend (R²={r_squared:.3f})")
        print(f"    → Use trend adjustment (detrend + forecast + retrend)")
    else:
        print(f"  ⚠ Weak trend (R²={r_squared:.3f})")
        print(f"    → Consider other approaches")

    if len(series_stats) <= 20:
        print(f"  ✓ Moderate series count ({len(series_stats)})")
        print(f"    → Series-level hierarchical forecasting is feasible")
    else:
        print(f"  ⚠ High series count ({len(series_stats)})")
        print(f"    → Category-level may be more stable")

    strong_decline = len(series_stats[series_stats['Change%'] < -20])
    if strong_decline > len(series_stats) / 2:
        print(f"  ⚠ Many series with strong decline ({strong_decline}/{len(series_stats)})")
        print(f"    → Need careful trend handling")

    print(f"\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
