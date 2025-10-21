"""一時的な大量発注（スパイク）の分析スクリプト"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Excelファイルを読み込み
file_path = 'data/raw/実績（5年分).xlsx'

def convert_yyyymm_to_date(yyyymm):
    """YYYYMMを日付に変換"""
    return pd.to_datetime(yyyymm.astype(str), format='%Y%m')

def detect_spikes_statistical(series, threshold=3):
    """
    統計的手法でスパイクを検出

    Parameters
    ----------
    series : pd.Series
        時系列データ
    threshold : float
        標準偏差の倍数（デフォルト: 3σ）

    Returns
    -------
    pd.Series
        スパイクのブール値
    """
    # Zスコア法
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold

def detect_spikes_iqr(series, multiplier=1.5):
    """
    IQR（四分位範囲）法でスパイクを検出

    Parameters
    ----------
    series : pd.Series
        時系列データ
    multiplier : float
        IQRの倍数（デフォルト: 1.5）

    Returns
    -------
    pd.Series
        スパイクのブール値
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (series < lower_bound) | (series > upper_bound)

def analyze_product_spikes(sheet_name, top_n_products=20):
    """製品別のスパイク分析"""

    print(f"\n{'='*80}")
    print(f"スパイク分析: {sheet_name}")
    print(f"{'='*80}\n")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['年月'] = convert_yyyymm_to_date(df['指定納期月度(YYYYMM)'])

    # 月次・製品別の集計
    monthly_product = df.groupby(['年月', '製品名'])['台数'].sum().reset_index()

    # 上位製品を選択
    top_products = df.groupby('製品名')['台数'].sum().nlargest(top_n_products).index

    spike_summary = []

    for product in top_products:
        product_data = monthly_product[monthly_product['製品名'] == product].copy()
        product_data = product_data.sort_values('年月')

        if len(product_data) < 5:  # データが少なすぎる場合はスキップ
            continue

        # スパイク検出（IQR法）
        sales = product_data['台数'].values
        is_spike_iqr = detect_spikes_iqr(pd.Series(sales), multiplier=2.0)

        # スパイク検出（Zスコア法）
        is_spike_z = detect_spikes_statistical(pd.Series(sales), threshold=2.5)

        # 両方で検出されたものを確実なスパイクとする
        is_spike_combined = is_spike_iqr | is_spike_z

        if is_spike_combined.sum() > 0:
            product_data['is_spike'] = is_spike_combined.values
            spike_months = product_data[product_data['is_spike']]

            mean_sales = sales[~is_spike_combined].mean()

            for idx, spike in spike_months.iterrows():
                spike_ratio = spike['台数'] / mean_sales if mean_sales > 0 else 0

                spike_summary.append({
                    '製品名': product,
                    'スパイク月': spike['年月'].strftime('%Y年%m月'),
                    '販売台数': int(spike['台数']),
                    '通常平均': int(mean_sales),
                    '倍率': spike_ratio
                })

    if spike_summary:
        spike_df = pd.DataFrame(spike_summary)
        spike_df = spike_df.sort_values('倍率', ascending=False)

        print(f"【検出されたスパイク数】: {len(spike_df)}\n")
        print(f"【TOP20 大量発注スパイク】")
        print(spike_df.head(20).to_string(index=False))

        print(f"\n【スパイクの統計】")
        print(f"平均倍率: {spike_df['倍率'].mean():.2f}x")
        print(f"中央値倍率: {spike_df['倍率'].median():.2f}x")
        print(f"最大倍率: {spike_df['倍率'].max():.2f}x")

        return spike_df
    else:
        print("明確なスパイクは検出されませんでした")
        return pd.DataFrame()

def analyze_category_spikes(sheet_name):
    """カテゴリ全体のスパイク分析"""

    print(f"\n{'='*80}")
    print(f"カテゴリ全体のスパイク分析: {sheet_name}")
    print(f"{'='*80}\n")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['年月'] = convert_yyyymm_to_date(df['指定納期月度(YYYYMM)'])

    # 月次総販売台数
    monthly_total = df.groupby('年月')['台数'].sum().sort_index()

    # スパイク検出
    is_spike_iqr = detect_spikes_iqr(monthly_total, multiplier=1.5)
    is_spike_z = detect_spikes_statistical(monthly_total, threshold=2.0)
    is_spike = is_spike_iqr | is_spike_z

    print(f"【月次総販売台数の統計】")
    print(f"平均: {monthly_total.mean():.0f}台")
    print(f"中央値: {monthly_total.median():.0f}台")
    print(f"標準偏差: {monthly_total.std():.0f}台")

    if is_spike.sum() > 0:
        print(f"\n【検出されたスパイク月】: {is_spike.sum()}か月")
        spike_months = monthly_total[is_spike]

        for month, sales in spike_months.items():
            ratio = sales / monthly_total.median()
            direction = "↑ 高" if sales > monthly_total.median() else "↓ 低"
            print(f"  {month.strftime('%Y年%m月')}: {sales:.0f}台 ({direction}, 中央値の{ratio:.2f}倍)")
    else:
        print("\n明確なスパイクは検出されませんでした")

    return monthly_total, is_spike

def analyze_consecutive_spikes(sheet_name):
    """連続したスパイク（一時的な大量発注キャンペーンなど）の検出"""

    print(f"\n{'='*80}")
    print(f"連続スパイクパターン分析: {sheet_name}")
    print(f"{'='*80}\n")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['年月'] = convert_yyyymm_to_date(df['指定納期月度(YYYYMM)'])

    # 月次・製品別の集計
    monthly_product = df.groupby(['年月', '製品名'])['台数'].sum().reset_index()

    # 上位製品
    top_products = df.groupby('製品名')['台数'].sum().nlargest(30).index

    consecutive_patterns = []

    for product in top_products:
        product_data = monthly_product[monthly_product['製品名'] == product].copy()
        product_data = product_data.sort_values('年月')

        if len(product_data) < 5:
            continue

        sales = product_data['台数'].values
        median_sales = np.median(sales)

        # 中央値の2倍を超える月を検出
        high_sales_mask = sales > median_sales * 2

        if high_sales_mask.sum() >= 2:  # 2か月以上ある場合
            # 連続性をチェック
            product_data['高販売'] = high_sales_mask
            product_data['グループ'] = (product_data['高販売'] != product_data['高販売'].shift()).cumsum()

            # 高販売月のグループを抽出
            high_groups = product_data[product_data['高販売']].groupby('グループ')

            for group_id, group_data in high_groups:
                if len(group_data) >= 2:  # 2か月以上連続
                    consecutive_patterns.append({
                        '製品名': product,
                        '開始月': group_data['年月'].min().strftime('%Y年%m月'),
                        '終了月': group_data['年月'].max().strftime('%Y年%m月'),
                        '期間': len(group_data),
                        '平均販売台数': group_data['台数'].mean(),
                        '通常中央値': median_sales,
                        '倍率': group_data['台数'].mean() / median_sales
                    })

    if consecutive_patterns:
        pattern_df = pd.DataFrame(consecutive_patterns)
        pattern_df = pattern_df.sort_values('倍率', ascending=False)

        print(f"【検出された連続高販売パターン】: {len(pattern_df)}パターン\n")
        print(pattern_df.head(15).to_string(index=False))

        return pattern_df
    else:
        print("明確な連続高販売パターンは検出されませんでした")
        return pd.DataFrame()

# 各カテゴリの分析
for sheet_name in ['PCA', 'PBA', 'TUHS']:
    # 製品別スパイク
    product_spikes = analyze_product_spikes(sheet_name, top_n_products=30)

    # カテゴリ全体のスパイク
    monthly_total, category_spikes = analyze_category_spikes(sheet_name)

    # 連続スパイクパターン
    consecutive = analyze_consecutive_spikes(sheet_name)

    print("\n" + "="*80 + "\n")

print("スパイク分析完了！")
