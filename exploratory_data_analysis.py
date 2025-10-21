"""詳細なEDA（探索的データ分析）スクリプト"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Excelファイルを読み込み
file_path = 'data/raw/実績（5年分).xlsx'

def convert_yyyymm_to_date(yyyymm):
    """YYYYMMを日付に変換"""
    return pd.to_datetime(yyyymm.astype(str), format='%Y%m')

def analyze_sheet(sheet_name):
    """各シートの詳細分析"""
    print(f"\n{'='*80}")
    print(f"詳細分析: {sheet_name}")
    print(f"{'='*80}\n")

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 日付列の追加
    df['年月'] = convert_yyyymm_to_date(df['指定納期月度(YYYYMM)'])

    # 1. 期間の確認
    print("【1. データ期間】")
    print(f"開始月: {df['年月'].min().strftime('%Y年%m月')}")
    print(f"終了月: {df['年月'].max().strftime('%Y年%m月')}")
    print(f"期間: {(df['年月'].max() - df['年月'].min()).days // 30}か月")

    # 2. ユニークな製品数
    print(f"\n【2. 製品情報】")
    print(f"モデル（シリーズ）数: {df['モデル名'].nunique()}")
    print(f"製品数: {df['製品名'].nunique()}")
    print(f"\nモデル一覧:")
    for model in sorted(df['モデル名'].unique()):
        count = len(df[df['モデル名'] == model]['製品名'].unique())
        print(f"  - {model}: {count}製品")

    # 3. 月次販売台数の統計
    print(f"\n【3. 月次販売台数の統計】")
    monthly_sales = df.groupby('年月')['台数'].sum().sort_index()
    print(f"月次販売台数の平均: {monthly_sales.mean():.0f}台")
    print(f"月次販売台数の中央値: {monthly_sales.median():.0f}台")
    print(f"月次販売台数の標準偏差: {monthly_sales.std():.0f}台")
    print(f"最大販売月: {monthly_sales.idxmax().strftime('%Y年%m月')} ({monthly_sales.max():.0f}台)")
    print(f"最小販売月: {monthly_sales.idxmin().strftime('%Y年%m月')} ({monthly_sales.min():.0f}台)")

    # 4. マイナス台数の確認
    negative_count = (df['台数'] < 0).sum()
    if negative_count > 0:
        print(f"\n【4. マイナス台数】")
        print(f"マイナス台数のレコード数: {negative_count}")
        print(f"マイナス台数の合計: {df[df['台数'] < 0]['台数'].sum()}")

    # 5. 製品の継続性分析
    print(f"\n【5. 製品の継続性】")
    product_months = df.groupby('製品名')['年月'].agg(['min', 'max', 'count'])
    product_months['期間_月'] = ((product_months['max'] - product_months['min']).dt.days / 30).round(0)

    # 全期間販売されている製品
    full_period_products = product_months[product_months['期間_月'] >= 50]
    print(f"長期販売製品（50か月以上）: {len(full_period_products)}製品")

    # 新製品（後半に登場）
    recent_products = product_months[product_months['min'] >= pd.Timestamp('2023-01-01')]
    print(f"比較的新しい製品（2023年以降登場）: {len(recent_products)}製品")

    # 6. 訓練期間とテスト期間の分析
    print(f"\n【6. データ分割】")
    train_end = pd.Timestamp('2024-05-01')
    test_start = pd.Timestamp('2024-06-01')

    train_df = df[df['年月'] < test_start]
    test_df = df[df['年月'] >= test_start]

    print(f"訓練期間: 2020年6月 ～ 2024年5月 ({len(train_df)}レコード)")
    print(f"テスト期間: 2024年6月 ～ 2025年5月 ({len(test_df)}レコード)")

    train_monthly = train_df.groupby('年月')['台数'].sum()
    test_monthly = test_df.groupby('年月')['台数'].sum()

    print(f"訓練期間の月次平均販売台数: {train_monthly.mean():.0f}台")
    print(f"テスト期間の月次平均販売台数: {test_monthly.mean():.0f}台")
    print(f"トレンド: {((test_monthly.mean() / train_monthly.mean() - 1) * 100):+.1f}%")

    # 7. 製品の売上集中度
    print(f"\n【7. 売上集中度（パレート分析）】")
    product_total_sales = df.groupby('製品名')['台数'].sum().sort_values(ascending=False)
    cumsum_pct = (product_total_sales.cumsum() / product_total_sales.sum() * 100)

    top20_pct = cumsum_pct.iloc[:int(len(cumsum_pct)*0.2)].iloc[-1] if len(cumsum_pct) > 5 else 100
    print(f"上位20%の製品が占める売上比率: {top20_pct:.1f}%")

    top_10_products = product_total_sales.head(10)
    print(f"\n売上TOP10製品:")
    for i, (product, sales) in enumerate(top_10_products.items(), 1):
        pct = sales / product_total_sales.sum() * 100
        print(f"  {i:2d}. {product}: {sales:.0f}台 ({pct:.1f}%)")

    return df, monthly_sales

# 各シートの分析
results = {}
for sheet_name in ['PCA', 'PBA', 'TUHS']:
    df, monthly_sales = analyze_sheet(sheet_name)
    results[sheet_name] = {'df': df, 'monthly_sales': monthly_sales}

# 全体サマリー
print(f"\n{'='*80}")
print("全体サマリー")
print(f"{'='*80}\n")

total_records = sum(len(results[sheet]['df']) for sheet in results)
total_products = sum(results[sheet]['df']['製品名'].nunique() for sheet in results)

print(f"総レコード数: {total_records:,}")
print(f"総製品数: {total_products}")

for sheet in results:
    total_sales = results[sheet]['df']['台数'].sum()
    print(f"{sheet}の総販売台数: {total_sales:,.0f}台")

print("\n分析完了！")
