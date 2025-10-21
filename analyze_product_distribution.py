"""製品レベルの売上分布分析スクリプト"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preparation import SalesForecastDataPreparation

sns.set_style('whitegrid')


def analyze_product_distribution(sheet_name: str):
    """製品別売上分布とパレート分析"""

    print(f"\n{'='*80}")
    print(f"{sheet_name} Category - Product Distribution Analysis")
    print(f"{'='*80}")

    # データ読み込み
    prep = SalesForecastDataPreparation()
    df = prep.load_data(sheet_name)

    # 訓練期間のデータのみ使用
    train_df = df[df['年月'] <= '2024-05-31'].copy()

    # 製品別の総売上を計算
    product_sales = train_df.groupby('製品名')['台数'].agg([
        ('total_sales', 'sum'),
        ('avg_monthly_sales', 'mean'),
        ('months_active', 'count'),
        ('max_sales', 'max'),
        ('std_sales', 'std')
    ]).reset_index()

    # 総売上でソート
    product_sales = product_sales.sort_values('total_sales', ascending=False).reset_index(drop=True)

    # 累積売上比率を計算
    total_sales = product_sales['total_sales'].sum()
    product_sales['sales_ratio'] = (product_sales['total_sales'] / total_sales) * 100
    product_sales['cumulative_ratio'] = product_sales['sales_ratio'].cumsum()
    product_sales['rank'] = range(1, len(product_sales) + 1)

    # 80%を占める製品数を特定
    top_80_count = (product_sales['cumulative_ratio'] <= 80).sum()
    top_80_products = product_sales.head(top_80_count)

    print(f"\n【Product Distribution Summary】")
    print(f"Total products: {len(product_sales)}")
    print(f"Total sales (training period): {total_sales:,.0f} units")
    print(f"Products contributing to 80% of sales: {top_80_count} ({top_80_count/len(product_sales)*100:.1f}%)")
    print(f"Products contributing to 50% of sales: {(product_sales['cumulative_ratio'] <= 50).sum()}")

    print(f"\n【Top 10 Products】")
    print(product_sales[['rank', '製品名', 'total_sales', 'avg_monthly_sales',
                         'months_active', 'sales_ratio', 'cumulative_ratio']].head(10).to_string(index=False))

    # パレート図を作成
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 棒グラフ（売上）
    x_pos = np.arange(min(30, len(product_sales)))  # 上位30製品まで表示
    ax1.bar(x_pos, product_sales['total_sales'].head(30), color='steelblue', alpha=0.7)
    ax1.set_xlabel('Product Rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Sales (units)', fontsize=12, fontweight='bold', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # 折れ線グラフ（累積比率）
    ax2 = ax1.twinx()
    ax2.plot(x_pos, product_sales['cumulative_ratio'].head(30),
             color='red', marker='o', linewidth=2, markersize=4)
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='80% Line')
    ax2.set_ylabel('Cumulative Sales Ratio (%)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')

    plt.title(f'{sheet_name} Category: Pareto Chart (Top 30 Products)',
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # 保存
    save_dir = Path('visualizations/hierarchical')
    save_dir.mkdir(exist_ok=True, parents=True)
    save_path = save_dir / f'{sheet_name}_pareto_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n  Saved: {save_path}")
    plt.close()

    # 製品リストをCSVに保存
    csv_path = Path('data/processed') / f'{sheet_name}_product_ranking.csv'
    product_sales.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Saved: {csv_path}")

    # 上位製品リストを返す
    return top_80_products


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("Product-Level Sales Distribution Analysis")
    print("="*80)

    results = {}
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        top_products = analyze_product_distribution(sheet_name)
        results[sheet_name] = top_products

    # 3カテゴリのサマリー
    print(f"\n{'='*80}")
    print("Summary: Products Contributing to 80% of Sales")
    print(f"{'='*80}")

    summary_data = []
    for category, top_products in results.items():
        prep = SalesForecastDataPreparation()
        df = prep.load_data(category)
        total_products = df['製品名'].nunique()

        summary_data.append({
            'Category': category,
            'Total Products': total_products,
            'Top 80% Products': len(top_products),
            'Concentration Ratio (%)': f"{len(top_products)/total_products*100:.1f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
