"""サンプルExcelデータの生成スクリプト"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_purchase_data(
    n_samples: int = 1000,
    output_path: str = 'data/raw/sample_purchase_data.xlsx'
) -> None:
    """
    購買予測用のサンプルデータを生成

    Parameters
    ----------
    n_samples : int, default 1000
        生成するサンプル数
    output_path : str
        出力ファイルパス
    """
    np.random.seed(42)

    print(f"サンプルデータ生成中... (サンプル数: {n_samples})")

    # 基本属性
    df = pd.DataFrame({
        'customer_id': [f'CUST{i:06d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['男性', '女性'], n_samples),
        'member_type': np.random.choice(['ブロンズ', 'シルバー', 'ゴールド'], n_samples, p=[0.5, 0.3, 0.2]),
        'registration_months': np.random.randint(1, 60, n_samples),
    })

    # 行動データ
    df['visit_count'] = np.random.poisson(10, n_samples) + 1
    df['page_views'] = df['visit_count'] * np.random.randint(5, 50, n_samples)
    df['browsing_time_minutes'] = np.random.randint(1, 120, n_samples)
    df['cart_items'] = np.random.poisson(3, n_samples)
    df['wishlist_items'] = np.random.poisson(5, n_samples)

    # 購買履歴
    df['past_purchases'] = np.random.poisson(5, n_samples)
    df['total_spent'] = df['past_purchases'] * np.random.randint(1000, 50000, n_samples)
    df['avg_purchase_value'] = np.where(
        df['past_purchases'] > 0,
        df['total_spent'] / df['past_purchases'],
        0
    )
    df['days_since_last_purchase'] = np.random.randint(1, 365, n_samples)

    # デモグラフィック情報
    df['income_bracket'] = np.random.choice(
        ['~300万', '300~500万', '500~700万', '700~1000万', '1000万~'],
        n_samples,
        p=[0.2, 0.3, 0.25, 0.15, 0.1]
    )
    df['occupation'] = np.random.choice(
        ['会社員', '自営業', '学生', '主婦/主夫', 'その他'],
        n_samples,
        p=[0.4, 0.2, 0.1, 0.2, 0.1]
    )
    df['prefecture'] = np.random.choice(
        ['東京', '神奈川', '大阪', '愛知', 'その他'],
        n_samples,
        p=[0.25, 0.15, 0.15, 0.1, 0.35]
    )

    # マーケティング反応
    df['email_opened'] = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    df['email_clicked'] = np.where(
        df['email_opened'] == 1,
        np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        0
    )
    df['coupon_used'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

    # 購買確率の計算（複雑なルールベース）
    purchase_score = (
        # 会員ランクの影響
        (df['member_type'] == 'ゴールド').astype(int) * 30 +
        (df['member_type'] == 'シルバー').astype(int) * 15 +

        # 行動の影響
        df['visit_count'] * 2 +
        df['cart_items'] * 8 +
        df['wishlist_items'] * 3 +

        # 過去の購買行動
        np.minimum(df['past_purchases'] * 5, 50) +

        # 収入の影響
        (df['income_bracket'] == '1000万~').astype(int) * 20 +
        (df['income_bracket'] == '700~1000万').astype(int) * 15 +
        (df['income_bracket'] == '500~700万').astype(int) * 10 +

        # マーケティング反応
        df['email_clicked'] * 25 +
        df['coupon_used'] * 30 +

        # ランダムノイズ
        np.random.randn(n_samples) * 10
    )

    # 購買フラグ（上位50%を購買とする）
    df['will_purchase'] = (purchase_score > np.median(purchase_score)).astype(int)

    # ファイル保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False, sheet_name='購買データ')

    print(f"\nサンプルデータを保存しました: {output_path}")
    print(f"\n=== データ統計 ===")
    print(f"総サンプル数: {len(df)}")
    print(f"購買予定あり: {df['will_purchase'].sum()} ({df['will_purchase'].mean()*100:.1f}%)")
    print(f"購買予定なし: {(1-df['will_purchase']).sum()} ({(1-df['will_purchase']).mean()*100:.1f}%)")
    print(f"\n=== カラム一覧 ===")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='購買予測用のサンプルExcelデータを生成')
    parser.add_argument(
        '-n', '--samples',
        type=int,
        default=1000,
        help='生成するサンプル数（デフォルト: 1000）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='data/raw/sample_purchase_data.xlsx',
        help='出力ファイルパス（デフォルト: data/raw/sample_purchase_data.xlsx）'
    )

    args = parser.parse_args()

    generate_purchase_data(n_samples=args.samples, output_path=args.output)


if __name__ == "__main__":
    main()
