"""Excelファイルの構造を分析するスクリプト"""

import pandas as pd
import sys

# Excelファイルを読み込み
file_path = 'data/raw/実績（5年分).xlsx'

# すべてのシート名を取得
xl_file = pd.ExcelFile(file_path)
sheet_names = xl_file.sheet_names

print("=" * 80)
print(f"ファイル: {file_path}")
print(f"シート数: {len(sheet_names)}")
print(f"シート名: {sheet_names}")
print("=" * 80)

# 各シートの内容を確認
for sheet_name in sheet_names:
    print(f"\n{'=' * 80}")
    print(f"シート名: {sheet_name}")
    print("=" * 80)

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print(f"\n形状: {df.shape} (行数: {df.shape[0]}, 列数: {df.shape[1]})")

    print(f"\nカラム名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nデータ型:")
    print(df.dtypes)

    print(f"\n最初の5行:")
    print(df.head())

    print(f"\n最後の5行:")
    print(df.tail())

    print(f"\n基本統計量:")
    print(df.describe())

    print(f"\n欠損値:")
    print(df.isnull().sum())
