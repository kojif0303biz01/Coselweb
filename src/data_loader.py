"""Excelデータの読み込みモジュール"""

import pandas as pd
from pathlib import Path
from typing import Union


def load_excel_data(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> pd.DataFrame:
    """
    Excelファイルからデータを読み込む

    Parameters
    ----------
    file_path : str or Path
        Excelファイルのパス
    sheet_name : str or int, default 0
        読み込むシート名またはインデックス
    **kwargs
        pd.read_excelに渡す追加のパラメータ

    Returns
    -------
    pd.DataFrame
        読み込んだデータフレーム

    Examples
    --------
    >>> df = load_excel_data('data/raw/sales_data.xlsx')
    >>> df = load_excel_data('data/raw/sales_data.xlsx', sheet_name='Sheet1')
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    print(f"データ読み込み中: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    print(f"データ読み込み完了: {len(df)}行, {len(df.columns)}列")
    print(f"カラム: {list(df.columns)}")

    return df


def load_csv_data(
    file_path: Union[str, Path],
    **kwargs
) -> pd.DataFrame:
    """
    CSVファイルからデータを読み込む

    Parameters
    ----------
    file_path : str or Path
        CSVファイルのパス
    **kwargs
        pd.read_csvに渡す追加のパラメータ

    Returns
    -------
    pd.DataFrame
        読み込んだデータフレーム
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    print(f"データ読み込み中: {file_path}")
    df = pd.read_csv(file_path, **kwargs)

    print(f"データ読み込み完了: {len(df)}行, {len(df.columns)}列")

    return df


def get_data_info(df: pd.DataFrame) -> None:
    """
    データフレームの基本情報を表示

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム
    """
    print("\n=== データ情報 ===")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"\n=== カラム情報 ===")
    print(df.info())
    print(f"\n=== 欠損値 ===")
    print(df.isnull().sum())
    print(f"\n=== 統計量 ===")
    print(df.describe())
