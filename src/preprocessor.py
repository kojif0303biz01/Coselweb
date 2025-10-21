"""データ前処理モジュール"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional


class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        欠損値処理

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム
        strategy : str, default 'mean'
            欠損値の補完方法 ('mean', 'median', 'mode', 'drop')

        Returns
        -------
        pd.DataFrame
            欠損値処理後のデータフレーム
        """
        df = df.copy()

        if strategy == 'drop':
            df = df.dropna()
        else:
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in ['int64', 'float64']:
                        if strategy == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif strategy == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # カテゴリカル変数は最頻値で補完
                        df[col].fillna(df[col].mode()[0], inplace=True)

        print(f"欠損値処理完了（strategy: {strategy}）")
        return df

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        カテゴリカル変数のエンコーディング

        Parameters
        ----------
        df : pd.DataFrame
            データフレーム
        categorical_columns : List[str], optional
            エンコードする列名のリスト

        Returns
        -------
        pd.DataFrame
            エンコード後のデータフレーム
        """
        df = df.copy()

        if categorical_columns is None:
            # オブジェクト型の列を自動検出
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

        print(f"カテゴリカル変数エンコード完了: {categorical_columns}")
        return df

    def scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True
    ) -> np.ndarray:
        """
        特徴量のスケーリング

        Parameters
        ----------
        X : pd.DataFrame
            特徴量データ
        fit : bool, default True
            スケーラーをfitするかどうか

        Returns
        -------
        np.ndarray
            スケーリング後の特徴量
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            print("特徴量のスケーリング完了（fit + transform）")
        else:
            X_scaled = self.scaler.transform(X)
            print("特徴量のスケーリング完了（transform）")

        return X_scaled

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        学習用データの準備

        Parameters
        ----------
        df : pd.DataFrame
            元データ
        target_column : str
            目的変数の列名
        test_size : float, default 0.2
            テストデータの割合
        random_state : int, default 42
            ランダムシード

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        self.target_column = target_column
        self.feature_columns = [col for col in df.columns if col != target_column]

        X = df[self.feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"データ分割完了:")
        print(f"  学習データ: {len(X_train)}行")
        print(f"  テストデータ: {len(X_test)}行")

        return X_train, X_test, y_train, y_test


def preprocess_data(
    df: pd.DataFrame,
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    missing_strategy: str = 'mean',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    データの前処理を一括実行

    Parameters
    ----------
    df : pd.DataFrame
        元データ
    target_column : str
        目的変数の列名
    categorical_columns : List[str], optional
        カテゴリカル変数の列名リスト
    missing_strategy : str, default 'mean'
        欠損値の補完方法
    test_size : float, default 0.2
        テストデータの割合
    random_state : int, default 42
        ランダムシード

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, X_test, y_train, y_test
    """
    preprocessor = DataPreprocessor()

    # 欠損値処理
    df = preprocessor.handle_missing_values(df, strategy=missing_strategy)

    # カテゴリカル変数のエンコーディング
    df = preprocessor.encode_categorical_features(df, categorical_columns)

    # データ分割
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        df, target_column, test_size, random_state
    )

    # スケーリング
    X_train_scaled = preprocessor.scale_features(X_train, fit=True)
    X_test_scaled = preprocessor.scale_features(X_test, fit=False)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values
