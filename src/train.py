"""モデル学習モジュール"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional
import xgboost as xgb
import lightgbm as lgb


class PurchasePredictionModel:
    """購買予測モデルクラス"""

    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Parameters
        ----------
        model_type : str, default 'random_forest'
            モデルタイプ ('random_forest', 'xgboost', 'lightgbm', 'logistic_regression', 'gradient_boosting')
        **kwargs
            モデルに渡すパラメータ
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)

    def _create_model(self, model_type: str, **kwargs):
        """モデルの作成"""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'random_state']}
            )
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            )
        elif model_type == 'lightgbm':
            return lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() if k not in ['max_iter', 'random_state']}
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 3),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                **{k: v for k, v in kwargs.items() if k not in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']}
            )
        else:
            raise ValueError(f"未対応のモデルタイプ: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        モデルの学習

        Parameters
        ----------
        X_train : np.ndarray
            学習用特徴量
        y_train : np.ndarray
            学習用目的変数
        """
        print(f"モデル学習開始: {self.model_type}")
        self.model.fit(X_train, y_train)
        print("モデル学習完了")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測実行

        Parameters
        ----------
        X : np.ndarray
            予測用特徴量

        Returns
        -------
        np.ndarray
            予測結果
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        予測確率の取得

        Parameters
        ----------
        X : np.ndarray
            予測用特徴量

        Returns
        -------
        np.ndarray
            予測確率
        """
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        モデルの評価

        Parameters
        ----------
        X_test : np.ndarray
            テスト用特徴量
        y_test : np.ndarray
            テスト用目的変数

        Returns
        -------
        Dict[str, float]
            評価指標の辞書
        """
        y_pred = self.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        print("\n=== モデル評価結果 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\n=== 分類レポート ===")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\n=== 混同行列 ===")
        print(confusion_matrix(y_test, y_pred))

        return metrics

    def save_model(self, file_path: str) -> None:
        """
        モデルの保存

        Parameters
        ----------
        file_path : str
            保存先のファイルパス
        """
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, file_path)
        print(f"モデル保存完了: {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        モデルの読み込み

        Parameters
        ----------
        file_path : str
            読み込むファイルパス
        """
        self.model = joblib.load(file_path)
        print(f"モデル読み込み完了: {file_path}")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    model_type: str = 'random_forest',
    save_path: Optional[str] = None,
    **kwargs
) -> PurchasePredictionModel:
    """
    モデルの学習と評価

    Parameters
    ----------
    X_train : np.ndarray
        学習用特徴量
    y_train : np.ndarray
        学習用目的変数
    X_test : np.ndarray, optional
        テスト用特徴量
    y_test : np.ndarray, optional
        テスト用目的変数
    model_type : str, default 'random_forest'
        モデルタイプ
    save_path : str, optional
        モデルの保存パス
    **kwargs
        モデルパラメータ

    Returns
    -------
    PurchasePredictionModel
        学習済みモデル
    """
    # モデル作成
    model = PurchasePredictionModel(model_type=model_type, **kwargs)

    # 学習
    model.train(X_train, y_train)

    # 評価
    if X_test is not None and y_test is not None:
        model.evaluate(X_test, y_test)

    # 保存
    if save_path:
        model.save_model(save_path)

    return model
