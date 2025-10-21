"""予測実行モジュール"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Optional


def load_model(model_path: Union[str, Path]):
    """
    保存されたモデルの読み込み

    Parameters
    ----------
    model_path : str or Path
        モデルファイルのパス

    Returns
    -------
    モデルオブジェクト
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

    model = joblib.load(model_path)
    print(f"モデル読み込み完了: {model_path}")

    return model


def predict(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    return_proba: bool = False
) -> np.ndarray:
    """
    予測の実行

    Parameters
    ----------
    model : モデルオブジェクト
        学習済みモデル
    X : np.ndarray or pd.DataFrame
        予測用特徴量
    return_proba : bool, default False
        確率を返すかどうか

    Returns
    -------
    np.ndarray
        予測結果または予測確率
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    if return_proba:
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X)
            print(f"予測完了: {len(predictions)}件（確率付き）")
        else:
            raise AttributeError("このモデルは確率予測をサポートしていません")
    else:
        predictions = model.predict(X)
        print(f"予測完了: {len(predictions)}件")

    return predictions


def predict_with_confidence(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    信頼度付きで予測を実行

    Parameters
    ----------
    model : モデルオブジェクト
        学習済みモデル
    X : np.ndarray or pd.DataFrame
        予測用特徴量
    threshold : float, default 0.5
        予測の閾値

    Returns
    -------
    pd.DataFrame
        予測結果と信頼度を含むデータフレーム
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    # 予測確率を取得
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_array)

        # 2クラス分類の場合
        if probabilities.shape[1] == 2:
            positive_proba = probabilities[:, 1]
            predictions = (positive_proba >= threshold).astype(int)

            results = pd.DataFrame({
                'prediction': predictions,
                'confidence': positive_proba,
                'will_purchase': predictions == 1
            })
        else:
            # 多クラス分類の場合
            predictions = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)

            results = pd.DataFrame({
                'prediction': predictions,
                'confidence': confidence
            })
    else:
        # 確率予測がない場合
        predictions = model.predict(X_array)
        results = pd.DataFrame({
            'prediction': predictions
        })

    print(f"予測完了: {len(results)}件")
    return results


def batch_predict(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    return_proba: bool = False
) -> pd.DataFrame:
    """
    バッチ予測の実行

    Parameters
    ----------
    model_path : str or Path
        モデルファイルのパス
    data_path : str or Path
        予測データのパス（Excel or CSV）
    output_path : str or Path, optional
        結果の保存先パス
    return_proba : bool, default False
        確率を返すかどうか

    Returns
    -------
    pd.DataFrame
        予測結果
    """
    # モデル読み込み
    model = load_model(model_path)

    # データ読み込み
    data_path = Path(data_path)
    if data_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"未対応のファイル形式: {data_path.suffix}")

    print(f"データ読み込み完了: {len(df)}行")

    # 予測実行
    predictions = predict(model, df, return_proba=return_proba)

    # 結果をデータフレームに追加
    if return_proba and predictions.ndim > 1:
        for i in range(predictions.shape[1]):
            df[f'prediction_proba_class_{i}'] = predictions[:, i]
    else:
        df['prediction'] = predictions

    # 結果の保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)

        print(f"予測結果を保存しました: {output_path}")

    return df


if __name__ == "__main__":
    # 使用例
    import sys

    if len(sys.argv) < 3:
        print("使用方法: python predict.py <model_path> <data_path> [output_path]")
        sys.exit(1)

    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    results = batch_predict(model_path, data_path, output_path, return_proba=True)
    print("\n予測結果のサマリー:")
    print(results.head())
