"""購買予測の実行例"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_excel_data, get_data_info
from src.preprocessor import preprocess_data
from src.train import train_model
from src.predict import predict_with_confidence


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    サンプルデータの作成

    Parameters
    ----------
    n_samples : int, default 1000
        サンプル数

    Returns
    -------
    pd.DataFrame
        サンプルデータ
    """
    np.random.seed(42)

    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'visit_count': np.random.randint(1, 50, n_samples),
        'cart_items': np.random.randint(0, 20, n_samples),
        'browsing_time': np.random.randint(1, 120, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'member_type': np.random.choice(['bronze', 'silver', 'gold'], n_samples),
    })

    # 購買有無を生成
    purchase_score = (
        df['income'] / 1000 +
        df['visit_count'] * 2 +
        df['cart_items'] * 5 +
        (df['member_type'] == 'gold').astype(int) * 50
    )
    df['purchased'] = (purchase_score > purchase_score.median()).astype(int)

    return df


def main():
    """メイン処理"""
    print("=== 購買予測システムの実行例 ===\n")

    # 1. データの準備
    print("1. データの準備")
    # 実際のExcelファイルがある場合は以下を使用
    # df = load_excel_data('data/raw/your_data.xlsx')

    # サンプルデータを作成
    df = create_sample_data(n_samples=1000)
    print(f"データ作成完了: {len(df)}行, {len(df.columns)}列\n")

    # データ情報の表示
    get_data_info(df)

    # 2. データの前処理
    print("\n2. データの前処理")
    X_train, X_test, y_train, y_test = preprocess_data(
        df,
        target_column='purchased',
        categorical_columns=['gender', 'member_type'],
        test_size=0.2,
        random_state=42
    )

    # 3. モデルの学習
    print("\n3. モデルの学習")

    # Random Forestモデル
    print("\n--- Random Forest ---")
    rf_model = train_model(
        X_train, y_train,
        X_test, y_test,
        model_type='random_forest',
        n_estimators=100,
        max_depth=10,
        random_state=42,
        save_path='models/random_forest_model.pkl'
    )

    # XGBoostモデル
    print("\n--- XGBoost ---")
    xgb_model = train_model(
        X_train, y_train,
        X_test, y_test,
        model_type='xgboost',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        save_path='models/xgboost_model.pkl'
    )

    # LightGBMモデル
    print("\n--- LightGBM ---")
    lgb_model = train_model(
        X_train, y_train,
        X_test, y_test,
        model_type='lightgbm',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        save_path='models/lightgbm_model.pkl'
    )

    # 4. 予測の実行
    print("\n4. 予測の実行")
    predictions = predict_with_confidence(rf_model.model, X_test[:10])
    print("\n予測結果（最初の10件）:")
    print(predictions)

    print("\n=== 実行完了 ===")


if __name__ == "__main__":
    main()
