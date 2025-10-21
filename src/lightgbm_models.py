"""LightGBMモデル - スパイク特徴量を含む時系列予測"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesFeatureEngineer:
    """時系列特徴量エンジニアリング"""

    def __init__(self):
        self.feature_names = []

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        lags: List[int] = [1, 2, 3, 6, 12]
    ) -> pd.DataFrame:
        """ラグ特徴量の作成"""
        df = df.copy()

        for lag in lags:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
            self.feature_names.append(f'lag_{lag}')

        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        windows: List[int] = [3, 6, 12]
    ) -> pd.DataFrame:
        """移動平均特徴量の作成"""
        df = df.copy()

        for window in windows:
            # 移動平均
            df[f'rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
            self.feature_names.append(f'rolling_mean_{window}')

            # 移動標準偏差
            df[f'rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
            self.feature_names.append(f'rolling_std_{window}')

        return df

    def create_spike_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """スパイク関連特徴量の作成"""
        df = df.copy()

        if 'is_spike' in df.columns:
            # 前月がスパイクだったか
            df['prev_spike'] = df['is_spike'].shift(1).fillna(0).astype(int)
            self.feature_names.append('prev_spike')

            # 過去3か月のスパイク回数
            df['spike_count_3m'] = df['is_spike'].shift(1).rolling(window=3).sum().fillna(0)
            self.feature_names.append('spike_count_3m')

            # 過去6か月のスパイク回数
            df['spike_count_6m'] = df['is_spike'].shift(1).rolling(window=6).sum().fillna(0)
            self.feature_names.append('spike_count_6m')

            # 前回スパイクからの経過月数
            spike_indices = df[df['is_spike']].index
            months_since_spike = []

            for idx in df.index:
                previous_spikes = [s for s in spike_indices if s < idx]
                if previous_spikes:
                    months_since = idx - max(previous_spikes)
                else:
                    months_since = 999  # スパイクがない場合は大きい値

                months_since_spike.append(months_since)

            df['months_since_spike'] = months_since_spike
            self.feature_names.append('months_since_spike')

        return df

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        date_col: str = 'ds'
    ) -> pd.DataFrame:
        """時間関連特徴量の作成"""
        df = df.copy()

        # 月
        df['month'] = df[date_col].dt.month
        self.feature_names.append('month')

        # 四半期
        df['quarter'] = df[date_col].dt.quarter
        self.feature_names.append('quarter')

        # 年
        df['year'] = df[date_col].dt.year
        self.feature_names.append('year')

        # 経過月数（最初の月からの経過）
        df['months_from_start'] = (df[date_col].dt.year - df[date_col].dt.year.min()) * 12 + \
                                   (df[date_col].dt.month - df[date_col].dt.month.iloc[0])
        self.feature_names.append('months_from_start')

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds'
    ) -> pd.DataFrame:
        """全特徴量を作成"""
        self.feature_names = []

        df = df.copy()
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        df = self.create_spike_features(df)
        df = self.create_temporal_features(df, date_col)

        return df


class LightGBMForecaster:
    """LightGBM時系列予測モデル"""

    def __init__(
        self,
        name: str = "LightGBM",
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 31,
        **kwargs
    ):
        self.name = name
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.kwargs = kwargs

        self.model = None
        self.feature_engineer = TimeSeriesFeatureEngineer()
        self.feature_names = []

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'y'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """データの準備と特徴量作成"""
        # 特徴量作成
        df_features = self.feature_engineer.create_all_features(df, target_col)

        # NaN除去（ラグ特徴量のため）
        df_features = df_features.dropna()

        # 特徴量カラムを取得
        self.feature_names = self.feature_engineer.feature_names

        return df_features, self.feature_names

    def fit(self, train_data: pd.DataFrame):
        """モデルの学習"""
        print(f"\n{self.name} - 学習開始")
        print(f"  訓練データ: {len(train_data)}期間")

        # データ準備
        train_features, feature_names = self.prepare_data(train_data)

        print(f"  特徴量数: {len(feature_names)}")
        print(f"  有効訓練データ: {len(train_features)}期間（NaN除去後）")

        X_train = train_features[feature_names]
        y_train = train_features['y']

        # LightGBMモデルの作成
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            random_state=42,
            verbose=-1,
            **self.kwargs
        )

        # 学習
        self.model.fit(X_train, y_train)

        print(f"  学習完了")

        return self

    def predict(
        self,
        train_data: pd.DataFrame,
        periods: int
    ) -> pd.DataFrame:
        """予測の実行（反復予測）"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        print(f"\n{self.name} - 予測開始")

        # 訓練データのコピー
        df_extended = train_data.copy()

        # 最後の日付を取得
        last_date = df_extended['ds'].max()

        # 反復予測
        predictions = []

        for i in range(periods):
            # 次の月の日付
            next_date = last_date + pd.DateOffset(months=i+1)

            # 一時的な行を追加（予測用）
            temp_row = pd.DataFrame({
                'ds': [next_date],
                'y': [np.nan],  # 仮の値
                'is_spike': [False]  # 仮定：スパイクなし
            })

            df_temp = pd.concat([df_extended, temp_row], ignore_index=True)

            # 特徴量作成
            df_features = self.feature_engineer.create_all_features(df_temp)

            # 最後の行（予測対象）の特徴量を取得
            last_row = df_features.iloc[-1]

            # NaNチェック
            if last_row[self.feature_names].isnull().any():
                # NaNがある場合は、利用可能な特徴量のみ使用
                available_features = [f for f in self.feature_names if not pd.isna(last_row[f])]
                if len(available_features) == 0:
                    # 全てNaNの場合は前月の値を使用
                    pred_value = df_extended['y'].iloc[-1]
                else:
                    X_pred = last_row[available_features].values.reshape(1, -1)
                    # モデルが期待する特徴量数と合わない場合の処理
                    pred_value = df_extended['y'].mean()  # フォールバック
            else:
                X_pred = last_row[self.feature_names].values.reshape(1, -1)
                pred_value = self.model.predict(X_pred)[0]

            # ネガティブな予測値を0にクリップ
            pred_value = max(0, pred_value)

            # 予測結果を保存
            predictions.append({
                'ds': next_date,
                'yhat': pred_value
            })

            # 予測値を追加して次の予測に使用
            new_row = pd.DataFrame({
                'ds': [next_date],
                'y': [pred_value],
                'is_spike': [False]
            })
            df_extended = pd.concat([df_extended, new_row], ignore_index=True)

        forecast = pd.DataFrame(predictions)
        print(f"  予測完了: {periods}期間")

        return forecast

    def evaluate(self, actual: pd.DataFrame, predictions: pd.DataFrame) -> Dict[str, float]:
        """評価"""
        # 実際のデータとマージ
        merged = pd.merge(
            actual[['ds', 'y']],
            predictions[['ds', 'yhat']],
            on='ds',
            how='inner'
        )

        if len(merged) == 0:
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

        y_true = merged['y'].values
        y_pred = merged['yhat'].values

        # 評価指標
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }


def run_lightgbm_comparison(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    category_name: str = "Category"
) -> pd.DataFrame:
    """
    複数のLightGBMモデルを比較

    Parameters
    ----------
    train_data : pd.DataFrame
        訓練データ（ds, y, is_spike列を含む）
    test_data : pd.DataFrame
        テストデータ（ds, y列を含む）
    category_name : str
        カテゴリ名

    Returns
    -------
    pd.DataFrame
        各モデルの評価結果
    """
    print(f"\n{'='*80}")
    print(f"LightGBMモデル比較: {category_name}")
    print(f"{'='*80}")

    # モデルのリスト
    models = [
        LightGBMForecaster(
            name="LightGBM_Standard",
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5
        ),

        LightGBMForecaster(
            name="LightGBM_Deep",
            n_estimators=150,
            learning_rate=0.03,
            max_depth=7
        ),

        LightGBMForecaster(
            name="LightGBM_Shallow",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        ),
    ]

    results = []
    periods = len(test_data)

    for model in models:
        try:
            # 学習
            model.fit(train_data)

            # 予測
            predictions = model.predict(train_data, periods)

            # 評価
            metrics = model.evaluate(test_data, predictions)

            results.append({
                'モデル': model.name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'MAPE': metrics['MAPE']
            })

            print(f"\n  評価結果:")
            print(f"    RMSE: {metrics['RMSE']:10.2f}")
            print(f"    MAE:  {metrics['MAE']:10.2f}")
            print(f"    MAPE: {metrics['MAPE']:6.2f}%")

        except Exception as e:
            print(f"\n  エラー発生: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'モデル': model.name,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE')

    print(f"\n{'='*80}")
    print("LightGBMモデル評価結果（MAPE順）")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    # ベストモデル
    if len(results_df) > 0 and not results_df['MAPE'].isna().all():
        best_model = results_df.iloc[0]
        print(f"\n【ベストLightGBM】: {best_model['モデル']}")
        print(f"  MAPE: {best_model['MAPE']:.2f}%")

    return results_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from src.data_preparation import SalesForecastDataPreparation

    # データ準備
    prep = SalesForecastDataPreparation()

    # TUHSでテスト（小さいデータ）
    sheet_name = 'TUHS'

    # カテゴリ全体の月次データ
    train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

    # LightGBM比較
    results = run_lightgbm_comparison(
        train_monthly,
        test_monthly,
        category_name=sheet_name
    )
