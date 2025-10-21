"""Prophetモデル - スパイク除外版とスパイク考慮版"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ProphetForecaster:
    """Prophet予測モデルの基底クラス"""

    def __init__(
        self,
        name: str = "Prophet",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = 'additive'
    ):
        self.name = name
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.model = None
        self.forecast = None

    def fit(self, train_data: pd.DataFrame):
        """モデルの学習"""
        print(f"\n{self.name} - 学習開始")
        print(f"  訓練データ: {len(train_data)}期間")

        # Prophetモデルの作成
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode
        )

        # 学習
        self.model.fit(train_data[['ds', 'y']])
        print(f"  学習完了")

        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """予測の実行"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")

        # 未来の日付を作成
        future = self.model.make_future_dataframe(periods=periods, freq='MS')

        # 予測
        self.forecast = self.model.predict(future)

        # 予測結果のみ抽出（テスト期間）
        forecast_only = self.forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # ネガティブな予測値を0にクリップ
        forecast_only['yhat'] = forecast_only['yhat'].clip(lower=0)
        forecast_only['yhat_lower'] = forecast_only['yhat_lower'].clip(lower=0)
        forecast_only['yhat_upper'] = forecast_only['yhat_upper'].clip(lower=0)

        print(f"  予測完了: {periods}期間")

        return forecast_only

    def evaluate(self, actual: pd.DataFrame) -> Dict[str, float]:
        """評価"""
        if self.forecast is None:
            raise ValueError("予測が実行されていません")

        # 実際のデータとマージ
        merged = pd.merge(
            actual[['ds', 'y']],
            self.forecast[['ds', 'yhat']],
            on='ds',
            how='inner'
        )

        if len(merged) == 0:
            return {'RMSE': np.nan, 'MAE': np.nan, 'MAPE': np.nan}

        y_true = merged['y'].values
        y_pred = merged['yhat'].values

        # 評価指標
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))

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


class RobustProphetForecaster(ProphetForecaster):
    """スパイク除外版Prophet（ロバスト）"""

    def __init__(
        self,
        name: str = "Prophet_Robust",
        changepoint_prior_scale: float = 0.05,
        **kwargs
    ):
        super().__init__(
            name=name,
            changepoint_prior_scale=changepoint_prior_scale,
            **kwargs
        )

    def fit(self, train_data: pd.DataFrame):
        """スパイクを除外して学習"""
        # スパイクを除外
        if 'is_spike' in train_data.columns:
            clean_data = train_data[~train_data['is_spike']].copy()
            spike_count = train_data['is_spike'].sum()
            print(f"\n{self.name} - スパイク除外版")
            print(f"  元データ: {len(train_data)}期間")
            print(f"  除外スパイク: {spike_count}個")
            print(f"  クリーンデータ: {len(clean_data)}期間")
        else:
            clean_data = train_data.copy()
            print(f"\n{self.name} - 標準版")

        # 親クラスのfitメソッドを呼び出し
        super().fit(clean_data)

        return self


class SpikeAwareProphetForecaster(ProphetForecaster):
    """スパイク考慮版Prophet"""

    def __init__(
        self,
        name: str = "Prophet_SpikeAware",
        changepoint_prior_scale: float = 0.1,
        **kwargs
    ):
        super().__init__(
            name=name,
            changepoint_prior_scale=changepoint_prior_scale,  # より柔軟に
            **kwargs
        )
        self.spike_dates = None

    def fit(self, train_data: pd.DataFrame):
        """スパイクをholidaysとして登録"""
        print(f"\n{self.name} - スパイク考慮版")
        print(f"  訓練データ: {len(train_data)}期間")

        # Prophetモデルの作成
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode
        )

        # スパイクをholidaysとして登録
        if 'is_spike' in train_data.columns:
            spike_data = train_data[train_data['is_spike']].copy()
            if len(spike_data) > 0:
                self.spike_dates = pd.DataFrame({
                    'holiday': '大量発注',
                    'ds': spike_data['ds'],
                    'lower_window': 0,
                    'upper_window': 0,
                })
                self.model = self.model.add_country_holidays(country_name='JP')

                print(f"  スパイク登録: {len(spike_data)}個")

        # 学習
        self.model.fit(train_data[['ds', 'y']])
        print(f"  学習完了")

        return self


def run_prophet_comparison(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    category_name: str = "Category"
) -> pd.DataFrame:
    """
    複数のProphetモデルを比較

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
    print(f"Prophetモデル比較: {category_name}")
    print(f"{'='*80}")

    # モデルのリスト
    models = [
        # 標準Prophet
        ProphetForecaster(
            name="Prophet_Standard",
            changepoint_prior_scale=0.05
        ),

        # ロバストProphet（スパイク除外）
        RobustProphetForecaster(
            name="Prophet_Robust",
            changepoint_prior_scale=0.05
        ),

        # スパイク考慮版
        SpikeAwareProphetForecaster(
            name="Prophet_SpikeAware",
            changepoint_prior_scale=0.1
        ),

        # 保守的設定
        ProphetForecaster(
            name="Prophet_Conservative",
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=5.0
        ),

        # 柔軟設定
        ProphetForecaster(
            name="Prophet_Flexible",
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=15.0
        ),
    ]

    results = []
    periods = len(test_data)

    for model in models:
        try:
            # 学習と予測
            model.fit(train_data)
            predictions = model.predict(periods)

            # 評価
            metrics = model.evaluate(test_data)

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
            results.append({
                'モデル': model.name,
                'RMSE': np.nan,
                'MAE': np.nan,
                'MAPE': np.nan
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE')

    print(f"\n{'='*80}")
    print("Prophetモデル評価結果（MAPE順）")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    # ベストモデル
    best_model = results_df.iloc[0]
    print(f"\n【ベストProphet】: {best_model['モデル']}")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    return results_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from src.data_preparation import SalesForecastDataPreparation

    # データ準備
    prep = SalesForecastDataPreparation()

    for sheet_name in ['TUHS', 'PCA', 'PBA']:  # TUHSから始める（小さいデータ）
        # カテゴリ全体の月次データ
        train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

        # Prophet比較
        results = run_prophet_comparison(
            train_monthly,
            test_monthly,
            category_name=sheet_name
        )

        print(f"\n{'-'*80}\n")

        # 一旦1カテゴリだけテスト
        if sheet_name == 'TUHS':
            print("\n初回テスト完了。全カテゴリを実行するにはコメントを解除してください。\n")
            break
