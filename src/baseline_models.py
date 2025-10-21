"""ベースラインモデル - 移動平均、前年同月比、季節性ナイーブ予測"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class BaselineForecaster:
    """ベースライン予測モデル"""

    def __init__(self, name: str = "Baseline"):
        self.name = name
        self.predictions = None

    def fit(self, train_data: pd.DataFrame):
        """学習（ベースラインは基本的に学習不要）"""
        self.train_data = train_data.copy()
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        """予測"""
        raise NotImplementedError("Subclass must implement predict method")

    def evaluate(self, actual: pd.DataFrame) -> Dict[str, float]:
        """評価"""
        if self.predictions is None:
            raise ValueError("予測が実行されていません")

        # 実際のデータとマージ
        merged = pd.merge(
            actual[['ds', 'y']],
            self.predictions[['ds', 'yhat']],
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

        # MAPE（ゼロ除算を回避）
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


class MovingAverageForecaster(BaselineForecaster):
    """移動平均予測"""

    def __init__(self, window: int = 3, name: str = None):
        self.window = window
        super().__init__(name or f"MovingAverage_{window}M")

    def predict(self, periods: int) -> pd.DataFrame:
        """移動平均で予測"""
        # 最後のN期間の平均を取得
        recent_values = self.train_data['y'].tail(self.window).values
        avg_value = np.mean(recent_values)

        # 全期間同じ値で予測
        last_date = self.train_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        self.predictions = pd.DataFrame({
            'ds': future_dates,
            'yhat': avg_value
        })

        return self.predictions


class SeasonalNaiveForecaster(BaselineForecaster):
    """季節性ナイーブ予測（前年同月値）"""

    def __init__(self, seasonal_period: int = 12, name: str = "SeasonalNaive"):
        self.seasonal_period = seasonal_period
        super().__init__(name)

    def predict(self, periods: int) -> pd.DataFrame:
        """前年同月の値で予測"""
        last_date = self.train_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        predictions = []

        for future_date in future_dates:
            # 前年同月のデータを探す
            target_date = future_date - pd.DateOffset(months=self.seasonal_period)
            matching = self.train_data[self.train_data['ds'] == target_date]

            if len(matching) > 0:
                pred_value = matching['y'].values[0]
            else:
                # 前年同月がない場合は全体平均
                pred_value = self.train_data['y'].mean()

            predictions.append({
                'ds': future_date,
                'yhat': pred_value
            })

        self.predictions = pd.DataFrame(predictions)
        return self.predictions


class ExponentialSmoothingForecaster(BaselineForecaster):
    """指数平滑法"""

    def __init__(self, alpha: float = 0.3, name: str = None):
        self.alpha = alpha
        super().__init__(name or f"ExpSmoothing_α{alpha}")

    def predict(self, periods: int) -> pd.DataFrame:
        """指数平滑法で予測"""
        # 指数平滑化の計算
        values = self.train_data['y'].values
        smoothed = [values[0]]

        for i in range(1, len(values)):
            smoothed_value = self.alpha * values[i] + (1 - self.alpha) * smoothed[-1]
            smoothed.append(smoothed_value)

        # 最後の平滑化値で予測
        forecast_value = smoothed[-1]

        last_date = self.train_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        self.predictions = pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast_value
        })

        return self.predictions


class TrendForecaster(BaselineForecaster):
    """線形トレンド予測"""

    def __init__(self, name: str = "LinearTrend"):
        super().__init__(name)

    def predict(self, periods: int) -> pd.DataFrame:
        """線形トレンドで予測"""
        # 時間インデックスを作成
        self.train_data['time_idx'] = range(len(self.train_data))

        # 線形回帰
        X = self.train_data['time_idx'].values.reshape(-1, 1)
        y = self.train_data['y'].values

        # 手動で線形回帰（numpy使用）
        X_mean = X.mean()
        y_mean = y.mean()
        numerator = ((X.flatten() - X_mean) * (y - y_mean)).sum()
        denominator = ((X.flatten() - X_mean) ** 2).sum()

        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * X_mean
        else:
            slope = 0
            intercept = y_mean

        # 予測
        last_date = self.train_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        last_idx = len(self.train_data) - 1
        future_predictions = []

        for i, future_date in enumerate(future_dates, start=1):
            time_idx = last_idx + i
            pred_value = intercept + slope * time_idx
            # ネガティブにならないように
            pred_value = max(0, pred_value)

            future_predictions.append({
                'ds': future_date,
                'yhat': pred_value
            })

        self.predictions = pd.DataFrame(future_predictions)
        return self.predictions


class RobustBaselineForecaster(BaselineForecaster):
    """スパイク除外版の移動平均（ロバスト）"""

    def __init__(self, window: int = 6, name: str = None):
        self.window = window
        super().__init__(name or f"RobustMA_{window}M")

    def predict(self, periods: int) -> pd.DataFrame:
        """スパイクを除外した移動平均で予測"""
        # スパイクを除外
        if 'is_spike' in self.train_data.columns:
            non_spike_data = self.train_data[~self.train_data['is_spike']].copy()
        else:
            non_spike_data = self.train_data.copy()

        # 最後のN期間の平均（スパイク除外）
        if len(non_spike_data) >= self.window:
            recent_values = non_spike_data['y'].tail(self.window).values
            avg_value = np.mean(recent_values)
        else:
            avg_value = non_spike_data['y'].mean()

        # 予測
        last_date = self.train_data['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )

        self.predictions = pd.DataFrame({
            'ds': future_dates,
            'yhat': avg_value
        })

        return self.predictions


def run_baseline_comparison(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    category_name: str = "Category"
) -> pd.DataFrame:
    """
    複数のベースラインモデルを比較

    Parameters
    ----------
    train_data : pd.DataFrame
        訓練データ（ds, y列を含む）
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
    print(f"ベースラインモデル比較: {category_name}")
    print(f"{'='*80}\n")

    # モデルのリスト
    models = [
        MovingAverageForecaster(window=3),
        MovingAverageForecaster(window=6),
        MovingAverageForecaster(window=12),
        SeasonalNaiveForecaster(seasonal_period=12),
        ExponentialSmoothingForecaster(alpha=0.3),
        ExponentialSmoothingForecaster(alpha=0.5),
        TrendForecaster(),
        RobustBaselineForecaster(window=6),
    ]

    results = []
    periods = len(test_data)

    for model in models:
        print(f"実行中: {model.name}")

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

        print(f"  RMSE: {metrics['RMSE']:10.2f}, "
              f"MAE: {metrics['MAE']:10.2f}, "
              f"MAPE: {metrics['MAPE']:6.2f}%")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('MAPE')

    print(f"\n{'='*80}")
    print("ベースラインモデル評価結果（MAPE順）")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))

    # ベストモデル
    best_model = results_df.iloc[0]
    print(f"\n【ベストベースライン】: {best_model['モデル']}")
    print(f"  MAPE: {best_model['MAPE']:.2f}%")

    return results_df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from src.data_preparation import SalesForecastDataPreparation

    # データ準備
    prep = SalesForecastDataPreparation()

    for sheet_name in ['PCA', 'PBA', 'TUHS']:
        # カテゴリ全体の月次データ
        train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

        # ベースライン比較
        results = run_baseline_comparison(
            train_monthly,
            test_monthly,
            category_name=sheet_name
        )

        print(f"\n{'-'*80}\n")
