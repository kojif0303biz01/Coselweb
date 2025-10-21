"""階層的予測: シリーズレベル予測 → 集計"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.data_preparation import SalesForecastDataPreparation
from src.baseline_models import (
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    RobustBaselineForecaster
)
import warnings
warnings.filterwarnings('ignore')


class HierarchicalSeriesForecaster:
    """シリーズレベル階層的予測クラス"""

    def __init__(self, sheet_name: str, top_n_series: int = None):
        """
        Parameters
        ----------
        sheet_name : str
            カテゴリ名
        top_n_series : int, optional
            予測する上位シリーズ数（Noneの場合は80%貢献シリーズ）
        """
        self.sheet_name = sheet_name
        self.top_n_series = top_n_series
        self.prep = SalesForecastDataPreparation()
        self.series_forecasts = {}

    def prepare_series_timeseries(self, df: pd.DataFrame, series_name: str) -> pd.DataFrame:
        """シリーズ別の月次時系列データを準備"""

        # シリーズデータをフィルタ
        series_df = df[df['モデル名'] == series_name].copy()

        # 月次集計
        monthly = series_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        return monthly

    def select_best_model(self, train_monthly: pd.DataFrame, periods: int) -> tuple:
        """
        各シリーズに最適なモデルを選択

        Returns
        -------
        tuple: (best_model, best_score, model_name)
        """

        if len(train_monthly) < 12:
            return None, None, None

        # 簡易CV: 最後の6か月を検証期間とする
        val_size = min(6, len(train_monthly) // 4)
        train_cv = train_monthly[:-val_size].copy()
        val_cv = train_monthly[-val_size:].copy()

        models_to_test = []

        # モデル候補
        if len(train_cv) >= 3:
            models_to_test.append(('MA3', MovingAverageForecaster(window=3)))
        if len(train_cv) >= 6:
            models_to_test.append(('MA6', MovingAverageForecaster(window=6)))
            models_to_test.append(('RobustMA6', RobustBaselineForecaster(window=6)))
        if len(train_cv) >= 12:
            models_to_test.append(('SeasonalNaive', SeasonalNaiveForecaster()))

        best_model = None
        best_score = float('inf')
        best_name = None

        for name, model in models_to_test:
            try:
                model.fit(train_cv)
                pred = model.predict(len(val_cv))

                # MAPE計算
                merged = pd.merge(val_cv[['ds', 'y']], pred[['ds', 'yhat']], on='ds', how='inner')
                if len(merged) > 0:
                    y_true = merged['y'].values
                    y_pred = merged['yhat'].values
                    mask = y_true != 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        if mape < best_score:
                            best_score = mape
                            best_model = model.__class__
                            best_name = name
            except Exception as e:
                continue

        # 最適モデルで全訓練データを学習
        if best_model is not None:
            if best_name == 'MA3':
                final_model = MovingAverageForecaster(window=3)
            elif best_name == 'MA6':
                final_model = MovingAverageForecaster(window=6)
            elif best_name == 'RobustMA6':
                final_model = RobustBaselineForecaster(window=6)
            elif best_name == 'SeasonalNaive':
                final_model = SeasonalNaiveForecaster()
            else:
                final_model = RobustBaselineForecaster(window=6)
                best_name = 'RobustMA6'

            final_model.fit(train_monthly)
            return final_model, best_score, best_name
        else:
            # デフォルト: ロバスト移動平均
            window = min(6, len(train_monthly) // 2)
            default_model = RobustBaselineForecaster(window=window)
            default_model.fit(train_monthly)
            return default_model, None, f'RobustMA{window}'

    def forecast_single_series(
        self,
        series_name: str,
        train_monthly: pd.DataFrame,
        periods: int
    ) -> tuple:
        """単一シリーズの予測"""

        # データが少ない場合はスキップ
        if len(train_monthly) < 6:
            print(f"    {series_name}: Insufficient data (< 6 months), skipping")
            return None, None

        # 売上がゼロばかりの場合もスキップ
        if train_monthly['y'].sum() == 0:
            print(f"    {series_name}: Zero sales, skipping")
            return None, None

        try:
            # 最適モデルを選択
            model, cv_score, model_name = self.select_best_model(train_monthly, periods)

            if model is None:
                return None, None

            forecast = model.predict(periods)
            return forecast, model_name

        except Exception as e:
            print(f"    {series_name}: Forecast error - {e}")
            return None, None

    def hierarchical_forecast(self, periods: int = 12) -> pd.DataFrame:
        """階層的予測の実行"""

        print(f"\n{'='*80}")
        print(f"Hierarchical Forecasting (Series-Level): {self.sheet_name}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(self.sheet_name)
        train_df = df[df['年月'] <= '2024-05-31'].copy()
        test_df = df[df['年月'] > '2024-05-31'].copy()

        # 上位シリーズリストを読み込み
        ranking_path = Path('data/processed') / f'{self.sheet_name}_series_ranking.csv'
        series_ranking = pd.read_csv(ranking_path)

        if self.top_n_series is None:
            # 80%貢献シリーズを使用
            top_series = series_ranking[series_ranking['cumulative_ratio'] <= 80]['モデル名'].tolist()
        else:
            top_series = series_ranking.head(self.top_n_series)['モデル名'].tolist()

        print(f"\nForecasting {len(top_series)} top series...")
        print(f"Series: {', '.join(top_series)}")

        # テスト期間の日付範囲を取得
        test_dates = sorted(test_df['年月'].unique())[:periods]

        # 各シリーズの予測
        successful_forecasts = []
        model_summary = []
        for i, series in enumerate(top_series, 1):
            print(f"  [{i}/{len(top_series)}] {series}...", end=" ")

            # シリーズ別時系列データ準備
            series_train = self.prepare_series_timeseries(train_df, series)

            # 予測実行
            forecast, model_name = self.forecast_single_series(series, series_train, periods)

            if forecast is not None:
                forecast['モデル名'] = series
                self.series_forecasts[series] = forecast
                successful_forecasts.append(series)
                model_summary.append({'Series': series, 'Model': model_name})
                print(f"OK (using {model_name})")
            else:
                print("SKIP")

        print(f"\nSuccessfully forecasted: {len(successful_forecasts)}/{len(top_series)} series")

        if len(model_summary) > 0:
            print("\nModel Selection Summary:")
            for item in model_summary:
                print(f"  {item['Series']}: {item['Model']}")

        # 予測結果を集計
        if len(successful_forecasts) == 0:
            print("ERROR: No successful forecasts!")
            return None

        # 全シリーズの予測を集計
        aggregated_forecast = self._aggregate_forecasts(test_dates)

        # 残りのシリーズ（Long Tail）の貢献を推定
        aggregated_forecast = self._add_longtail_forecast(
            aggregated_forecast,
            train_df,
            top_series,
            test_dates
        )

        return aggregated_forecast

    def _aggregate_forecasts(self, test_dates: list) -> pd.DataFrame:
        """個別予測を集計"""

        # 日付ごとに集計
        aggregated = []
        for date in test_dates:
            total_forecast = 0
            for series, forecast_df in self.series_forecasts.items():
                # 該当日付の予測値を取得
                date_forecast = forecast_df[forecast_df['ds'] == date]['yhat'].values
                if len(date_forecast) > 0:
                    total_forecast += date_forecast[0]

            aggregated.append({
                'ds': date,
                'yhat': total_forecast
            })

        return pd.DataFrame(aggregated)

    def _add_longtail_forecast(
        self,
        aggregated_forecast: pd.DataFrame,
        train_df: pd.DataFrame,
        top_series: list,
        test_dates: list
    ) -> pd.DataFrame:
        """Long Tailシリーズの予測を追加"""

        # Long Tailシリーズ（上位以外）の訓練期間の平均貢献率を計算
        longtail_df = train_df[~train_df['モデル名'].isin(top_series)]
        longtail_monthly = longtail_df.groupby('年月')['台数'].sum()

        if len(longtail_monthly) > 0:
            # Long Tailの平均月次売上
            longtail_avg = longtail_monthly.mean()

            # 予測に追加
            aggregated_forecast['yhat'] = aggregated_forecast['yhat'] + longtail_avg

            longtail_ratio = longtail_monthly.sum() / train_df.groupby('年月')['台数'].sum().sum()
            print(f"\nLong Tail series contribution: {longtail_ratio*100:.1f}% (avg: {longtail_avg:.0f} units/month)")

        return aggregated_forecast


def evaluate_hierarchical_series_forecast(sheet_name: str, periods: int = 12):
    """階層的シリーズ予測の評価"""

    # 階層的予測
    hf = HierarchicalSeriesForecaster(sheet_name)
    hierarchical_pred = hf.hierarchical_forecast(periods)

    if hierarchical_pred is None:
        return None

    # 実際の値を取得
    prep = SalesForecastDataPreparation()
    _, test_monthly = prep.create_category_aggregation(sheet_name)

    # マージ
    merged = pd.merge(test_monthly[['ds', 'y']], hierarchical_pred[['ds', 'yhat']],
                     on='ds', how='inner')

    if len(merged) == 0:
        print("ERROR: No matching dates!")
        return None

    # メトリクス計算
    y_true = merged['y'].values
    y_pred = merged['yhat'].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    print(f"\n{'='*80}")
    print(f"Hierarchical Series Forecast Performance: {sheet_name}")
    print(f"{'='*80}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAE:  {mae:,.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        'category': sheet_name,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'forecast': merged
    }


def main():
    """メイン処理"""

    print("\n" + "="*80)
    print("Hierarchical Forecasting (Series-Level) Evaluation")
    print("="*80)

    results = {}
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        result = evaluate_hierarchical_series_forecast(sheet_name, periods=12)
        if result is not None:
            results[sheet_name] = result

    # 結果のサマリー
    print(f"\n{'='*80}")
    print("Summary: Hierarchical Series Forecast Performance")
    print(f"{'='*80}")

    summary_data = []
    for category, result in results.items():
        summary_data.append({
            'Category': category,
            'MAPE (%)': f"{result['mape']:.2f}",
            'MAE': f"{result['mae']:,.0f}",
            'RMSE': f"{result['rmse']:,.0f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # 結果を保存
    save_path = Path('data/processed/hierarchical_series_forecast_results.csv')
    pd.DataFrame(summary_data).to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved: {save_path}")


if __name__ == '__main__':
    main()
