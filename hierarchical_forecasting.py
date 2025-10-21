"""階層的予測: 主要製品個別予測 → 集計"""

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


class HierarchicalForecaster:
    """階層的予測クラス"""

    def __init__(self, sheet_name: str, top_n_products: int = None):
        """
        Parameters
        ----------
        sheet_name : str
            カテゴリ名
        top_n_products : int, optional
            予測する上位製品数（Noneの場合は80%貢献製品）
        """
        self.sheet_name = sheet_name
        self.top_n_products = top_n_products
        self.prep = SalesForecastDataPreparation()
        self.product_forecasts = {}

    def prepare_product_timeseries(self, df: pd.DataFrame, product_name: str) -> pd.DataFrame:
        """製品別の月次時系列データを準備"""

        # 製品データをフィルタ
        product_df = df[df['製品名'] == product_name].copy()

        # 月次集計
        monthly = product_df.groupby('年月')['台数'].sum().reset_index()
        monthly.columns = ['ds', 'y']
        monthly = monthly.sort_values('ds').reset_index(drop=True)

        return monthly

    def forecast_single_product(
        self,
        product_name: str,
        train_monthly: pd.DataFrame,
        periods: int
    ) -> pd.DataFrame:
        """単一製品の予測"""

        # データが少ない場合はスキップ
        if len(train_monthly) < 12:
            print(f"    {product_name}: Insufficient data (< 12 months), skipping")
            return None

        # 売上がゼロばかりの場合もスキップ
        if train_monthly['y'].sum() == 0:
            print(f"    {product_name}: Zero sales, skipping")
            return None

        # 最適なモデルを自動選択（シンプルにロバスト移動平均を使用）
        # より高度な実装では、各製品ごとにCV で最適モデルを選択
        try:
            # ロバスト移動平均（6か月）を使用
            window = min(6, len(train_monthly) // 2)
            model = RobustBaselineForecaster(window=window, name=f"Product_{product_name}")
            model.fit(train_monthly)
            forecast = model.predict(periods)

            return forecast

        except Exception as e:
            print(f"    {product_name}: Forecast error - {e}")
            return None

    def hierarchical_forecast(self, periods: int = 12) -> pd.DataFrame:
        """階層的予測の実行"""

        print(f"\n{'='*80}")
        print(f"Hierarchical Forecasting: {self.sheet_name}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.prep.load_data(self.sheet_name)
        train_df = df[df['年月'] <= '2024-05-31'].copy()
        test_df = df[df['年月'] > '2024-05-31'].copy()

        # 上位製品リストを読み込み
        ranking_path = Path('data/processed') / f'{self.sheet_name}_product_ranking.csv'
        product_ranking = pd.read_csv(ranking_path)

        if self.top_n_products is None:
            # 80%貢献製品を使用
            top_products = product_ranking[product_ranking['cumulative_ratio'] <= 80]['製品名'].tolist()
        else:
            top_products = product_ranking.head(self.top_n_products)['製品名'].tolist()

        print(f"\nForecasting {len(top_products)} top products...")

        # テスト期間の日付範囲を取得
        test_dates = sorted(test_df['年月'].unique())[:periods]

        # 各製品の予測
        successful_forecasts = []
        for i, product in enumerate(top_products, 1):
            print(f"  [{i}/{len(top_products)}] {product}...", end=" ")

            # 製品別時系列データ準備
            product_train = self.prepare_product_timeseries(train_df, product)

            # 予測実行
            forecast = self.forecast_single_product(product, product_train, periods)

            if forecast is not None:
                forecast['製品名'] = product
                self.product_forecasts[product] = forecast
                successful_forecasts.append(product)
                print("OK")
            else:
                print("SKIP")

        print(f"\nSuccessfully forecasted: {len(successful_forecasts)}/{len(top_products)} products")

        # 予測結果を集計
        if len(successful_forecasts) == 0:
            print("ERROR: No successful forecasts!")
            return None

        # 全製品の予測を集計
        aggregated_forecast = self._aggregate_forecasts(test_dates)

        # 残りの製品（Long Tail）の貢献を推定
        aggregated_forecast = self._add_longtail_forecast(
            aggregated_forecast,
            train_df,
            top_products,
            test_dates
        )

        return aggregated_forecast

    def _aggregate_forecasts(self, test_dates: list) -> pd.DataFrame:
        """個別予測を集計"""

        # 日付ごとに集計
        aggregated = []
        for date in test_dates:
            total_forecast = 0
            for product, forecast_df in self.product_forecasts.items():
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
        top_products: list,
        test_dates: list
    ) -> pd.DataFrame:
        """Long Tail製品の予測を追加"""

        # Long Tail製品（上位以外）の訓練期間の平均貢献率を計算
        longtail_df = train_df[~train_df['製品名'].isin(top_products)]
        longtail_monthly = longtail_df.groupby('年月')['台数'].sum()

        if len(longtail_monthly) > 0:
            # Long Tailの平均月次売上
            longtail_avg = longtail_monthly.mean()

            # 予測に追加
            aggregated_forecast['yhat'] = aggregated_forecast['yhat'] + longtail_avg

            longtail_ratio = longtail_monthly.sum() / train_df.groupby('年月')['台数'].sum().sum()
            print(f"\nLong Tail products contribution: {longtail_ratio*100:.1f}% (avg: {longtail_avg:.0f} units/month)")

        return aggregated_forecast


def evaluate_hierarchical_forecast(sheet_name: str, periods: int = 12):
    """階層的予測の評価"""

    # 階層的予測
    hf = HierarchicalForecaster(sheet_name)
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
    print(f"Hierarchical Forecast Performance: {sheet_name}")
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
    print("Hierarchical Forecasting Evaluation")
    print("="*80)

    results = {}
    for sheet_name in ['TUHS', 'PCA', 'PBA']:
        result = evaluate_hierarchical_forecast(sheet_name, periods=12)
        if result is not None:
            results[sheet_name] = result

    # 結果のサマリー
    print(f"\n{'='*80}")
    print("Summary: Hierarchical Forecast Performance")
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
    save_path = Path('data/processed/hierarchical_forecast_results.csv')
    pd.DataFrame(summary_data).to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"\nResults saved: {save_path}")


if __name__ == '__main__':
    main()
