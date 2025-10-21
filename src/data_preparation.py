"""データ準備パイプライン - スパイク検出を含む"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SalesForecastDataPreparation:
    """販売予測用のデータ準備クラス"""

    def __init__(self, file_path: str = 'data/raw/実績（5年分).xlsx'):
        self.file_path = file_path
        self.data = {}
        self.spike_info = {}

    def load_data(self, sheet_name: str) -> pd.DataFrame:
        """データの読み込み"""
        print(f"\n{'='*80}")
        print(f"データ読み込み: {sheet_name}")
        print(f"{'='*80}")

        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        df['年月'] = pd.to_datetime(df['指定納期月度(YYYYMM)'].astype(str), format='%Y%m')

        print(f"レコード数: {len(df):,}")
        print(f"期間: {df['年月'].min().strftime('%Y-%m')} ～ {df['年月'].max().strftime('%Y-%m')}")
        print(f"製品数: {df['製品名'].nunique()}")

        self.data[sheet_name] = df
        return df

    def detect_spikes(
        self,
        series: pd.Series,
        method: str = 'combined',
        iqr_multiplier: float = 2.0,
        z_threshold: float = 2.5
    ) -> pd.Series:
        """
        スパイクの検出

        Parameters
        ----------
        series : pd.Series
            時系列データ
        method : str
            検出方法 ('iqr', 'zscore', 'combined')
        iqr_multiplier : float
            IQR法の倍数
        z_threshold : float
            Zスコア法の閾値

        Returns
        -------
        pd.Series
            スパイクのブール値
        """
        if len(series) < 5:
            return pd.Series([False] * len(series), index=series.index)

        # IQR法
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        is_spike_iqr = (series < lower_bound) | (series > upper_bound)

        # Zスコア法
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        is_spike_z = z_scores > z_threshold

        if method == 'iqr':
            return is_spike_iqr
        elif method == 'zscore':
            return is_spike_z
        else:  # combined
            return is_spike_iqr | is_spike_z

    def prepare_time_series_data(
        self,
        sheet_name: str,
        train_end_date: str = '2024-05-31',
        detect_product_spikes: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        時系列予測用のデータ準備

        Parameters
        ----------
        sheet_name : str
            シート名
        train_end_date : str
            訓練データの終了日
        detect_product_spikes : bool
            製品別にスパイクを検出するか

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train_data, test_data
        """
        if sheet_name not in self.data:
            self.load_data(sheet_name)

        df = self.data[sheet_name].copy()

        # 月次・製品別の集計
        monthly_product = df.groupby(['年月', '製品名', 'モデル名'])['台数'].sum().reset_index()

        # スパイク検出（製品別）
        if detect_product_spikes:
            print(f"\nスパイク検出中...")
            spike_flags = []

            for product in monthly_product['製品名'].unique():
                product_data = monthly_product[monthly_product['製品名'] == product].copy()
                product_data = product_data.sort_values('年月')

                if len(product_data) >= 5:
                    is_spike = self.detect_spikes(product_data['台数'])
                    product_data['is_spike'] = is_spike.values
                else:
                    product_data['is_spike'] = False

                spike_flags.append(product_data)

            monthly_product = pd.concat(spike_flags, ignore_index=True)

            # スパイク統計
            spike_count = monthly_product['is_spike'].sum()
            spike_ratio = spike_count / len(monthly_product) * 100
            print(f"検出されたスパイク: {spike_count}個 ({spike_ratio:.1f}%)")

        else:
            monthly_product['is_spike'] = False

        # 訓練/テストデータの分割
        train_end = pd.Timestamp(train_end_date)
        train_data = monthly_product[monthly_product['年月'] <= train_end].copy()
        test_data = monthly_product[monthly_product['年月'] > train_end].copy()

        print(f"\n訓練データ: {train_data['年月'].min().strftime('%Y-%m')} ～ "
              f"{train_data['年月'].max().strftime('%Y-%m')} ({len(train_data):,}レコード)")
        print(f"テストデータ: {test_data['年月'].min().strftime('%Y-%m')} ～ "
              f"{test_data['年月'].max().strftime('%Y-%m')} ({len(test_data):,}レコード)")

        return train_data, test_data

    def create_category_aggregation(
        self,
        sheet_name: str,
        train_end_date: str = '2024-05-31'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        カテゴリ全体の月次集計データを作成

        Parameters
        ----------
        sheet_name : str
            シート名
        train_end_date : str
            訓練データの終了日

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train_monthly, test_monthly
        """
        if sheet_name not in self.data:
            self.load_data(sheet_name)

        df = self.data[sheet_name].copy()

        # 月次総販売台数
        monthly_total = df.groupby('年月')['台数'].sum().reset_index()
        monthly_total.columns = ['ds', 'y']  # Prophet用の列名
        monthly_total = monthly_total.sort_values('ds')

        # スパイク検出
        is_spike = self.detect_spikes(monthly_total['y'])
        monthly_total['is_spike'] = is_spike

        # 訓練/テストデータの分割
        train_end = pd.Timestamp(train_end_date)
        train_monthly = monthly_total[monthly_total['ds'] <= train_end].copy()
        test_monthly = monthly_total[monthly_total['ds'] > train_end].copy()

        print(f"\n【カテゴリ全体の月次データ】")
        print(f"訓練期間: {train_monthly['ds'].min().strftime('%Y-%m')} ～ "
              f"{train_monthly['ds'].max().strftime('%Y-%m')} ({len(train_monthly)}か月)")
        print(f"テスト期間: {test_monthly['ds'].min().strftime('%Y-%m')} ～ "
              f"{test_monthly['ds'].max().strftime('%Y-%m')} ({len(test_monthly)}か月)")
        print(f"訓練期間の平均販売台数: {train_monthly['y'].mean():.0f}台/月")
        print(f"テスト期間の平均販売台数: {test_monthly['y'].mean():.0f}台/月")

        return train_monthly, test_monthly

    def get_top_products(
        self,
        sheet_name: str,
        top_n: int = 30
    ) -> pd.DataFrame:
        """
        上位製品のリストを取得

        Parameters
        ----------
        sheet_name : str
            シート名
        top_n : int
            上位N製品

        Returns
        -------
        pd.DataFrame
            上位製品の情報
        """
        if sheet_name not in self.data:
            self.load_data(sheet_name)

        df = self.data[sheet_name]

        # 製品別の総販売台数
        product_sales = df.groupby(['製品名', 'モデル名'])['台数'].agg([
            ('総販売台数', 'sum'),
            ('販売月数', 'count'),
            ('平均月次販売', 'mean'),
            ('最大月次販売', 'max')
        ]).reset_index()

        product_sales = product_sales.sort_values('総販売台数', ascending=False)
        top_products = product_sales.head(top_n)

        print(f"\n【上位{top_n}製品】")
        for i, row in top_products.head(10).iterrows():
            print(f"{i+1:2d}. {row['製品名']:30s} "
                  f"総販売: {row['総販売台数']:7.0f}台 "
                  f"平均: {row['平均月次販売']:5.0f}台/月")

        return top_products


if __name__ == "__main__":
    # 使用例
    prep = SalesForecastDataPreparation()

    for sheet_name in ['PCA', 'PBA', 'TUHS']:
        print(f"\n{'#'*80}")
        print(f"# {sheet_name} カテゴリのデータ準備")
        print(f"{'#'*80}")

        # データ読み込み
        df = prep.load_data(sheet_name)

        # カテゴリ全体の月次データ
        train_monthly, test_monthly = prep.create_category_aggregation(sheet_name)

        # 製品別データ
        train_product, test_product = prep.prepare_time_series_data(sheet_name)

        # 上位製品
        top_products = prep.get_top_products(sheet_name, top_n=30)

    print(f"\n{'='*80}")
    print("データ準備完了！")
    print(f"{'='*80}")
