# 購買予測システム

Excelファイルのデータを使用して購買予測を行うプロジェクトです。

## プロジェクト構成

```
.
├── data/
│   ├── raw/          # 生データ（Excelファイルなど）
│   └── processed/    # 前処理済みデータ
├── models/           # 学習済みモデル
├── notebooks/        # Jupyter Notebook
├── src/              # ソースコード
│   ├── data_loader.py      # データ読み込み
│   ├── preprocessor.py     # データ前処理
│   ├── train.py            # モデル学習
│   └── predict.py          # 予測実行
├── requirements.txt  # 依存ライブラリ
└── README.md         # このファイル
```

## セットアップ手順

### 1. Python仮想環境の作成

```bash
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

### 2. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. データの配置

Excelファイルを `data/raw/` ディレクトリに配置してください。

## 使用方法

### データの読み込みと前処理

```python
from src.data_loader import load_excel_data
from src.preprocessor import preprocess_data

# データ読み込み
df = load_excel_data('data/raw/your_data.xlsx')

# 前処理
X_train, X_test, y_train, y_test = preprocess_data(df)
```

### モデルの学習

```python
from src.train import train_model

# モデル学習
model = train_model(X_train, y_train)
```

### 予測の実行

```python
from src.predict import predict

# 予測
predictions = predict(model, X_test)
```

## 使用ライブラリ

- pandas: データ処理
- numpy: 数値計算
- scikit-learn: 機械学習
- xgboost, lightgbm: 勾配ブースティング
- matplotlib, seaborn: データ可視化
- openpyxl: Excel読み込み

## ライセンス

MIT License
