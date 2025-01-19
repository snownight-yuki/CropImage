
# 画像クロッパーアプリケーション

このアプリケーションは、Python を使用した画像クロップツールです。
選択したフォルダ内の画像を **連続して処理** できるため、大量の画像を効率的に編集できます。
動作はシンプルで、中央に表示される赤枠（512x512）の中に画像を収め、選択した範囲をクロップして保存します。
※保存された画像は 1024x1024 にリサイズされます。

---

## 特徴

1. 画像をドラッグ・ズームして赤枠に収める操作が可能です。
2. 赤枠は固定サイズ（512x512）で、範囲を正確に選択できます。
3. 選択範囲を切り抜き、1024x1024 の画像として保存します。
4. **複数画像を一括処理**: フォルダ内の画像を自動で次々に読み込み、効率よく処理できます。

---

## 動作環境

- **Python バージョン**: 3.7 以上
- **必要なライブラリ**:
  - Pillow (Python Imaging Library)
  - tkinter（通常 Python に同梱されています）

---

## インストール手順

### 1. Python のインストール

1. [Python の公式サイト](https://www.python.org/) から Python をダウンロードし、インストールしてください。
2. インストール時に「Add Python to PATH」にチェックを入れてください。

### 2. 必要なライブラリのインストール

以下のコマンドをターミナルまたはコマンドプロンプトで実行して、必要なライブラリをインストールします：

```bash
pip install pillow
```

### 3. スクリプトの準備

1. このプロジェクトのスクリプト（`image_cropper.py`）をダウンロードまたはコピーしてください。
2. スクリプトを保存したディレクトリに移動します。

---

## 使用方法

### 1. アプリケーションの起動

以下のコマンドをターミナルまたはコマンドプロンプトで実行します：

```bash
python image_cropper.py
```

---

### 2. 画像の処理手順

#### (1) フォルダの選択

1. アプリケーション起動後、「Open Folder」ボタンをクリックします。
2. クロップ処理を行いたい画像ファイル（`.png`, `.jpg`, `.jpeg`）が含まれるフォルダを選択します。
3. 選択されたフォルダ内の最初の画像が表示されます。

#### (2) 画像の調整

- **ドラッグ操作**: マウスで画像をドラッグして、赤枠内に収めます。
- **ズーム操作**: マウスホイールで画像を拡大・縮小します。

#### (3) クロップと保存

1. 「Crop & Save」ボタンをクリックすると、赤枠内の画像が 1024x1024 サイズで保存されます。
2. 保存後、自動的に次の画像がロードされます。
3. この手順を繰り返して、フォルダ内のすべての画像を連続して処理できます。

---

## 保存先

- 処理された画像は、スクリプトが保存されているディレクトリ内の `output` フォルダに保存されます。
- 保存される画像の形式は PNG です。

---

## 注意事項

- 入力可能な画像形式は `.png`, `.jpg`, `.jpeg` のみです。
- 赤枠内の選択範囲のみ保存されます。
- フォルダ内の画像を順番に処理します。
