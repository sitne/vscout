# V-SCOUT

**Valorant VOD分析 & スカウティングツール**

プロのValorant試合VODからイベント・プレイヤー状態を自動抽出し、Webダッシュボードで可視化するツール。

## アーキテクチャ

```
vscout/
├── packages/
│   ├── valoscribe/    # VOD解析エンジン（コンピュータビジョン + OCR）
│   │                  # → event_log.jsonl, frame_states.csv を出力
│   └── vscout/        # Webアプリ（FastAPI + React）
│                      # → valoscribeの出力を読み込み・可視化
└── scripts/           # バッチ処理スクリプト
```

### パッケージ構成

- **valoscribe**: VODからのデータ抽出パイプライン
  - テンプレートマッチングによるHUD解析（スコア、タイマー、エージェント、HP等）
  - キルフィード解析、アビリティ/ウルト使用検出
  - VLR.gg連携によるメタデータ取得
  - 出力: JSONL イベントログ + CSV フレームステート

- **vscout**: Webフロントエンド & API
  - FastAPI バックエンド（valoscribeの出力を読み込み）
  - React ダッシュボード（イベントタイムライン、ラウンド詳細等）
  - セッション管理、レポート生成

## セットアップ

### 前提条件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- [deno](https://deno.land/) (yt-dlpのYouTube JS challenge解決用)
- Tesseract OCR
- ffmpeg

### インストール

```bash
# 依存インストール（Python + エントリポイント登録）
uv sync

# フロントエンドビルド
cd packages/vscout/frontend && npm install && npm run build && cd ../../..

# Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr ffmpeg

# deno (yt-dlpのYouTube JS challenge解決用)
curl -fsSL https://deno.land/install.sh | sh
```

> **Note**: `uv sync` を実行しないと `uv run vscout` で "No such file or directory" エラーが出ます。

## 使い方

### Webダッシュボード起動

```bash
uv run vscout
# http://localhost:8000 でアクセス
```

UIからVLR.gg URLを入力すると、自動でスクレイピング→VODダウンロード→解析まで実行されます。

### CLIでのVOD解析（valoscribe単体）

```bash
# VLR.ggスクレイピング
uv run valoscribe scrape-vlr "https://www.vlr.gg/MATCH_ID/..." -o match.json

# メタデータ分割
uv run valoscribe split-metadata match.json -o ./maps/

# VODダウンロード
uv run valoscribe download "https://youtube.com/watch?v=..." -o ./videos/

# VOD解析
uv run valoscribe orchestrate process-vod video.mp4 maps/map1.json -o ./output/
```

### API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| POST | `/api/analyze` | VLR分析パイプライン開始 (`{vlr_url}`) |
| GET | `/api/status` | パイプライン進捗・ステップログ |
| POST | `/api/stop` | 実行中のパイプラインをキャンセル |
| GET | `/api/sessions` | セッション一覧 |
| GET | `/api/matches/{session}/{map}` | 試合概要 |
| GET | `/api/matches/{session}/{map}/events` | イベント一覧 |
| GET | `/api/matches/{session}/{map}/rounds/{n}` | ラウンド詳細 |
| GET | `/api/matches/{session}/{map}/kills` | キルタイムライン |

## 開発

```bash
# テスト
uv run pytest

# 型チェック
uv run mypy packages/valoscribe/src packages/vscout/src

# Lint
uv run ruff check packages/
```

## ライセンス

MIT
