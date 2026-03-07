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
# 依存インストール
uv sync

# Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr ffmpeg

# deno
curl -fsSL https://deno.land/install.sh | sh
```

## 使い方

### 1. VOD解析（valoscribe）

```bash
# VLR.ggのURLから全マップを処理
uv run bash scripts/process_vlr_series.sh "https://www.vlr.gg/MATCH_ID/..."

# 単体動画を処理
uv run valoscribe orchestrate process \
  --video-path ./video.mp4 \
  --output-dir ./data/session_name/map1/output
```

### 2. Webダッシュボード（vscout）

```bash
# APIサーバー起動
uv run vscout

# http://localhost:8000 でアクセス
```

### API エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| GET | `/api/sessions` | セッション一覧 |
| GET | `/api/matches/{session}/{map}/` | 試合概要 |
| GET | `/api/matches/{session}/{map}/events` | イベント一覧 |
| GET | `/api/matches/{session}/{map}/rounds/{n}` | ラウンド詳細 |
| GET | `/api/matches/{session}/{map}/kills` | キルタイムライン |
| POST | `/api/analyze` | VOD解析開始 |

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
