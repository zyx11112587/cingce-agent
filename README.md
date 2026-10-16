# 考公 RAG 真题库

这是一个不依赖第三方包的本地 RAG 项目，直接读取 `xingcezhenti` 目录中的 Markdown 和图片，支持：

- 真题检索
- 考点问答
- 解析查询
- OpenAI 兼容 API 接入
- 无 API 时的本地证据摘录回退

核心检索使用 Python 标准库 `sqlite3` 的 FTS5 trigram 索引。题目、答案、官方解析和图片引用会被结构化保存，不依赖云端服务。
真题来源：https://github.com/ERRRC/xingcezhenti

## 目录

```text
exam_rag/
  cli.py         命令行入口
  parser.py      Markdown、题目、答案、图片解析
  index.py       SQLite 结构化索引和 FTS5 检索
  rag.py         证据组装和问答回退
  llm.py         OpenAI 兼容 API 客户端
  server.py      本地 HTTP API 和页面服务
  web/index.html 本地 Web UI
prompts/
  qa_prompt.txt  问答提示词
run.ps1           PowerShell 运行脚本
run.bat           Windows 批处理运行脚本
```

## 建立索引

不创建虚拟环境、不安装依赖时，直接运行：

```powershell
.\run.ps1 ingest
```

如果本机 PowerShell 禁止执行脚本，可以改用 `.\run.bat ingest`，或使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 ingest
```

也可以指定目录和数据库：

```powershell
.\run.ps1 ingest --input .\xingcezhenti --db .\data\index\exam.sqlite3
```

首次入库会解析全部 Markdown 文件，并重建 SQLite 索引。

## 使用命令行

```powershell
.\run.ps1 search "对称轴 图形推理"
.\run.ps1 ask "图形推理中如何判断对称轴数量？"
.\run.ps1 explain "下面哪一个图形具有唯一对称轴"
.\run.ps1 stats
```

检索词可以直接是题干、关键词或 `qid`。解析查询会优先返回原题、选项、答案和官方解析。

## 启动本地服务

```powershell
.\run.ps1 serve --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

API：

- `GET /api/health`
- `GET /api/stats`
- `GET /api/search?q=...&mode=search&top_k=5`
- `POST /api/qa`
- `GET /api/question/{qid}`
- `GET /api/image/{image_id}`

## 接入 OpenAI 兼容 API

在 `.env` 中配置，或直接使用环境变量：

```text
EXAM_RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1
EXAM_RAG_LLM_API_KEY=
EXAM_RAG_LLM_MODEL=qwen2.5:7b
```

也兼容以下常见变量：

```text
OPENAI_BASE_URL
OPENAI_API_KEY
OPENAI_MODEL
```

配置后，问答请求会调用 `chat/completions`；未配置或调用失败时，服务自动回退为本地原文证据模式，不会中断检索和解析查询。

## 说明

- 图片不复制到数据库中，只保存图片路径和 image_id，通过 `/api/image/{image_id}` 按需提供。
- Markdown 中无法识别的非题目内容会作为知识点块入库。
- 索引重建是幂等操作，重新执行 `ingest` 会清空旧索引并重新生成。
