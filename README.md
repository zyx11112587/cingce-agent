# 考公 RAG 真题与申论资料库

本地 RAG 项目，支持行测真题、政治理论 PDF、申论材料、Markdown 和 DOCX，采用：

- SQLite FTS5 trigram + BM25 关键词召回
- FAISS 向量库（未安装 FAISS 时自动使用 NumPy cosine 索引）
- BM25 与向量检索的 Reciprocal Rank Fusion 融合
- OpenAI 兼容的 LLM 和 embeddings API
- 未配置生成模型时的本地证据摘录回退

## 目录约定

```text
xingcezhenti/          行测 Markdown 真题
data/raw/              其他 Markdown、PDF、DOCX、TXT
data/politics/         政治理论 PDF 或 DOCX
data/shenlun/          申论真题、材料、范文
data/index/            SQLite 和向量索引
exam_rag/              项目代码
```

材料目录不存在时会自动跳过。可以通过环境变量继续增加目录：

```text
EXAM_RAG_INPUT_DIRS=./xingcezhenti;./data/raw;./data/politics;./data/shenlun
```

## 安装依赖

项目代码已写好，但没有自动创建虚拟环境或安装依赖。需要手动执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果使用 API 生成 embeddings，仍然建议安装 FAISS；没有 FAISS 时检索会自动降级为 NumPy 向量索引。

## 建立索引

默认读取 `.env` 中配置的全部输入目录：

```powershell
python -m exam_rag.cli ingest
```

也可以显式指定多个目录：

```powershell
python -m exam_rag.cli ingest `
  --input .\xingcezhenti `
  --input .\data\politics `
  --input .\data\shenlun
```

只重建关键词索引、不调用 embedding API：

```powershell
python -m exam_rag.cli ingest --skip-vectors
```

## 接入 Embeddings 和 LLM

在 `.env` 中配置：

```text
EXAM_RAG_EMBEDDING_PROVIDER=api
EXAM_RAG_EMBEDDING_BASE_URL=http://127.0.0.1:11434/v1
EXAM_RAG_EMBEDDING_MODEL=bge-m3
EXAM_RAG_EMBEDDING_API_KEY=

EXAM_RAG_LLM_BASE_URL=http://127.0.0.1:11434/v1
EXAM_RAG_LLM_MODEL=qwen2.5:7b
EXAM_RAG_LLM_API_KEY=
```

`auto` 模式会在 API 配置存在时使用 API；否则尝试 `sentence-transformers`。没有 embedding 模型时会保留 BM25 检索，不会中断问答服务。

## 查询

```powershell
python -m exam_rag.cli search "新时代党的建设总要求"
python -m exam_rag.cli ask "申论综合分析题如何提炼材料观点？" --doc-type 申论
python -m exam_rag.cli ask "如何理解中国式现代化？" --subject 政治理论
python -m exam_rag.cli explain "下面哪一个图形具有唯一对称轴"
```

检索结果会返回 `retrieval` 字段，分别标记 BM25 排名和向量排名。未配置 embedding 时，`retrieval_mode` 为 `bm25`；双路索引可用时为 `hybrid`。

## 启动 Web 和 API

```powershell
python -m exam_rag.cli serve --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

API：

- `GET /api/health`：索引、LLM、向量后端状态
- `GET /api/stats`：题目、文档块、图片、向量数量
- `GET /api/search?q=...&mode=search&top_k=5`
- `POST /api/qa`：支持 `mode`、`top_k`、`filters`
- `GET /api/question/{qid}`
- `GET /api/image/{image_id}`

`POST /api/qa` 示例：

```json
{
  "question": "申论综合分析题如何提炼材料观点？",
  "mode": "knowledge",
  "top_k": 5,
  "filters": {
    "doc_type": "申论"
  }
}
```

## PDF 说明

- 文本型 PDF 使用 `pypdf` 提取；分页后会保留页码。
- 扫描版 PDF 没有可提取文字时会明确报错，需要先做 OCR 或接入 PaddleOCR/RapidOCR。
- DOCX 使用 `python-docx` 提取段落和表格文本。
- 当前工作区没有检测到 PDF、DOCX 或申论材料；把文件放入 `data/politics`、`data/shenlun` 或 `data/raw` 后重新执行 `ingest`。

## 运行脚本

```powershell
.\run.ps1 ingest
.\run.ps1 serve
```

如果 PowerShell 禁止执行脚本，可以使用：

```powershell
.\run.bat ingest
.\run.bat serve
```

或：

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 ingest
```
