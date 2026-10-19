# 资料目录

- `data/raw/`: 其他 Markdown、PDF、DOCX、TXT 资料
- `data/politics/`: 政治理论 PDF/DOCX
- `data/shenlun/`: 申论真题、材料、范文
- `data/index/`: 运行时生成的 SQLite 和向量索引

把文件放入对应目录后执行：

```powershell
python -m exam_rag.cli ingest
```
