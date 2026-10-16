@echo off
setlocal
cd /d "%~dp0"
python -m exam_rag.cli %*
