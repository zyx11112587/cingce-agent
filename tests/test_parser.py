from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from exam_rag.parser import parse_front_matter, parse_source_file


SAMPLE = """---
试卷: "示例试卷"
地区: "国家"
年份: "2024"
模块: "判断推理"
---

# 示例试卷

## 第 1 题　<sub>qid 10001 · 图形推理</sub>

下面哪一个图形具有唯一对称轴？

- **A**. 圆形
- **B**. 正方形
- **C**. 正三角形
- **D**. 不等边三角形

**答案**：C

**官方解析**

正三角形具有三条对称轴，但选项中只有该项为等边三角形。

---
"""


class ParserTest(unittest.TestCase):
    def test_front_matter(self) -> None:
        frontmatter = parse_front_matter(SAMPLE)
        self.assertEqual(frontmatter["年份"], "2024")
        self.assertEqual(frontmatter["模块"], "判断推理")

    def test_parse_question(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "sample.md"
            path.write_text(SAMPLE, encoding="utf-8")
            questions, chunks = parse_source_file(path, root)
        self.assertEqual(len(questions), 1)
        question = questions[0]
        self.assertEqual(question.qid, "10001")
        self.assertEqual(question.answer, "C")
        self.assertEqual(question.region, "国家")
        self.assertIn("正三角形", question.options["C"])
        self.assertTrue(chunks == [] or all(chunk.chunk_type == "knowledge" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
