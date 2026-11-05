# 飞链 RAG · FeiLian

> 把散落在文件夹里的文档，链成一条能回查的答案链。

飞链是一个本地优先的检索增强生成（RAG）工具：文档不出机器，答案必须带出处。
它从零构建索引、支持增量同步、按页按块给出引用，答不出来就直说答不出来。

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LangChain](https://img.shields.io/badge/langchain-1.x-1c3c3c)

## 它解决什么问题

大模型很会说，但你不敢信。凭印象编出来的答案最危险的地方，不是错，而是错得像对的。

飞链把「依据」变成一等公民：

- 每句结论后面都有 `[1]` `[2]`，对应到具体文件、页码、文本块
- 资料里没有的，明确回「没有找到」，并告诉你它查到了哪些相近内容
- 所有回答都来自你自己的文档，本地模型跑，不上传

## 特性

| 能力 | 说明 |
| --- | --- |
| 增量索引 | 按内容指纹（sha1）判断文件是否变化，第二次同步只跳过、不重算 |
| 持久化向量库 | Chroma 落盘到 `.feilian/`，进程退出不丢，启动秒开 |
| 多格式加载 | `.txt` `.md` `.pdf` `.docx`，PDF 按页切分并保留页码 |
| 带引用作答 | 答案附「资料来源」清单，含文件名、页码、块号、相关度 |
| 拒答不编造 | 检索为空或资料不覆盖时直接说明，不给幻觉留口子 |
| 多轮记忆 | 保留最近 N 轮上下文，追问「那第二步呢」接得上 |
| 双模型后端 | Ollama（默认，本地）与 OpenAI（或任何兼容网关）一行切换 |
| 中文友好 | 切块按中文标点优先，提示词与输出全中文 |
| 离线可测 | 内置假模型，`pytest` 13 个用例不联网全绿 |

## 快速开始

```bash
git clone https://github.com/FeinaoLab/feilian-rag.git
cd feilian-rag

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

用本地模型（推荐，零成本、数据不出机器）：

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

体检 → 建索引 → 提问：

```bash
python -m feilian doctor          # 检查配置、模型、索引状态
python -m feilian index           # 把 docs/ 同步进向量库
python -m feilian ask "切块大小建议设多少？"
python -m feilian chat            # 交互式多轮问答
```

## 命令一览

```bash
feilian index [--rebuild]     # 增量同步；--rebuild 忽略 manifest 整库重建
feilian ask "问题" [--json]   # 单次问答，--json 输出结构化引用
feilian chat                  # 交互模式：/index /stats /reset /quit
feilian stats                 # 文件数、块数、孤儿块、索引目录
feilian doctor                # 环境体检：配置、模型连通、索引概况
```

交互模式里输入 `/help` 看全部指令。

## 输出长什么样

```
> 切块大小建议设多少？

切块建议 500–800 字符，重叠 100–150 字符 [1]。切块太大检索不准，太小上下文不足 [2]。

资料来源：
[1] 02-RAG-工作原理.md · 块1（相关度 0.612）
    常见的默认值是 1000 字符、重叠 200 字符。但如果文档本身只有一千字……
[2] 03-索引与检索调优.md · 块0（相关度 0.507）
    切块大小决定检索精度……
```

`--json` 模式下每条来源都带 `label` / `score` / `snippet`，方便接自己的前端。

## 项目结构

```
feilian-rag/
├── feilian/
│   ├── config.py      # 配置：环境变量 → 不可变 Settings
│   ├── loaders.py     # 加载：递归遍历 + 四类格式解析
│   ├── index.py       # 索引：落盘 Chroma + manifest 增量同步 + 检索
│   ├── chain.py       # 问答：提示词、引用、多轮记忆
│   ├── providers.py   # 模型工厂 + 健康检查（Ollama / OpenAI）
│   ├── cli.py         # 命令行入口
│   ├── testing.py     # 离线假模型，供测试与演示
│   └── errors.py      # 统一异常，错误信息说人话
├── docs/              # 示例知识库（3 篇讲 RAG 的文档）
├── tests/             # 13 个离线用例
└── .env.example
```

## 工作原理

```
docs/*.md|pdf|docx
      │  loaders：递归加载，保留 source / page 元数据
      ▼
  splitter：按中文标点优先切 700 字，重叠 120
      │
      ▼
  embeddings：nomic-embed-text / text-embedding-3-small
      │
      ▼
   Chroma（.feilian/chroma，落盘）
      │  manifest.json 记录每个文件的指纹与块 id
      ▼
  retrieve：top_k=4，可加相关度阈值或切 MMR
      │
      ▼
  prompt：只依据资料 + 必须标引用编号 + 找不到就说找不到
      ▼
  answer + 资料来源清单
```

### 增量同步怎么判断

同步时比对三件事：文件是否新增、指纹是否变化、文件是否已删除。
指纹不变直接跳过；参数（切块大小或嵌入模型）变了，签名对不上，自动整库重建，避免新旧向量维度冲突。

## 配置项

全部写在 `.env`，也可以用命令行临时覆盖（`--provider` `--docs` `--k`）。

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama` 或 `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | 本地服务地址 |
| `OLLAMA_MODEL` | `llama3.2` | 对话模型 |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | 空 / `gpt-4o-mini` | 用 OpenAI 时必填 |
| `OPENAI_BASE_URL` | 空 | 接第三方兼容网关时填 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | 换模型会触发整库重建 |
| `DOCS_DIR` / `STORE_DIR` | `docs` / `.feilian` | 知识目录与索引目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `120` | 切块大小与重叠 |
| `TOP_K` / `FETCH_K` | `4` / `20` | 召回条数 / MMR 候选数 |
| `SEARCH_TYPE` | `similarity` | `similarity` 或 `mmr` |
| `SCORE_THRESHOLD` | `0.2` | 低于此相关度的块不进上下文；留空关闭 |
| `HISTORY_TURNS` | `6` | 记忆保留轮数 |

## 测试

```bash
pytest
# 13 passed in ~3s，不联网、不需要 Ollama
```

测试用 `feilian.testing` 里的确定性假向量与假模型，覆盖增量同步、删除同步、参数变更重建、引用生成、拒答、记忆裁剪等路径。

## 和原始 demo 的差别

这个项目脱胎于一个教学用的 LangChain RAG demo，重写时修掉了几个会让人踩坑的点：

- `langchain.text_splitter` 在 LangChain 1.x 已移除，改用独立包 `langchain-text-splitters`
- 索引不再每次启动全量重建，改成落盘 + 指纹增量
- 切块从「1000 字 + 200 重叠」（小文件等于不切）改为 700/120 并按中文标点切
- 只支持 `.txt` 扩展为四种格式，PDF 保留页码用于引用
- 加了相关度阈值、引用清单与拒答约束，降低幻觉
- 补了 13 个离线测试与统一异常处理

## 已知限制

- 纯向量检索，没做关键词混合召回与查询改写；专有名词多、用词差异大的场景召回会掉
- 单机工具，适合万级块；百万级请上专门的向量数据库
- 引用编号依赖模型遵守提示词，小模型偶尔会漏标，`temperature=0` 可显著缓解
- 不做 OCR，扫描版 PDF 提取不到文字

## 路线图

- [ ] 混合检索（BM25 + 向量）与查询改写
- [ ] 父子块检索：小块召回、大块喂给模型
- [ ] 索引健康检查：提示「哪些文档从没被召回」
- [ ] Web UI 与 HTTP API
- [ ] 支持 CSV / XLSX 结构化问答

## 许可证

MIT © 2026 FeinaoLab
