# 数字人文叙事 AI 分析引擎

> 基于大语言模型的叙事文本分析：人物关系网络抽取与可视化、人物画像与叙事生成。

## 功能

- **人物关系网络**：从文本中抽取人物关系，生成交互式关系图（vis-network）
- **人物画像**：LLM 驱动的角色分析
- **叙事生成**：AI 叙事报告与命运推演

## 入口

| 入口 | 说明 |
|------|------|
| `ai_cli.py` | 终端 CLI（typer + rich，美观排版） |
| `app.py` | Streamlit Web 界面 |
| `examples/` | 示例分析（如名著人物关系网络） |

## 技术栈

Python（Streamlit / NetworkX / Typer）+ 前端 React（Vite + TypeScript）。

## 示例

```bash
python ai_cli.py --help
streamlit run app.py
```

## 环境变量（审计后新增）

| 变量 | 默认 | 作用 |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173` | CORS 白名单，逗号分隔。原来是 `*`（配合 `allow_credentials=True` 实际会被浏览器拒绝） |
| `MAX_UPLOAD_MB` | `5` | 上传 txt 上限，超了返回 413，不会整份读进内存 |
| `TASKS_DB_PATH` | `tasks.db` | 任务状态 SQLite 路径。原来是进程内 dict：多 worker 不共享、重启即丢 |

> 说明：SQLite 只解决"重启不丢"。多 worker / 多实例部署仍应换 Redis/Postgres 或独立任务队列。

## 质量与回归（2026-11-11 迭代）

### 测试

```bash
pip install -r requirements.txt && pip install pytest
python -m pytest tests -q      # 38 passed
```

- `tests/conftest.py` 提供 NER / 别名 / 关系 / 事件 / 画像 / 链路的夹具。
- `tests/test_alias_rules.py` 覆盖别名合并与非人物过滤。
- `tests/test_sentence_relations.py` / `tests/test_directed_relations.py` 覆盖句级共现与有向关系。

### 依赖是可选的，不是硬门槛

| 依赖 | 缺失时行为 |
|---|---|
| spaCy 及 `zh_core_web_sm` | 打 `RuntimeWarning` 并降级为**词典模式**，流水线照常跑 |
| chromadb（RAG / 多智能体沙盘） | API 正常启动，`/api/v1/simulation/run` 返回 503 并说明缺哪个包 |
| `python-multipart` | 必需，缺失时 FastAPI 导入即报错（已写进 requirements） |

### 本版修的问题（都有实测依据）

| 问题 | 表现 | 处理 |
|---|---|---|
| 测试收集失败 | 7 个 fixture 未定义，pytest 全红 | 新增 `conftest.py` |
| 规则分支缺证据句 | 关系的 `context_snippet` 恒为空 | 规则分支也写窗口原文；去重改为每对保留 2 条 |
| `presence_ratio` 单位错 | 长文只有 1 段，任何角色都显示 1.0 | 改为"出现过的人数/总段数" |
| 角色判档全成主角 | 同上，阈值永远命中 | 单段文本改用"出现次数占全体比重"分档 |
| 别名不合并 | 诺诺/陈墨瞳、昂热/校长 各算一人 | `explicit_mapping` 支持 `角色 → [别名]`，冲突会报错而不是静默覆盖 |
| 地名进人物网络 | `卡塞尔`（学院名）被当作人物节点 | `src/base/entity_filter.py` 按作品停用 |
| 关系是"共现网络" | 100 字窗口步长 20，人人两两相连，85% 是 `association` | 改为**句子级共现 + 谓词类型**（`src/network/sentence_relations.py`），每条边带原句证据 |
| 别名替换残留 | 占位符方案在 Windows 上会带入换行，正文出现 `___` | 改为"按长度降序单遍替换"，无需占位符 |

### 《龙族1·火之晨曦》前 6 章实测（30,191 字，13 人）

| 指标 | 改前（窗口共现） | 改后（句级 + 谓词） |
|---|---|---|
| 关系 | 115 → 71 | **21** |
| `association` 占比 | **88%**（101/115） | **19%**（4/21） |
| 关系类型 | 2 种 | **6 种**（social 7 / dialogic 5 / hostile 3 / association 4 / kinship 1 / affection 1） |
| 图谱密度 | 0.40–0.56 | 0.38（边数降 70%，密度只反映可连对数不完全可比） |
| 证据覆盖 | 100% | 100% |
| 画像种类 | 1 | 3 |
| 角色分布 | 全部 protagonist | 1 主角 / 2 minor / 10 background |

典型边（含证据句）：

```
路明非 —[social]— 陈墨瞳   权重 26  «QQ上一个大脸猫头像跳闪起来，名字是"陈墨瞳"…»
路明非 —[dialogic]— 陈雯雯 权重 23  «…问陈雯雯明天晚上要不要参加文学社的活动»
路明非 —[kinship]— 路鸣泽 权重 20  «托那笔钱的福，路明非可以上私立贵族高中…»
路明非 —[hostile]— 诺顿    权重 8  «如果诺顿亲眼看见路明非的操作…只会骂一句"变态"»
```

## 有向关系（P2，2026-11-11）

边现在是**有向的**：`主体 —[类型]→ 客体`，方向来自句内语序，类型来自谓词词典
（`src/network/directed_relations.py`）。

### 为什么没用 LTP / HanLP

实测记录（本机）：

| 检查 | 结果 |
|---|---|
| `curl https://huggingface.co/LTP/small/resolve/main/config.json` | **000**（不可达） |
| `curl https://hf-mirror.com/...` | 307（镜像可用，模型 172 MB） |
| 装 `ltp` | 成功，但同时升级 **transformers 5.22.2** |
| LTP + transformers 5 | `BertTokenizer has no attribute batch_encode_plus` → 不兼容 |
| 装回旧版 transformers | 会牵连用户已装的 **torch 2.14.1+cpu** |

→ 结论：在这台机器上句法模型这条路当前走不通，已回滚 ltp/transformers/safetensors 的安装。

### 现在的做法（零模型、可离线）

1. `jieba.posseg` 分词 + 词性
2. 同句两两相邻人物配对，**语序在前的是主体**
3. 动词在两位人物之间或紧随其后（中文"对X说"动词后置）都能取到
4. 动词 → 关系类型；取不到动词再退回句级谓词

### 验收

| 指标 | 结果 |
|---|---|
| 人工标定 8 句方向 | **8/8 = 100%** |
| 人工标定 8 句类型 | **8/8 = 100%** |
| 龙族前 6 章（13 人） | 有向边 **53** 条，带具体类型 **27** 条（51%），其余 26 条是 `association` |
| pytest | **46 passed** |

> 这条验收线偏乐观：标定句是照着"主谓宾"写法的典型句。真实长文里 `association`
> 仍占一半，因为大量句子是"两个人名同现 + 没有谓词"（如"路明非和陈雯雯的对话记录"）。

---

## LLM 复核候选边（可选）

`src/network/llm_verifier.py`：规则版先给候选边，模型只回答「这条关系在证据句里成立吗、
类型对不对」，比从零抽取便宜。

```python
from src.network.relations import LLMRelationshipExtractor
from src.network.llm_verifier import review_candidates, summarize_review

verifier = LLMRelationshipExtractor()          # 无 key 时 available=False
kept = review_candidates(verifier, candidates) # 无 key 时原样返回，不发任何请求
print(summarize_review(candidates, kept))
```

### 复核强度（strict / normal / loose）

模型对不同"弱关系"的容忍度差别很大，所以把判定标准做成参数：

| 强度 | 规则 | 适合 |
|---|---|---|
| `strict`（默认） | 只保留模型判成立的边 | 互动/剧情分析 |
| `normal` | `association` 一律保留，其余以模型为准 | 通用 |
| `loose` | 只剔除明显误判（数值当人物、假设语气、转述、比喻） | 社交网络分析 |

### 一次真实判定（53 条候选，模型逐条读证据句）

| 强度 | 保留 | 剔除 | `association` 占比 |
|---|---:|---:|---:|
| 不复核 | 53 | 0 | 49% |
| `strict` | **20** | 33 | 50% |
| `normal` | **36** | 17 | 72% |
| `loose` | **45** | 8 | 51% |

> 这次判定由 DSH 里的一个空白会话完成（本地脚本抽候选 → 会话逐条判 → 回写 JSON），
> 不是本文档作者手判。模型比人工严格得多：人工判不成立的 11 条它全判不成立，
> 另外还判掉了 22 条人工保留的弱关系（单方评价外貌、同校而已、对比提及、军事比喻等）。

### 在《龙族》前 6 章上的复核结果（人工充当复核方，逐条读证据句判定）

| 指标 | 结果 |
|---|---|
| 候选边 | 53 |
| 判定后保留 | **42**（剔除 11 条 = 21%） |
| 类型被纠正 | **16** 条 |
| 模型判不成立的主因 | "两人只是同句出现，没有互动"（9/11） |

被剔除的典型例子：

| 边 | 原文依据 | 剔除理由 |
|---|---|---|
| 零 → 路明非 | "出枪速度**为零**" | "零"是数值，不是人物 |
| 陈雯雯 → 路明非（affection） | "**如果**陈雯雯…喜欢路明非三年了" | 假设语气，未发生 |
| 路明非 → 陈雯雯（hostile） | "路明非上去**打岔**" | "打岔"是想听她说话，不是冲突 |
| 路鸣泽 → 路明非（kinship） | "…让路鸣泽一时没忍住笑了起来" | 同场笑，无亲属表述 |

> 注意：这次复核由人工完成（逐条读证据句），不是模型跑出来的。
> 换到真实模型上，数字会变——收录这个结果是为了证明**复核这一步有实质收益**，
> 以及给 `llm_verifier.py` 的接口定下输入输出。
> 类型细化仍需真的句法或 LLM 抽取器；`directed=False` 可退回句级版本做对比。

---

## 长文本实测：《遮天》（辰东，698 万字）

比《龙族》样本大 40 倍的语料，用来验证"没有现成人名词典"时能不能跑。

### 自动抽人物（`src/base/entity_extract.py`）

没有 `LONGZU_MAPPING` 这类人工表时，先用 jieba 的 `nr` 标签抽候选，再用三个信号过滤：

| 信号 | 作用 |
|---|---|
| 说话归属 | 「候选 + 说道/问道/笑道」的次数——小说里人物一定说话 |
| 动作主体 | 「候选 + 看着/转身/点头」的次数 |
| 多标签 | 同一个串在别处被标成 v/n（如「冷笑」「巴掌」）→ 判死 |
| 后缀 | 家/门/宗/族/派/经/诀/术 → 判死 |
| 碎片合并 | 「姬紫」是「姬紫月」的子串 → 并入长名 |

1,200,000 字样本：`nr` 候选 2004 个 → 过滤后 **54 人** → 人工过一眼再清 19 个 → **35 人**。

### 结果

| 口径 | 人物 | 关系边 | 垃圾边占比 | 密度 | 主角 | 耗时 |
|---|---:|---:|---:|---:|---|---:|
| 自动（未清洗） | 54 | 626 | **33%** | 0.162 | 叶凡 ✓ | 21.1s |
| 自动 + 人工清洗 | 35 | 366 | **0%** | 0.177 | 叶凡 ✓ | 19.3s |

清洗后类型分布：`association` 152 / `dialogic` 90 / `affection` 48 / `hostile` 37 /
`social` 26 / `collaborative` 10 / `hierarchical` 3；证据覆盖 **100%**。

垃圾节点长这样（进图后会污染所有中心性指标）：

```
叶凡 →[association]→ 巴掌    «…裂缝末端已不足巴掌宽»
叶凡 →[association]→ 明白    «唯有叶凡明白，金刚宝杵最后的一丝神力…»
叶凡 →[association]→ 向前走  «这时叶凡忽然向前走去»
叶凡 →[association]→ 冷笑    «叶凡站在远处，脸上带着淡淡地冷笑»
```

### 这次暴露的两个新问题（未修）

1. **时间线模块在长文上基本失效**：120 万字只抽出 **1 个事件**。
   `TimelineExtractor` 依赖句内的显式时间词，网文里"三日后""次日"这类写法覆盖不到。
2. **别名仍靠人工表**：`黑狗` 与 `黑皇` 是同一只（原文"大黑狗…张五爷领回来的"），
   自动模式下会算成两个人。`LONGZU_MAPPING` 那种表能解决，但每本书都要写一遍。
