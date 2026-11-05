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
