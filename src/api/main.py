import json
import os
import sqlite3
import uuid
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.pipeline import NarrativePipeline
from src.base.character_dict import get_character_dict, get_character_mapping
from src.network import LLMRelationshipExtractor, RelationshipExtractor
from src.character import LLMCharacterProfiler, CharacterProfiler
from src.narrative import LLMNarrativeGenerator, NarrativeGenerator

load_dotenv()

app = FastAPI(
    title="数字人文叙事 AI API (To-C Edition)",
    description="面向文学爱好者的 AI 驱动的小说世界观解析与可视化平台 API",
    version="1.0.0"
)


def _cors_origins() -> list[str]:
    """CORS 白名单。

    审计项：原来是 allow_origins=["*"] + allow_credentials=True —— 浏览器规范下
    这两个组合会被直接拒绝，而且等于对任意站点开放读接口。
    现在默认只放行本地前端；需要别的来源请设 CORS_ORIGINS（逗号分隔）。
    """
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:5173", "http://127.0.0.1:5173",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 上传大小上限（MB），避免把整个文件读进内存
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5"))


class TaskStore:
    """任务状态存储。

    审计项：原来是进程内 dict —— 多 worker 不共享、重启即丢。
    这里换成 SQLite（同进程单文件），至少做到**跨重启可恢复**；
    多 worker 部署仍建议换 Redis/Postgres，这一点写在 docstring 里而不是假装解决。
    """

    def __init__(self, path: str):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    def create(self, task: Dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO tasks (task_id, data) VALUES (?, ?)",
            (task["task_id"], json.dumps(task, ensure_ascii=False)),
        )
        self._conn.commit()

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT data FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, task_id: str, **fields: Any) -> None:
        task = self.get(task_id)
        if task is None:
            return
        task.update(fields)
        self.create(task)

    def close(self) -> None:
        self._conn.close()


TASKS_DB_PATH = os.getenv("TASKS_DB_PATH", "tasks.db")
tasks_db = TaskStore(TASKS_DB_PATH)

# 全局单例 RAG 知识库，避免重复冷启动
rag_db_instance = None

def get_rag_db():
    global rag_db_instance
    if rag_db_instance is None:
        from src.rag.knowledge_base import RAGKnowledgeBase
        rag_db_instance = RAGKnowledgeBase()
    return rag_db_instance

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    book_name: str
    created_at: float
    completed_at: Optional[float] = None
    results: Optional[Dict[str, Any]] = None

def run_analysis_task(task_id: str, text: str, book_name: str):
    """后台异步任务：运行叙事分析流水线"""
    try:
        tasks_db.update(task_id, status="processing", message="正在初始化 AI 引擎...")
        
        # 将文本拆分为段落
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 5]
        
        # 初始化 Pipeline
        llm_api_key = os.getenv("DEEPSEEK_API_KEY")
        if llm_api_key:
            relation_extractor = LLMRelationshipExtractor(api_key=llm_api_key)
            profiler = LLMCharacterProfiler(api_key=llm_api_key)
            narrative_generator = LLMNarrativeGenerator(api_key=llm_api_key)
        else:
            relation_extractor = RelationshipExtractor()
            profiler = CharacterProfiler()
            narrative_generator = NarrativeGenerator()
            
        pipeline = NarrativePipeline(
            relation_extractor=relation_extractor,
            profiler=profiler,
            narrative_generator=narrative_generator,
            character_names=get_character_dict(book_name),
            convert_traditional=True,
            strict_dict_mode=True,
            explicit_mapping=get_character_mapping(book_name)
        )
        
        tasks_db.update(
            task_id,
            message="正在深度阅读文本并进行实体与关系抽取 (这可能需要较长时间)...",
            progress=10.0,
        )
        
        # 运行分析 (这里如果需要真实进度条，需重构 pipeline.run 支持 callback，目前为了 MVP 简化处理)
        # TODO: 注入进度回调 callback 给 pipeline
        results = pipeline.run(paragraphs, verbose=False)
        
        tasks_db.update(
            task_id,
            status="completed",
            progress=100.0,
            message="分析完成",
            completed_at=time.time(),
            results=results.model_dump(),
        )
        
    except Exception as e:
        tasks_db.update(task_id, status="failed", message=f"分析失败: {str(e)}")
        print(f"Task {task_id} failed: {e}")

@app.post("/api/v1/tasks/upload", response_model=TaskStatusResponse)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book: str = Form("hongloumeng")
):
    """上传文本文件并创建分析任务"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported currently.")

    # 先按上限+1 字节读，超了就 413，不把超大文件整个吞进内存
    limit = int(MAX_UPLOAD_MB * 1024 * 1024)
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过 {MAX_UPLOAD_MB} MB 上限（可用 MAX_UPLOAD_MB 调整）",
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gbk")
        except:
            raise HTTPException(status_code=400, detail="Failed to decode file. Please ensure it is UTF-8 or GBK encoded.")
    
    task_id = str(uuid.uuid4())
    current_time = time.time()
    
    task = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "任务已创建，等待调度",
        "book_name": book,
        "created_at": current_time,
    }
    tasks_db.create(task)
    
    background_tasks.add_task(run_analysis_task, task_id, text, book)
    
    return TaskStatusResponse(**task)

@app.get("/api/v1/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取指定任务的状态与进度"""
    task_info = tasks_db.get(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
        
    # 如果任务未完成，不返回完整的 results 以减少网络传输
    if task_info["status"] != "completed":
        task_info["results"] = None
        
    return TaskStatusResponse(**task_info)

@app.get("/api/v1/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    """获取指定任务的完整分析结果"""
    task_info = tasks_db.get(task_id)
    if task_info is None:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task is not completed yet")
        
    return task_info["results"]

from src.simulation.sandbox import SandboxRequest, MultiAgentSandbox

@app.post("/api/v1/simulation/run")
async def run_simulation(req: SandboxRequest):
    task_data = tasks_db.get(req.task_id)
    if task_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if task_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task is not completed yet")
        
    profiles_list = task_data["results"].get("profiles", [])
    profiles_dict = {p["name"]: p for p in profiles_list}
    graph_data = task_data["results"].get("network_analysis", {}).get("graph_data", {})
    
    # 将沙盘实例化，并注入单例的 rag_db
    sandbox = MultiAgentSandbox()
    sandbox.rag_db = get_rag_db()
    
    try:
        res = await sandbox.simulate(req, profiles_dict, graph_data)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    task_id: str
    query: str
    book: str

@app.post("/api/v1/rag/chat")
async def rag_chat(req: ChatRequest):
    try:
        from openai import AsyncOpenAI
        import os
        
        rag_db = get_rag_db()
        snippets = rag_db.search(query=req.query, book_name=req.book, top_k=5)
        
        # 2. 图谱检索 (Graph RAG)
        graph_context = ""
        task_for_graph = tasks_db.get(req.task_id) if req.task_id else None
        if task_for_graph and task_for_graph["status"] == "completed":
            graph_data = task_for_graph["results"]["network_analysis"]["graph_data"]
            nodes = [n["id"] for n in graph_data.get("nodes", [])]
            # 简单的实体链接：看 query 中命中了哪些人物节点
            matched_entities = [n for n in nodes if n in req.query]
            if matched_entities:
                graph_context = "【图谱知识库检索结果】：\n"
                for entity in matched_entities:
                    # 查找该实体的一度关联边
                    edges = [e for e in graph_data.get("links", []) if e["source"] == entity or e["target"] == entity]
                    # 按照权重排序，取 top 5
                    edges = sorted(edges, key=lambda x: x.get("weight", 0), reverse=True)[:5]
                    if edges:
                        graph_context += f"人物 [{entity}] 的核心关系网：\n"
                        for e in edges:
                            other = e["target"] if e["source"] == entity else e["source"]
                            sentiment_str = "亲密" if e.get("sentiment") == "positive" else "敌对" if e.get("sentiment") == "negative" else "中立"
                            graph_context += f"  - 与 [{other}] 是 {e.get('type', '关联')} 关系 ({sentiment_str})，互动片段：\"{e.get('context_snippet', '')}\"\n"
                graph_context += "\n"
        
        if not snippets and not graph_context:
            return {"answer": "在原著及图谱中未检索到相关内容。", "sources": []}
            
        # 3. 组装上下文
        text_context = "【原著文本检索结果】：\n" + "\n\n".join([f"片段 {i+1}:\n{s['text']}" for i, s in enumerate(snippets)])
        
        combined_context = f"{graph_context}{text_context}"
        
        # 4. 大模型回答
        client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        
        prompt = f"""你是一个数字人文研究助手。请基于以下基于 GraphRAG (图谱+文本双重检索) 获取的上下文，回答用户的问题。
要求：
1. 必须严格基于提供的图谱关系和原文片段回答，不要编造。
2. 尽可能综合图谱中的宏观关系与文本片段中的微观细节。
3. 如果上下文不足以回答，请明确告知。

{combined_context}

【用户问题】：{req.query}
"""
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": snippets,
            "graph_entities": matched_entities if 'matched_entities' in locals() else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
