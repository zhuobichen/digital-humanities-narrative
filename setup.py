from setuptools import setup, find_packages

# 审计项：原来 install_requires 一次性安装 torch / transformers / chromadb，
# `pip install .` 会拉几 GB。这里拆成 extras：默认装"核心 + Web + 可视化"，
# 重依赖（torch / transformers / chromadb）走 .[llm]。
#
#   pip install -e .              # 默认
#   pip install -e ".[llm]"       # LLM 实体关系 / 人物画像 / 向量检索
#   pip install -e ".[all]"       # 全量
#   pip install -e ".[dev]"       # 测试

CORE = [
    "networkx>=3.2",
    "python-dateutil>=2.8",
    "jieba>=0.42",
    "numpy>=1.24",
    "pandas>=2.0",
    "diskcache>=5.6",
    "tenacity>=8.2",
    "opencc>=1.1",
    "rich>=13.0",
    "typer>=0.12",
    "pydantic>=2.0",
    "python-dotenv>=1.0",
]

WEB = [
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "streamlit>=1.30",
    "pyecharts>=2.0",
]

VIZ = [
    "matplotlib>=3.8",
    "pyvis>=0.3",
]

LLM = [
    "openai>=1.0",
    "transformers>=4.36",
    "torch>=2.1",
    "spacy>=3.7",
    "textblob>=0.17",
    "sentence-transformers>=2.6",
    "chromadb>=0.4",
]

DEV = [
    "pytest>=7.4",
    "httpx>=0.27",
]

setup(
    name="digital-humanities-narrative",
    version="0.1.0",
    description="数字人文叙事分析 - 从零散资料重构故事世界",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=CORE + WEB + VIZ,
    extras_require={
        "llm": LLM,
        "dev": DEV,
        "all": LLM,
    },
)
