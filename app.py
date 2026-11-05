"""Streamlit 交互式可视化界面

为数字人文叙事分析提供易用的 Web UI。
"""

import os
import sys
import streamlit as st
import networkx as nx
from datetime import datetime

# 确保能找到 src 模块
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import NarrativePipeline
from src.base.character_dict import HONGLOUMENG_CHARACTERS

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="数字人文叙事分析",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 隐藏右上角的菜单和水印
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 辅助函数：绘制关系图 (PyEcharts)
# ==========================================
def render_network_graph(network_data, width="100%", height="600px"):
    """使用 PyEcharts 渲染交互式关系图"""
    from pyecharts import options as opts
    from pyecharts.charts import Graph
    import streamlit.components.v1 as components

    nodes = []
    links = []
    
    # 构建节点
    degree_cent = network_data.get("degree_centrality", {})
    communities = network_data.get("communities", [])
    
    # 将节点分配到社区类别
    categories = [{"name": f"社区 {i+1}"} for i in range(len(communities))] if communities else [{"name": "默认"}]
    
    node_comm_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_comm_map[node] = i
            
    for char in network_data.get("characters", []):
        size = int(degree_cent.get(char, 0.1) * 100) + 10
        category_idx = node_comm_map.get(char, 0)
        
        nodes.append({
            "name": char,
            "symbolSize": size,
            "category": category_idx,
            "value": round(degree_cent.get(char, 0), 3)
        })

    # 从内存重建边信息 (这里简化处理，在实际重构中应直接传递边数据)
    # 因为原分析结果中 network_data 未直接暴露所有边，我们需要从 pipeline 获取
    if "G_edges" in st.session_state:
        for u, v, data in st.session_state.G_edges:
            weight = data.get("weight", 1)
            links.append({
                "source": u,
                "target": v,
                "value": weight,
                "lineStyle": {"width": min(weight, 5)}
            })

    graph = (
        Graph()
        .add(
            "",
            nodes,
            links,
            categories=categories,
            layout="force",
            repulsion=4000,
            edge_symbol=["none", "none"],
            label_opts=opts.LabelOpts(is_show=True, position="right"),
            linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3, opacity=0.7),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="人物社交关系图谱"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
            tooltip_opts=opts.TooltipOpts(trigger="item", formatter="{b}: 度中心性 {c}")
        )
    )
    
    # 渲染到 HTML 并嵌入 Streamlit
    html_content = graph.render_embed()
    components.html(html_content, width=None, height=int(height.replace("px", "")), scrolling=True)

# ==========================================
# 辅助函数：绘制性格雷达图
# ==========================================
def render_radar_chart(profile):
    """渲染单个人物的性格雷达图"""
    from pyecharts import options as opts
    from pyecharts.charts import Radar
    import streamlit.components.v1 as components

    traits = profile.get("traits", {})
    if not traits:
        st.warning("无足够特质数据")
        return
        
    schema = [
        opts.RadarIndicatorItem(name="领导力", max_=1.0),
        opts.RadarIndicatorItem(name="创造力", max_=1.0),
        opts.RadarIndicatorItem(name="随和性", max_=1.0),
        opts.RadarIndicatorItem(name="尽责性", max_=1.0),
        opts.RadarIndicatorItem(name="外向性", max_=1.0),
        opts.RadarIndicatorItem(name="情绪稳定", max_=1.0),
        opts.RadarIndicatorItem(name="主见性", max_=1.0),
        opts.RadarIndicatorItem(name="合作性", max_=1.0),
    ]
    
    values = [[
        traits.get("leadership", 0),
        traits.get("creativity", 0),
        traits.get("agreeableness", 0),
        traits.get("conscientiousness", 0),
        traits.get("extraversion", 0),
        traits.get("emotional_stability", 0),
        traits.get("assertiveness", 0),
        traits.get("cooperativeness", 0),
    ]]
    
    radar = (
        Radar()
        .add_schema(schema=schema)
        .add(profile.get("name", "未知"), values, color="#4A90D9", 
             areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title=f"{profile.get('name')} 核心特质"))
    )
    
    components.html(radar.render_embed(), width=None, height=400)


# ==========================================
# 侧边栏：参数配置
# ==========================================
with st.sidebar:
    st.title("⚙️ 分析配置")
    
    input_mode = st.radio("选择输入方式", ["使用内置示例 (红楼梦)", "手动输入文本"])
    
    use_dict = st.checkbox("启用红楼梦专属人名词典", value=True)
    convert_t2s = st.checkbox("繁体转简体", value=True)
    
    st.markdown("---")
    st.markdown("### 👨‍💻 关于本项目")
    st.markdown("数字人文叙事分析工具，将非结构化文学文本转化为结构化的知识图谱、因果链与命运预测。")


# ==========================================
# 主界面
# ==========================================
st.title("📚 数字人文叙事分析平台")
st.markdown("上传或输入文本，AI 将自动重构故事网络并进行深度人文分析。")

# 1. 数据输入
text_input = ""
if input_mode == "使用内置示例 (红楼梦)":
    st.info("已加载《红楼梦》大观园经典桥段示例文本。")
    default_text = """宝玉和黛玉在大观园里散步，宝钗也走了过来。
王熙凤管理着贾府的大小事务，贾母非常信任她。
贾政对宝玉很严厉，经常训斥他不务正业。
林黛玉身体虚弱，经常咳嗽，宝玉很担心她。
薛宝钗性格温和，大家都很喜欢她。
王熙凤和贾琏是夫妻，但她和贾蓉也有暧昧关系。
贾母是贾府的最高权威，所有人都尊敬她。
袭人是宝玉的贴身丫鬟，细心照顾着他的起居。
贾宝玉和林黛玉经常在一起看书写诗。
贾宝玉对贾政很害怕，每次见父亲都紧张。
薛宝钗劝宝玉要好好读书，考取功名。
林黛玉因为宝玉和宝钗的关系而暗自伤心。
贾母决定让宝玉娶宝钗为妻。
林黛玉听到宝玉要娶宝钗的消息后病重。"""
    text_input = st.text_area("文本预览 (可编辑)", value=default_text, height=250)
else:
    text_input = st.text_area("在此粘贴您要分析的小说或剧本段落（建议 500-2000 字）", height=250)

# 2. 执行分析
if st.button("🚀 开始深度分析", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("请输入有效文本！")
    else:
        with st.spinner("正在运行 NLP 流水线（实体识别 -> 关系抽取 -> 因果分析 -> 画像构建）..."):
            try:
                # 初始化 Pipeline
                char_dict = HONGLOUMENG_CHARACTERS if use_dict else None
                pipeline = NarrativePipeline(
                    character_names=char_dict,
                    convert_traditional=convert_t2s
                )
                
                # 分段
                paragraphs = [p.strip() for p in text_input.split('\n') if p.strip()]
                
                # 运行
                results = pipeline.run(paragraphs, verbose=False)
                
                # 将图的边存入 session_state 供 PyEcharts 使用
                if pipeline.network_builder.G:
                    st.session_state.G_edges = list(pipeline.network_builder.G.edges(data=True))
                
                st.session_state.results = results
                st.success("分析完成！请在下方查看结果。")
                
            except Exception as e:
                st.error(f"分析过程中发生错误: {str(e)}")

st.markdown("---")

# 3. 结果展示面板
if "results" in st.session_state:
    results = st.session_state.results
    
    # 顶部指标卡片
    net_stats = results.get("network_analysis", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("识别人物数", len(results.get("characters", [])))
    col2.metric("提取关系数", len(results.get("relations", [])))
    col3.metric("叙事事件数", len(results.get("events", [])))
    col4.metric("网络中心主角", net_stats.get("main_character", "无"))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 标签页切换视图
    tab1, tab2, tab3, tab4 = st.tabs(["🕸️ 人物图谱", "🎭 人物画像", "📈 叙事与情节", "📖 自动故事生成"])
    
    # Tab 1: 人物图谱
    with tab1:
        st.subheader("人物社交网络拓扑")
        render_network_graph(net_stats)
        
        with st.expander("查看原始关系数据"):
            st.dataframe([{
                "发起方": r["source"], 
                "接收方": r["target"], 
                "关系类型": r.get("type", ""),
                "情感极性": f"{r.get('sentiment', 0.5):.2f}",
                "上下文": r.get("context", "")[:50] + "..."
            } for r in results.get("relations", [])])
            
    # Tab 2: 人物画像
    with tab2:
        profiles = results.get("profiles", [])
        if profiles:
            # 左侧选人，右侧看详情
            char_names = [p["name"] for p in profiles]
            selected_char = st.selectbox("选择人物查看详情", char_names)
            
            # 找到对应 profile
            profile = next((p for p in profiles if p["name"] == selected_char), None)
            
            if profile:
                col_left, col_right = st.columns([1, 1.5])
                
                with col_left:
                    st.markdown(f"### {profile['name']}")
                    st.markdown(f"**角色定位**：`{profile.get('role_in_story', 'unknown')}`")
                    st.markdown(f"**性格总评**：\n> {profile.get('personality_summary', '暂无')}")
                    
                    # 命运预测
                    destiny = results.get("destiny_predictions", {}).get(selected_char, {})
                    if destiny:
                        st.markdown("---")
                        outlook_color = "green" if destiny.get("overall_outlook") == "positive" else "red" if destiny.get("overall_outlook") == "negative" else "gray"
                        st.markdown(f"**命运预测走势**：:{outlook_color}[{destiny.get('overall_outlook', '未知').upper()}] (置信度: {destiny.get('overall_confidence', 0):.2f})")
                        st.markdown(f"*{destiny.get('summary', '')}*")
                
                with col_right:
                    render_radar_chart(profile)
    
    # Tab 3: 叙事与情节
    with tab3:
        arc = results.get("narrative_arc", {})
        st.subheader("五幕剧叙事弧线")
        
        # 简单的时间线渲染
        stage_names = {
            "exposition": "1. 开端 (Exposition)",
            "rising_action": "2. 发展 (Rising Action)",
            "climax": "3. 高潮 (Climax)",
            "falling_action": "4. 下降 (Falling Action)",
            "resolution": "5. 结局 (Resolution)"
        }
        
        for stage_key, title in stage_names.items():
            events = arc.get(stage_key, [])
            with st.container():
                st.markdown(f"#### {title}")
                if not events:
                    st.caption("该阶段暂无明显事件")
                for e in events:
                    chars = "、".join(e.get("characters", []))
                    st.markdown(f"- **[{chars}]** {e.get('description', '')}")
                    
    # Tab 4: 自动故事生成
    with tab4:
        st.subheader("AI 结构化重写")
        st.markdown(results.get("story", "未生成故事"))
