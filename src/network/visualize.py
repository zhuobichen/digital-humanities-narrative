"""人物关系网络可视化

支持 matplotlib 静态图和 PyVis 交互式图两种输出方式。
"""

import os
from typing import Any, Optional
import networkx as nx


class NetworkVisualizer:
    """网络可视化器

    使用 matplotlib 生成静态图，PyVis 生成交互式 HTML 图。
    """

    def __init__(self, network_builder=None):
        """
        Args:
            network_builder: CharacterNetworkBuilder 实例
        """
        self.builder = network_builder
        self.G: nx.Graph | None = (
            network_builder.G if network_builder else None
        )

    def set_graph(self, G: nx.Graph):
        """设置要可视化的图"""
        self.G = G

    def to_matplotlib(
        self,
        output_path: Optional[str] = None,
        figsize: tuple[int, int] = (14, 10),
        title: str = "人物关系网络",
        show_labels: bool = True,
    ):
        """使用 matplotlib 绘制静态网络图

        Args:
            output_path: 输出文件路径（不指定则显示）
            figsize: 画布大小
            title: 图标题
            show_labels: 是否显示标签
        """
        import matplotlib.pyplot as plt

        if self.G is None or self.G.number_of_nodes() == 0:
            raise ValueError("图为空，请先 build_network")

        G = self.G

        plt.figure(figsize=figsize)

        # 布局
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

        # 计算节点大小（基于度中心性）
        degree_cent = nx.degree_centrality(G)
        node_sizes = [degree_cent.get(node, 0.1) * 3000 + 300 for node in G.nodes()]

        # 节点颜色基于社区
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            community_map: dict[str, int] = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    community_map[node] = i
            node_colors = [community_map.get(node, 0) for node in G.nodes()]
        except Exception:
            node_colors = [0] * G.number_of_nodes()

        # 边的粗细基于权重
        edge_widths = [
            G[u][v].get("weight", 1) * 1.5 for u, v in G.edges()
        ]

        # 绘制
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.Set3,
            alpha=0.85,
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            alpha=0.4,
            edge_color="#888888",
            style="solid",
        )

        if show_labels:
            nx.draw_networkx_labels(
                G, pos,
                font_size=11,
                font_family="SimHei",
            )

        plt.title(title, fontsize=16, fontfamily="SimHei")
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        else:
            plt.show()
            return None

    def to_standalone_html(
        self,
        output_path: str = "character_network.html",
        height: str = "700px",
        width: str = "100%",
        bgcolor: str = "#f5f5f5",
        title: str = "人物关系网络",
    ) -> str:
        """生成完全自包含的 HTML 交互式网络图（零外部依赖）

        将 vis-network JS/CSS 内嵌到 HTML 中，无需任何 CDN 即可离线查看。

        Args:
            output_path: HTML 输出路径
            height: 画布高度
            width: 画布宽度
            bgcolor: 背景色
            title: 页面标题

        Returns:
            输出文件路径
        """
        if self.G is None or self.G.number_of_nodes() == 0:
            raise ValueError("图为空，请先 build_network")

        G = self.G
        person_nodes = [n for n, d in G.nodes(data=True) if d.get("type") != "location"]
        person_graph = G.subgraph(person_nodes) if person_nodes else G

        degree_cent = nx.degree_centrality(person_graph) if len(person_graph) > 1 else {}
        try:
            betweenness = nx.betweenness_centrality(person_graph) if len(person_graph) > 2 else {}
        except Exception:
            betweenness = {}
        try:
            communities = list(nx.community.greedy_modularity_communities(person_graph))
        except Exception:
            communities = []
        community_of: dict[str, int] = {}
        for idx, group in enumerate(communities):
            for name in group:
                community_of[name] = idx

        # 节点：颜色按社区，大小按度中心性（开方避免主角一个吃掉整个画面）
        nodes_json = []
        for node in G.nodes():
            if G.nodes[node].get("type") == "location":
                continue
            deg = person_graph.degree(node) if node in person_graph else 0
            dc = degree_cent.get(node, 0.0)
            bc = betweenness.get(node, 0.0)
            size = 14 + (dc ** 0.5) * 46
            nodes_json.append({
                "id": node,
                "label": node,
                "value": size,
                "community": community_of.get(node, -1),
                "degree": deg,
                "degree_centrality": round(dc, 4),
                "betweenness": round(bc, 4),
                "title": (
                    f"<b>{node}</b><br>"
                    f"连接数 {deg} · 度中心性 {dc:.3f}<br>"
                    f"介数中心性 {bc:.3f}"
                ),
            })
        nodes_json.sort(key=lambda n: -n["degree"])

        # 边：颜色按主类型，粗细按权重
        edges_json = []
        for u, v, data in G.edges(data=True):
            if data.get("type") == "location_link":
                continue
            weight = data.get("weight", 1)
            types = data.get("types") or [data.get("type") or "association"]
            edges_json.append({
                "from": u,
                "to": v,
                "value": weight,
                "types": types,
                "type": types[0],
                "title": f"{' / '.join(types)}<br>共现 {weight} 次",
            })

        main_character = max(degree_cent, key=degree_cent.get) if degree_cent else None
        stats = {
            "characters": len(nodes_json),
            "relations": len(edges_json),
            "density": round(nx.density(person_graph), 3) if len(person_graph) > 1 else 0.0,
            "main": main_character or "-",
            "communities": len(communities),
        }

        # 生成内嵌 vis-network 的独立 HTML
        html = self._build_standalone_html(
            nodes_json, edges_json, height, width, bgcolor, title, stats
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return os.path.abspath(output_path)

    def _build_standalone_html(
        self,
        nodes: list,
        edges: list,
        height: str,
        width: str,
        bgcolor: str,
        title: str,
        stats: dict | None = None,
    ) -> str:
        """构建自包含 HTML（内嵌 vis-network JS/CSS）"""
        stats = stats or {}
        import base64
        import json as jmod

        # 读取本地 vis-network JS（从 unpkg 下载的）
        js_code = "// vis-network placeholder"
        css_code = "/* vis-network placeholder */"

        # 尝试从 examples 目录加载 JS/CSS
        lib_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        examples_dir = os.path.join(lib_dir, "examples")
        js_path = os.path.join(examples_dir, "vis-network.min.js")
        css_path = os.path.join(examples_dir, "vis-network.min.css")

        try:
            with open(js_path, "r", encoding="utf-8") as f:
                js_code = f.read()
        except FileNotFoundError:
            pass

        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_code = f.read()
        except FileNotFoundError:
            pass

        nodes_str = jmod.dumps(nodes, ensure_ascii=False)
        edges_str = jmod.dumps(edges, ensure_ascii=False)
        stats_str = jmod.dumps(stats, ensure_ascii=False)

        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{css_code}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --bg: #0d1117; --panel: #161b22; --panel-2: #1c2430;
    --line: #2b3444; --text: #e6edf3; --dim: #8b98a9; --accent: #58a6ff;
  }}
  html, body {{ height: 100%; }}
  body {{
    font-family: -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
    background: radial-gradient(1200px 800px at 20% -10%, #1b2735 0%, var(--bg) 55%);
    color: var(--text); overflow: hidden;
  }}
  #app {{ display: grid; grid-template-columns: 268px 1fr; grid-template-rows: auto 1fr;
          height: 100vh; }}

  /* ---------- 顶栏 ---------- */
  #topbar {{
    grid-column: 1 / -1; display: flex; align-items: center; gap: 18px;
    padding: 14px 22px; border-bottom: 1px solid var(--line);
    background: rgba(13,17,23,0.85); backdrop-filter: blur(8px);
  }}
  #topbar h1 {{ font-size: 16px; font-weight: 600; letter-spacing: .3px; }}
  #topbar h1 span {{ color: var(--dim); font-weight: 400; font-size: 13px; margin-left: 8px; }}
  .chips {{ display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }}
  .chip {{
    background: var(--panel-2); border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 12px; font-size: 12px; color: var(--dim); white-space: nowrap;
  }}
  .chip b {{ color: var(--text); font-weight: 600; margin-left: 4px; }}

  /* ---------- 侧栏 ---------- */
  #side {{
    border-right: 1px solid var(--line); background: var(--panel);
    padding: 16px 14px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px;
  }}
  #side::-webkit-scrollbar {{ width: 8px; }}
  #side::-webkit-scrollbar-thumb {{ background: #2b3444; border-radius: 4px; }}
  .sec-title {{
    font-size: 11px; color: var(--dim); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; font-weight: 600;
  }}
  #search {{
    width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--bg); color: var(--text); font-size: 13px; outline: none;
  }}
  #search:focus {{ border-color: var(--accent); }}
  .legend-row {{
    display: flex; align-items: center; gap: 8px; padding: 5px 6px;
    border-radius: 6px; font-size: 12.5px; cursor: pointer; user-select: none;
  }}
  .legend-row:hover {{ background: var(--panel-2); }}
  .legend-row.off {{ opacity: .35; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }}
  .legend-name {{ flex: 1; }}
  .legend-count {{ color: var(--dim); font-variant-numeric: tabular-nums; }}
  .bar {{ height: 3px; border-radius: 2px; margin-left: 17px; margin-bottom: 2px; opacity: .55; }}

  /* ---------- 画布 ---------- */
  #stage {{ position: relative; }}
  #stage {{ min-height: 0; }}  /* grid 子项必须允许收缩 */
  #mnt {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
  #hint {{
    position: absolute; left: 16px; bottom: 14px; font-size: 11.5px; color: var(--dim);
    background: rgba(13,17,23,.72); padding: 6px 10px; border-radius: 8px;
    border: 1px solid var(--line); pointer-events: none;
  }}

  /* ---------- 详情面板 ---------- */
  #detail {{
    position: absolute; right: 16px; top: 16px; width: 290px; max-height: calc(100% - 32px);
    overflow-y: auto; background: rgba(22,27,34,.96); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px; font-size: 13px;
    box-shadow: 0 12px 32px rgba(0,0,0,.45); display: none;
  }}
  #detail.show {{ display: block; }}
  #detail h2 {{ font-size: 17px; margin-bottom: 10px; }}
  #detail .kv {{ display: flex; justify-content: space-between; padding: 4px 0;
                 border-bottom: 1px dashed var(--line); font-size: 12.5px; }}
  #detail .kv span:last-child {{ color: var(--accent); font-variant-numeric: tabular-nums; }}
  #detail .neighbor {{ display: flex; align-items: center; gap: 6px; padding: 3px 0; font-size: 12px; }}
  #detail .close {{ position: absolute; right: 12px; top: 10px; cursor: pointer;
                    color: var(--dim); font-size: 16px; line-height: 1; }}
  #detail .close:hover {{ color: var(--text); }}
</style>
</head>
<body>
<div id="app">
  <div id="topbar">
    <h1>{title}<span>人物关系网络</span></h1>
    <div class="chips" id="chips"></div>
  </div>

  <aside id="side">
    <div>
      <div class="sec-title">搜索人物</div>
      <input id="search" placeholder="输入名字高亮…" autocomplete="off">
    </div>
    <div>
      <div class="sec-title">关系类型（点击过滤）</div>
      <div id="type-legend"></div>
    </div>
    <div>
      <div class="sec-title">社群（颜色）</div>
      <div id="comm-legend"></div>
    </div>
  </aside>

  <main id="stage">
    <div id="mnt"></div>
    <div id="hint">滚轮缩放 · 拖拽平移 · 点击节点看详情</div>
    <div id="detail">
      <div class="close" id="close-detail">×</div>
      <div id="detail-body"></div>
    </div>
  </main>
</div>

<script>
{js_code}
</script>
<script>
(function () {{
  var RAW_NODES = {nodes_str};
  var RAW_EDGES = {edges_str};
  var STATS = {stats_str};

  // 关系类型 → 颜色（语义化：冲突红、亲密粉、对话蓝、协作青…）
  var TYPE_COLOR = {{
    hostile:       '#f85149',
    affection:     '#ff7b9c',
    kinship:       '#c297ff',
    dialogic:      '#58a6ff',
    collaborative: '#3fb950',
    hierarchical:  '#f0a020',
    social:        '#39c5cf',
    association:   '#6e7b8b'
  }};
  var TYPE_CN = {{
    hostile: '冲突', affection: '亲密', kinship: '亲缘', dialogic: '对话',
    collaborative: '协作', hierarchical: '上下级', social: '社交', association: '泛关联'
  }};
  // 社区配色（12 色循环）
  var COMM_COLOR = ['#58a6ff','#3fb950','#f0a020','#c297ff','#39c5cf','#ff7b9c',
                    '#f85149','#a5d6ff','#8ddb8c','#ffd479','#d2a8ff','#7ee0e6'];

  function typeColor(t) {{ return TYPE_COLOR[t] || TYPE_COLOR.association; }}

  // ---- 统计条 ----
  var chips = [
    ['人物', STATS.characters],
    ['关系', STATS.relations],
    ['社群', STATS.communities],
    ['密度', STATS.density],
    ['主角', STATS.main]
  ];
  document.getElementById('chips').innerHTML = chips.map(function (c) {{
    return '<div class="chip">' + c[0] + '<b>' + c[1] + '</b></div>';
  }}).join('');

  // ---- 节点/边 ----
  var nodes = new vis.DataSet(RAW_NODES.map(function (n) {{
    var c = n.community >= 0 ? COMM_COLOR[n.community % COMM_COLOR.length] : '#6e7b8b';
    var r = Math.max(11, Math.min(n.value * 0.58, 30));
    return {{
      id: n.id, label: n.label, value: r, title: n.title,
      color: {{
        background: c + 'cc', border: '#0d1117',
        highlight: {{ background: c, border: '#ffffff' }},
        hover: {{ background: c, border: '#ffffff' }}
      }},
      font: {{ color: '#ffffff', size: 12, face: 'Microsoft YaHei',
               strokeWidth: 3, strokeColor: '#0d1117' }},
      borderWidth: 1.5, borderWidthSelected: 3,
      shadow: {{ enabled: true, color: c + '66', size: 14, x: 0, y: 0 }}
    }};
  }}));

  // 一条边可能同时有多个类型；取最"具体"的那个上色（泛关联排最后）
  var TYPE_PRIORITY = ['hostile','affection','kinship','hierarchical','dialogic','collaborative','social','association'];
  function primaryType(types) {{
    for (var i = 0; i < TYPE_PRIORITY.length; i++) {{
      if (types.indexOf(TYPE_PRIORITY[i]) >= 0) return TYPE_PRIORITY[i];
    }}
    return 'association';
  }}

  var edges = new vis.DataSet(RAW_EDGES.map(function (e) {{
    e.type = primaryType(e.types || [e.type]);
    var c = typeColor(e.type);
    return {{
      from: e.from, to: e.to, title: e.title, types: e.types,
      value: e.value,
      color: {{ color: c + '80', highlight: c, hover: c, opacity: 0.9 }},
      width: Math.max(1.2, Math.min(0.9 + Math.sqrt(e.value) * 1.35, 6)),
      selectionWidth: 2.5, hoverWidth: 1.6,
      smooth: {{ enabled: true, type: 'continuous', roundness: 0.18 }}
    }};
  }}));

  // ---- 图例 ----
  var typeCount = {{}};
  RAW_EDGES.forEach(function (e) {{
    var t = primaryType(e.types || [e.type]);
    typeCount[t] = (typeCount[t] || 0) + 1;
  }});
  var maxType = Math.max.apply(null, Object.keys(typeCount).map(function (k) {{ return typeCount[k]; }}));
  var legendEl = document.getElementById('type-legend');
  Object.keys(typeCount).sort(function (a, b) {{ return typeCount[b] - typeCount[a]; }})
    .forEach(function (t) {{
      var pct = Math.round(typeCount[t] / maxType * 100);
      var row = document.createElement('div');
      row.className = 'legend-row';
      row.dataset.type = t;
      row.innerHTML = '<span class="dot" style="background:' + typeColor(t) + '"></span>' +
        '<span class="legend-name">' + (TYPE_CN[t] || t) + '</span>' +
        '<span class="legend-count">' + typeCount[t] + '</span>';
      var bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.width = pct + '%';
      bar.style.background = typeColor(t);
      row.addEventListener('click', function () {{
        row.classList.toggle('off');
        var hidden = {{}};
        Array.prototype.forEach.call(legendEl.children, function (c) {{
          if (c.classList && c.classList.contains('off')) hidden[c.dataset.type] = true;
        }});
        edges.forEach(function (e) {{
          var visible = !hidden[e.types[0]] && !e.types.some(function (t) {{ return hidden[t]; }});
          edges.update({{ id: e.id, hidden: !visible }});
        }});
      }});
      legendEl.appendChild(row);
      legendEl.appendChild(bar);
    }});

  // ---- 社群图例 ----
  var commEl = document.getElementById('comm-legend');
  var commSeen = {{}};
  RAW_NODES.forEach(function (n) {{
    if (n.community < 0 || commSeen[n.community]) return;
    commSeen[n.community] = true;
    var d = document.createElement('div');
    d.className = 'legend-row';
    d.innerHTML = '<span class="dot" style="background:' + COMM_COLOR[n.community % COMM_COLOR.length] + '"></span>' +
      '<span class="legend-name">社群 ' + (n.community + 1) + '</span>';
    commEl.appendChild(d);
  }});

  // ---- 网络 ----
  var container = document.getElementById('mnt');
  var options = {{
    physics: {{
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {{
        gravitationalConstant: -95, centralGravity: 0.008,
        springLength: 240, springConstant: 0.075, damping: 0.42, avoidOverlap: 0.65
      }},
      stabilization: {{ iterations: 260, fit: true }},
      minVelocity: 0.6
    }},
    nodes: {{ shape: 'dot', scaling: {{ min: 11, max: 30 }} }},
    edges: {{ selectionWidth: 2 }},
    interaction: {{
      hover: true, hoverConnectedEdges: true, tooltipDelay: 140,
      multiselect: false, navigationButtons: false, keyboard: false
    }}
  }};
  var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);

  // 物理稳定后关掉，省电、也避免拖拽时乱飘
  network.once('stabilizationIterationsDone', function () {{
    network.setOptions({{ physics: {{ enabled: false }} }});
    network.fit({{ animation: {{ duration: 500 }} }});
  }});
  // 兜底：个别情况下 stabilization 事件不触发，1.2s 后强制 fit 一次
  setTimeout(function () {{ network.fit(); }}, 1200);

  // ---- 详情面板 ----
  var detail = document.getElementById('detail');
  var detailBody = document.getElementById('detail-body');
  document.getElementById('close-detail').onclick = function () {{
    detail.classList.remove('show');
    network.unselectAll();
  }};

  network.on('click', function (params) {{
    if (!params.nodes.length) {{ detail.classList.remove('show'); return; }}
    var id = params.nodes[0];
    var n = RAW_NODES.filter(function (x) {{ return x.id === id; }})[0];
    if (!n) return;

    var neigh = network.getConnectedNodes(id);
    var rows = neigh.map(function (m) {{
      var es = network.getConnectedEdges(id).map(function (eid) {{
        var e = edges.get(eid);
        return e && (e.from === m || e.to === m) ? e : null;
      }}).filter(Boolean);
      var t = es.length && es[0].types ? es[0].types[0] : 'association';
      return '<div class="neighbor"><span class="dot" style="background:' + typeColor(t) + '"></span>' +
             m + '<span style="margin-left:auto;color:#8b98a9">' + (TYPE_CN[t] || t) + '</span></div>';
    }}).join('');

    detailBody.innerHTML =
      '<h2>' + n.label + '</h2>' +
      '<div class="kv"><span>连接数</span><span>' + n.degree + '</span></div>' +
      '<div class="kv"><span>度中心性</span><span>' + n.degree_centrality + '</span></div>' +
      '<div class="kv"><span>介数中心性</span><span>' + n.betweenness + '</span></div>' +
      '<div class="sec-title" style="margin-top:14px">直接相连（' + neigh.length + '）</div>' +
      (rows || '<div style="color:#8b98a9;font-size:12px">无（孤立节点）</div>');
    detail.classList.add('show');
  }});

  // ---- 搜索高亮 ----
  var search = document.getElementById('search');
  search.addEventListener('input', function () {{
    var q = search.value.trim();
    nodes.forEach(function (n) {{
      var hit = q && n.label.indexOf(q) >= 0;
      nodes.update({{ id: n.id, borderWidth: hit ? 4 : 2,
                     color: hit ? {{ background: '#ffffff', border: '#58a6ff' }} : undefined }});
    }});
    if (q) {{
      var first = RAW_NODES.filter(function (n) {{ return n.label.indexOf(q) >= 0; }})[0];
      if (first) network.focus(first.id, {{ scale: 1.1, animation: true }});
    }}
  }});
}})();
</script>
</body>
</html>"""


    def to_cytoscape_elements(self) -> list[dict[str, Any]]:
        """导出为 Cytoscape.js 格式的 elements

        Returns:
            Cytoscape.js elements 列表
        """
        if self.G is None:
            return []

        elements = []
        for node in self.G.nodes():
            elements.append({"data": {"id": node, "label": node}})

        for u, v, data in self.G.edges(data=True):
            elements.append({
                "data": {
                    "id": f"{u}-{v}",
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1),
                    "types": ",".join(data.get("types", [])),
                }
            })

        return elements

    def to_d3_json(self) -> dict[str, Any]:
        """导出为 D3.js 力导向图格式的 JSON

        Returns:
            {"nodes": [...], "links": [...]}
        """
        if self.G is None:
            return {"nodes": [], "links": []}

        nodes = [{"id": n, "name": n, "top_locations": self.G.nodes[n].get("top_locations", [])} for n in self.G.nodes()]
        links = [
            {
                "source": u,
                "target": v,
                "weight": d.get("weight", 1),
                "types": d.get("types", []),
                "sentiment": d.get("sentiment", "neutral"),
                "context_snippet": d.get("context_snippet", ""),
                "contexts": d.get("contexts", [])
            }
            for u, v, d in self.G.edges(data=True)
        ]

        return {"nodes": nodes, "links": links}
