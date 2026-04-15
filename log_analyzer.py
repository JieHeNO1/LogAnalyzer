import streamlit as st
import re
import os
import json
import pickle
import base64
import requests
from datetime import datetime
from pathlib import Path
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

import openai
import httpx
import chardet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(page_title="智能日志分析助手 (支持大文件+图片OCR)", layout="wide")
st.title("🔍 智能日志分析助手 (AI Powered)")

# ==================== 初始化组件 ====================

# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

# DashScope OCR 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_OCR_MODEL = os.getenv("DASHSCOPE_OCR_MODEL", "qwen-vl-plus")

# 配置 OpenAI 客户端指向 DeepSeek
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = DEEPSEEK_BASE_URL

# 知识库文件路径
KB_FILE = Path("knowledge_base.json")
VECTORIZER_FILE = Path("vectorizer.pkl")
VECTORS_FILE = Path("vectors.npy")
METADATA_FILE = Path("metadata.json")

# 错误码定义目录
ERROR_DEFS_DIR = Path("./error_defs")

# 尝试导入 pytesseract（备用 OCR）
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# ==================== 错误码定义加载 ====================

@st.cache_resource
def load_error_definitions():
    """
    加载 ./error_defs 目录下所有 .txt 文件，解析错误码定义。
    返回字典: { "0x00000070": { "component": "SCU", "user_msg": "...", "service_msg": "...", "severity": "Error", ... } }
    """
    error_db = {}
    if not ERROR_DEFS_DIR.exists():
        st.warning(f"错误码定义目录不存在: {ERROR_DEFS_DIR}")
        return error_db

    txt_files = list(ERROR_DEFS_DIR.glob("*.txt"))
    if not txt_files:
        st.warning(f"错误码定义目录中没有 .txt 文件: {ERROR_DEFS_DIR}")
        return error_db

    for filepath in txt_files:
        component = filepath.stem  # 文件名作为组件名，如 SCU, ASD
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                continue

            # 解析TSV格式（制表符分隔）
            # 第一行通常是表头，我们跳过，从第二行开始解析
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 8:
                    continue
                code_raw = parts[0].strip()
                # 提取十六进制错误码，如 "2(0x00000002)" -> "0x00000002"
                match = re.search(r'(0x[0-9a-fA-F]+)', code_raw)
                if not match:
                    continue
                code_hex = match.group(1).lower()  # 统一小写便于匹配
                user_msg_cn = parts[1].strip()
                user_msg_en = parts[2].strip()
                severity = parts[3].strip()
                recoverable = parts[4].strip()
                upload = parts[5].strip()
                service_msg = parts[6].strip()
                service_action = parts[7].strip()

                error_db[code_hex] = {
                    "component": component,
                    "code": code_hex,
                    "user_msg_cn": user_msg_cn,
                    "user_msg_en": user_msg_en,
                    "severity": severity,
                    "recoverable": recoverable,
                    "service_msg": service_msg,
                    "service_action": service_action,
                }
        except Exception as e:
            st.warning(f"解析错误码文件失败 {filepath}: {e}")

    return error_db

# ==================== 知识库管理 ====================

def load_knowledge_base():
    if KB_FILE.exists():
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_knowledge_base(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

def update_vectorizer_and_vectors(kb):
    if not kb:
        return None, None
    texts = [f"{item['query']} {item['solution']}" for item in kb]
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    np.save(VECTORS_FILE, vectors.toarray())
    metadata = [{"id": item["id"], "query": item["query"], "solution": item["solution"]} for item in kb]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return vectorizer, vectors

def load_vectorizer_and_vectors():
    if VECTORIZER_FILE.exists() and VECTORS_FILE.exists() and METADATA_FILE.exists():
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        vectors = np.load(VECTORS_FILE)
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return vectorizer, vectors, metadata
    return None, None, []

def find_similar_solutions(query, vectorizer, vectors, metadata, top_k=2):
    if vectorizer is None or vectors is None or not metadata:
        return []
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, vectors).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if sims[idx] > 0.1:
            results.append(metadata[idx]["solution"])
    return results

# ==================== OCR 功能 ====================

def ocr_image_dashscope(image_bytes):
    if not DASHSCOPE_API_KEY:
        return None
    try:
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": DASHSCOPE_OCR_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请识别并提取图片中的所有文字内容，不要添加额外说明。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                }
            ],
            "max_tokens": 1000
        }
        response = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            st.warning(f"DashScope OCR 请求失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.warning(f"DashScope OCR 异常: {e}")
        return None

def ocr_image_tesseract(image_bytes):
    if not TESSERACT_AVAILABLE:
        return None
    try:
        image = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(image, lang='eng+chi_sim')
        return text.strip()
    except Exception as e:
        st.warning(f"Tesseract OCR 异常: {e}")
        return None

def perform_ocr(uploaded_file):
    image_bytes = uploaded_file.read()
    text = ocr_image_dashscope(image_bytes)
    if text is not None:
        return text, "DashScope"
    text = ocr_image_tesseract(image_bytes)
    if text is not None:
        return text, "Tesseract (本地)"
    return None, "OCR 失败"

# ==================== 日志解析与检索 ====================

def parse_log_line(line, line_num):
    pattern = r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(\w+)\s+(\w+)\s+(\w+)\s+(\S+)\s+(.*)$'
    match = re.match(pattern, line.strip())
    if match:
        ts_str, level, type_, module, code, content = match.groups()
        try:
            ts = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S.%f")
        except:
            ts = ts_str
        return {
            "line_num": line_num,
            "timestamp": ts,
            "timestamp_str": ts_str,
            "level": level,
            "module": module,
            "code": code,
            "content": content,
            "raw": line.strip()
        }
    return None

def find_relevant_context(log_lines, query, context_lines=80):
    """
    极简检索：完全模拟 search_log.py 独立脚本的行为。
    - 将用户输入的 query 作为唯一关键词，不拆分。
    - 找出所有包含该关键词的行（忽略大小写）。
    - 合并所有匹配点的前后 context_lines 行。
    """
    if not query or not query.strip():
        return [], 0

    keyword = query.strip().lower()
    matched_indices = []
    for i, line in enumerate(log_lines):
        if keyword in line.lower():
            matched_indices.append(i)

    if not matched_indices:
        return [], 0

    included = set()
    for idx in matched_indices:
        start = max(0, idx - context_lines)
        end = min(len(log_lines), idx + context_lines + 1)
        included.update(range(start, end))

    return sorted(included), len(matched_indices)

def extract_error_codes_from_text(text):
    """从文本中提取十六进制错误码，如 0x00000070"""
    pattern = r'0x[0-9a-fA-F]{8}'
    matches = re.findall(pattern, text.lower())
    return list(set(matches))

def generate_analysis(log_snippet, user_query, similar_cases="", error_defs=None):
    """调用 DeepSeek 生成分析报告，并注入错误码定义"""
    # 提取日志片段中的错误码
    error_codes = extract_error_codes_from_text(log_snippet)
    error_context = ""
    if error_defs and error_codes:
        error_context = "\n【相关错误码官方定义】\n"
        for code in error_codes:
            if code in error_defs:
                defn = error_defs[code]
                error_context += f"- 错误码 {code} (组件: {defn['component']}):\n"
                error_context += f"  用户提示: {defn['user_msg_cn']}\n"
                error_context += f"  严重程度: {defn['severity']}\n"
                error_context += f"  服务提示: {defn['service_msg']}\n"
                error_context += f"  解决措施: {defn['service_action']}\n"
            else:
                error_context += f"- 错误码 {code}: 未在定义库中找到对应说明。\n"
        error_context += "\n"

    prompt = f"""
你是一名资深的MRI系统软件和硬件日志分析专家。请根据以下日志片段（按时间顺序排列）和用户的问题描述，进行详细的原因分析。

【用户问题描述】
{user_query}

【相关日志片段】
{log_snippet}
{error_context}
【历史相似案例参考】
{similar_cases if similar_cases else "无"}

【分析要求】
1. **错误直接原因**：明确指出报错代码或现象对应的直接失败点。如果有官方错误码定义，请优先参考。
2. **故障链追溯**：按时间顺序描述错误发生前的关键状态变化。
3. **潜在根本原因**：基于日志和领域知识，列出3个最可能的根本原因，并解释推断依据。
4. **排查建议**：提供具体、可操作的后续排查步骤，可结合官方解决措施。

请以清晰的结构化Markdown格式输出。
"""
    try:
        proxy_url = os.getenv("HTTP_PROXY")
        http_client = httpx.Client(proxy=proxy_url) if proxy_url else None
        
        client = openai.OpenAI(
            api_key=openai.api_key,
            base_url=openai.api_base,
            http_client=http_client
        )
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的日志分析专家，擅长MRI系统软硬件问题诊断。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 分析失败: {str(e)}"

def render_screenshot_content(analysis_md, log_highlight):
    return f"""
    <div id="screenshot-area" style="background:white; padding:20px; font-family:Arial; max-width:900px;">
        <h2>📋 日志分析报告</h2>
        <div style="margin:20px 0; line-height:1.6;">
            {analysis_md.replace(chr(10), '<br>')}
        </div>
        <h3>📄 关键日志上下文</h3>
        <pre style="background:#f5f5f5; padding:15px; border-radius:5px; overflow-x:auto; white-space:pre-wrap;">
{log_highlight}
        </pre>
    </div>
    """

# ==================== Streamlit 界面 ====================

# 加载错误码定义
error_definitions = load_error_definitions()
st.sidebar.markdown(f"📚 已加载 {len(error_definitions)} 条错误码定义")

with st.sidebar:
    st.header("⚙️ 配置")
    with st.expander("API 设置"):
        api_key_input = st.text_input("DeepSeek API Key", type="password", value=DEEPSEEK_API_KEY)
        base_url_input = st.text_input("Base URL", value=DEEPSEEK_BASE_URL)
        model_input = st.text_input("Model", value=DEEPSEEK_MODEL)
        if api_key_input:
            openai.api_key = api_key_input
            openai.api_base = base_url_input
            DEEPSEEK_MODEL = model_input

    st.divider()
    st.markdown("### 📚 知识库统计")
    kb = load_knowledge_base()
    st.metric("已存储解决方案", len(kb))

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 上传日志文件")
    uploaded_file = st.file_uploader("选择 .log 或 .txt 文件 (支持大文件/ANSI编码)", type=["log", "txt"])

    log_content = ""
    log_lines = []
    if uploaded_file:
        raw_data = uploaded_file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'gbk'
        st.caption(f"检测到文件编码: {encoding}")
        log_content = raw_data.decode(encoding, errors='ignore')
        log_lines = log_content.splitlines()
        st.success(f"已加载 {len(log_lines)} 行日志")

        with st.expander("📄 日志预览 (前100行)"):
            st.text("\n".join(log_lines[:100]))

    st.subheader("❓ 问题描述")
    user_query = st.text_area(
        "请描述问题（建议直接输入错误代码，如：SSW_0x00000070）",
        height=100,
        placeholder="例如：SSW_0x00000070"
    )

    st.subheader("🖼️ 图片上传 (可选，用于OCR识别)")
    uploaded_image = st.file_uploader("支持 PNG / JPG / JPEG", type=["png", "jpg", "jpeg"])
    ocr_text = ""
    if uploaded_image:
        with st.spinner("正在进行 OCR 识别..."):
            ocr_result, method = perform_ocr(uploaded_image)
            if ocr_result:
                st.success(f"OCR 识别成功 (方法: {method})")
                st.text_area("识别结果", ocr_result, height=100)
                ocr_text = ocr_result
            else:
                st.error("OCR 识别失败，请检查配置或网络")

with col2:
    st.subheader("🔎 分析设置")
    context_half_lines = st.slider("每个匹配点前后上下文行数", 20, 500, 80)
    max_total_lines = st.slider("最大输出总行数", 100, 5000, 1500)

    analyze_btn = st.button("🚀 开始智能分析", type="primary", use_container_width=True)

if analyze_btn and uploaded_file and user_query:
    if not openai.api_key:
        st.error("请先在侧边栏输入 DeepSeek API Key")
    else:
        full_query = user_query
        if ocr_text:
            full_query = f"{user_query}\n\n[图片OCR识别内容]\n{ocr_text}"

        with st.spinner("正在检索相关日志..."):
            relevant_indices, match_count = find_relevant_context(log_lines, full_query, context_half_lines)
            
            if not relevant_indices:
                st.error(f"未找到包含 '{user_query}' 的日志行。请检查关键词是否正确。")
            else:
                st.success(f"✅ 在 {len(log_lines)} 行日志中找到 {match_count} 处匹配，提取了 {len(relevant_indices)} 行上下文。")
                
                if len(relevant_indices) > max_total_lines:
                    relevant_indices = relevant_indices[:max_total_lines]
                    st.warning(f"上下文行数超过限制，已截取前 {max_total_lines} 行。")

                context_lines = [log_lines[i] for i in relevant_indices]
                log_snippet = "\n".join(context_lines)

                # 提取错误码并在界面上显示定义（可选）
                extracted_codes = extract_error_codes_from_text(log_snippet)
                if extracted_codes and error_definitions:
                    with st.expander("📋 检测到的错误码及官方定义"):
                        for code in extracted_codes:
                            if code in error_definitions:
                                defn = error_definitions[code]
                                st.markdown(f"**{code}** (组件: {defn['component']})")
                                st.caption(f"用户提示: {defn['user_msg_cn']}")
                                st.caption(f"服务提示: {defn['service_msg']}")
                                st.caption(f"解决措施: {defn['service_action']}")
                            else:
                                st.markdown(f"**{code}** - 未找到定义")

                highlighted = []
                for line in context_lines:
                    if user_query.lower() in line.lower():
                        highlighted.append(f"<span style='background-color:#ffcccc'>{line}</span>")
                    else:
                        highlighted.append(line)
                log_highlight_html = "<br>".join(highlighted)

                st.subheader("📌 检索到的关键上下文")
                with st.expander("查看高亮日志", expanded=True):
                    st.markdown(f"<pre>{log_highlight_html}</pre>", unsafe_allow_html=True)

                similar_text = ""
                vectorizer, vectors, metadata = load_vectorizer_and_vectors()
                if vectorizer and len(metadata) > 0:
                    solutions = find_similar_solutions(full_query, vectorizer, vectors, metadata)
                    if solutions:
                        similar_text = "\n".join([f"- {sol}" for sol in solutions])

                with st.spinner("🤖 AI 正在深度分析..."):
                    analysis_result = generate_analysis(log_snippet, full_query, similar_text, error_definitions)

                st.subheader("📊 分析报告")
                st.markdown(analysis_result)

                st.session_state.analysis = analysis_result
                st.session_state.log_highlight = log_highlight_html.replace("<br>", "\n")

                st.divider()
                st.subheader("📸 导出报告")

                screenshot_html = render_screenshot_content(
                    analysis_result,
                    st.session_state.log_highlight
                )

                st.components.v1.html(f"""
                <html>
                <head>
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                </head>
                <body>
                    <div id="capture" style="display:none;">{screenshot_html}</div>
                    <button id="screenshot-btn" style="padding:10px 20px; background:#4CAF50; color:white; border:none; border-radius:5px; cursor:pointer;">
                        ⬇️ 下载分析报告截图
                    </button>
                    <script>
                        document.getElementById('screenshot-btn').addEventListener('click', function() {{
                            var element = document.getElementById('capture');
                            html2canvas(element, {{ scale: 2, backgroundColor: '#ffffff' }}).then(canvas => {{
                                var link = document.createElement('a');
                                link.download = 'log_analysis_report.png';
                                link.href = canvas.toDataURL();
                                link.click();
                            }});
                        }});
                    </script>
                </body>
                </html>
                """, height=80)

                st.divider()
                st.subheader("📝 提交解决方案 (帮助 AI 学习)")
                user_solution = st.text_area("如果问题已解决，请分享您的解决方案...")
                if st.button("提交解决方案"):
                    if user_solution:
                        kb = load_knowledge_base()
                        new_id = f"sol_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        kb.append({
                            "id": new_id,
                            "query": full_query,
                            "solution": user_solution,
                            "timestamp": datetime.now().isoformat()
                        })
                        save_knowledge_base(kb)
                        update_vectorizer_and_vectors(kb)
                        st.success("感谢反馈！知识库已更新，AI 将越来越智能。")
                        st.rerun()
                    else:
                        st.warning("请输入解决方案内容")

else:
    st.info("👆 请上传日志文件并输入问题描述（建议直接输入错误代码如 SSW_0x00000070），然后点击「开始智能分析」")

st.divider()
st.caption("Powered by Streamlit + DeepSeek + DashScope OCR + scikit-learn + 错误码定义库")