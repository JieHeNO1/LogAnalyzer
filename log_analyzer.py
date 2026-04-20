"""
智能日志分析助手 (集成错误码知识库 + AI 诊断)
修复了自动填充关键词时的 Streamlit API 冲突。
"""

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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import openai
import httpx
import chardet
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
st.set_page_config(page_title="智能日志分析助手", layout="wide")
st.title("🔍 智能日志分析助手 (错误码知识库 + AI 深度诊断)")

# ==================== 配置 ====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_OCR_MODEL = os.getenv("DASHSCOPE_OCR_MODEL", "qwen-vl-plus")

openai.api_key = DEEPSEEK_API_KEY
openai.api_base = DEEPSEEK_BASE_URL

KB_FILE = Path("knowledge_base.json")
VECTORIZER_FILE = Path("vectorizer.pkl")
VECTORS_FILE = Path("vectors.npy")
METADATA_FILE = Path("metadata.json")
ERROR_DEFS_DIR = Path("./error_defs")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# ==================== 错误码知识库 ====================
@dataclass
class ErrorCodeEntry:
    code: str
    code_hex: str
    user_prompt_cn: str
    user_prompt_en: str
    severity: str
    software_recoverable: bool
    upload_to_remote: bool
    service_error_info: str
    service_solution: str

class ComponentErrorCodeManager:
    ABBREVIATION_MAP = {
        'SSW': 'Software', 'SCU': 'SCU', 'GA': 'GA', 'RFA': 'RFA',
        'Couch': 'Couch', 'LCC': 'LCC', 'MMU': 'MMU', 'ACS': 'ACS',
        'HUB': 'HUB', 'GC': 'GC', 'VICP': 'VICP', 'DRTH': 'DeviceRoomTRH',
        'MRTH': 'MagnetRoomTRH', 'ECG': 'ECG', 'RESP': 'RESP', 'Finger': 'Finger'
    }

    def __init__(self, definitions_dir: str):
        self.definitions_dir = definitions_dir
        self.db: Dict[str, Dict[str, ErrorCodeEntry]] = {}
        self._load_all_files()

    def _extract_component_from_filename(self, filename: str) -> Optional[str]:
        base = os.path.basename(filename)
        name_without_ext = os.path.splitext(base)[0]
        parts = name_without_ext.split('_')
        return parts[0] if parts else None

    def _parse_bool_field(self, value: str) -> bool:
        if not value:
            return False
        return value.strip().lower() == 'yes'

    def _extract_hex_code(self, code_raw: str) -> str:
        match = re.search(r'(0x[0-9A-Fa-f]+)', code_raw)
        return match.group(1) if match else code_raw.strip()

    def _parse_file(self, filepath: str, component_full_name: str) -> List[ErrorCodeEntry]:
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='gbk') as f:
                lines = f.readlines()
        if not lines:
            return entries

        header_line = lines[0].strip()
        if header_line.startswith('\ufeff'):
            header_line = header_line[1:]
        headers = header_line.split('\t')
        headers = [h.strip() for h in headers]
        col_map = {name.lower(): idx for idx, name in enumerate(headers)}
        required_lower = ['code', '用户提示信息（医生、技师）', '英文', 'severity',
                          'software recoverable', '上传至远程平台', '服务提示报错信息', '服务提示解决措施']
        for col_lower in required_lower:
            if col_lower not in col_map:
                raise ValueError(f"缺少必要列: {col_lower}")

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            while len(parts) < len(headers):
                parts.append('')
            code_raw = parts[col_map['code']].strip()
            if not code_raw:
                continue
            code_hex = self._extract_hex_code(code_raw)
            entry = ErrorCodeEntry(
                code=code_raw, code_hex=code_hex,
                user_prompt_cn=parts[col_map['用户提示信息（医生、技师）']].strip(),
                user_prompt_en=parts[col_map['英文']].strip(),
                severity=parts[col_map['severity']].strip(),
                software_recoverable=self._parse_bool_field(parts[col_map['software recoverable']]),
                upload_to_remote=self._parse_bool_field(parts[col_map['上传至远程平台']]),
                service_error_info=parts[col_map['服务提示报错信息']].strip(),
                service_solution=parts[col_map['服务提示解决措施']].strip()
            )
            entries.append(entry)
        return entries

    def _load_all_files(self):
        if not os.path.isdir(self.definitions_dir):
            st.warning(f"目录不存在: {self.definitions_dir}")
            return
        for filename in os.listdir(self.definitions_dir):
            if not filename.lower().endswith('.txt'):
                continue
            comp = self._extract_component_from_filename(filename)
            if not comp:
                continue
            try:
                entries = self._parse_file(os.path.join(self.definitions_dir, filename), comp)
                if comp not in self.db:
                    self.db[comp] = {}
                for e in entries:
                    self.db[comp][e.code_hex] = e
                    if e.code != e.code_hex:
                        self.db[comp][e.code] = e
            except ValueError:
                pass
            except Exception as e:
                print(f"警告: 解析 {filename} 失败: {e}")

    def get_all_components(self) -> List[str]:
        return list(self.db.keys())

    def get_component_errors(self, component_full: str) -> Dict[str, ErrorCodeEntry]:
        for comp, entries in self.db.items():
            if comp.lower() == component_full.lower():
                return entries.copy()
        return {}

    def query_by_abbreviation(self, abbr: str, error_code: str) -> Optional[ErrorCodeEntry]:
        full = self.ABBREVIATION_MAP.get(abbr.upper())
        if not full:
            return None
        if full in self.db:
            if error_code in self.db[full]:
                return self.db[full][error_code]
            hex_code = self._extract_hex_code(error_code)
            if hex_code in self.db[full]:
                return self.db[full][hex_code]
        for comp, entries in self.db.items():
            if comp.lower() == full.lower():
                if error_code in entries:
                    return entries[error_code]
                hex_code = self._extract_hex_code(error_code)
                if hex_code in entries:
                    return entries[hex_code]
        return None

class ErrorDiagnosisAssistant:
    ERROR_PATTERN = re.compile(r'\b([A-Za-z]+)_(0x[0-9A-Fa-f]+)\b')
    HEX_CODE_PATTERN = re.compile(r'\b(0x[0-9A-Fa-f]+)\b')

    def __init__(self, manager: ComponentErrorCodeManager):
        self.manager = manager

    def extract_error_codes(self, text: str) -> List[str]:
        return [m.group(0) for m in self.ERROR_PATTERN.finditer(text)]

    def extract_hex_codes(self, text: str) -> List[str]:
        return [m.group(1) for m in self.HEX_CODE_PATTERN.finditer(text)]

    def extract_keywords(self, text: str, extra_keywords: List[str] = None) -> List[str]:
        keywords = []
        err_codes = self.extract_error_codes(text)
        keywords.extend(err_codes)
        existing_hex = set(re.findall(r'0x[0-9A-Fa-f]+', ' '.join(err_codes)))
        for hc in self.extract_hex_codes(text):
            if hc not in existing_hex:
                keywords.append(hc)
        if extra_keywords:
            for kw in extra_keywords:
                kw = kw.strip()
                if kw and kw not in keywords:
                    keywords.append(kw)
        seen = set()
        unique = []
        for kw in keywords:
            low = kw.lower()
            if low not in seen:
                seen.add(low)
                unique.append(kw)
        return unique

    def parse_and_query(self, text: str) -> List[Tuple[str, str, Optional[ErrorCodeEntry]]]:
        res = []
        for m in self.ERROR_PATTERN.finditer(text):
            abbr, code = m.group(1), m.group(2)
            entry = self.manager.query_by_abbreviation(abbr, code)
            res.append((abbr, code, entry))
        return res

@st.cache_resource
def init_error_manager():
    return ComponentErrorCodeManager(str(ERROR_DEFS_DIR))

error_manager = init_error_manager()
error_assistant = ErrorDiagnosisAssistant(error_manager)

# ==================== 知识库操作 ====================
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
    metadata = [{"id": it["id"], "query": it["query"], "solution": it["solution"]} for it in kb]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return vectorizer, vectors

def load_vectorizer_and_vectors():
    if VECTORIZER_FILE.exists() and VECTORS_FILE.exists() and METADATA_FILE.exists():
        with open(VECTORIZER_FILE, "rb") as f:
            vectorizer = pickle.load(f)
        vectors = np.load(VECTORS_FILE)
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        return vectorizer, vectors, metadata
    return None, None, []

def find_similar_solutions(query, vectorizer, vectors, metadata, top_k=2):
    if vectorizer is None or vectors is None or not metadata:
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, vectors).flatten()
    idxs = sims.argsort()[-top_k:][::-1]
    return [metadata[i]["solution"] for i in idxs if sims[i] > 0.1]

# ==================== OCR ====================
def ocr_image_dashscope(image_bytes):
    if not DASHSCOPE_API_KEY:
        return None
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": DASHSCOPE_OCR_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "请识别并提取图片中的所有文字内容，不要添加额外说明。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}],
            "max_tokens": 1000
        }
        resp = requests.post(f"{DASHSCOPE_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.warning(f"DashScope OCR 异常: {e}")
    return None

def ocr_image_tesseract(image_bytes):
    if not TESSERACT_AVAILABLE:
        return None
    try:
        img = Image.open(BytesIO(image_bytes))
        return pytesseract.image_to_string(img, lang='eng+chi_sim').strip()
    except Exception as e:
        st.warning(f"Tesseract OCR 异常: {e}")
    return None

def perform_ocr(uploaded_file):
    img_bytes = uploaded_file.read()
    text = ocr_image_dashscope(img_bytes)
    if text is not None:
        return text, "DashScope"
    text = ocr_image_tesseract(img_bytes)
    if text is not None:
        return text, "Tesseract"
    return None, "OCR 失败"

# ==================== 日志检索 ====================
def find_relevant_context(log_lines, keywords: List[str], context_lines=80):
    if not log_lines or not keywords:
        return [], 0
    matched = []
    kw_lower = [k.lower() for k in keywords]
    for i, line in enumerate(log_lines):
        if any(k in line.lower() for k in kw_lower):
            matched.append(i)
    if not matched:
        return [], 0
    included = set()
    for idx in matched:
        start = max(0, idx - context_lines)
        end = min(len(log_lines), idx + context_lines + 1)
        included.update(range(start, end))
    return sorted(included), len(matched)

def generate_analysis(log_snippet, user_query, similar_cases="", error_kb_info=""):
    prompt = f"""你是一名资深的MRI系统软硬件日志分析专家。请根据以下信息进行分析。

【用户问题描述】
{user_query}

【本地错误码知识库信息】
{error_kb_info or "无"}

【相关日志片段】
{log_snippet or "（未检索到相关日志）"}

【历史相似案例】
{similar_cases or "无"}

【分析要求】
1. 错误直接原因
2. 故障链追溯
3. 潜在根本原因（3个，按可能性排序）
4. 排查建议

请以结构化Markdown格式输出。"""
    try:
        proxy = os.getenv("HTTP_PROXY")
        http_client = httpx.Client(proxy=proxy) if proxy else None
        client = openai.OpenAI(api_key=openai.api_key, base_url=openai.api_base, http_client=http_client)
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": "你是一个专业的MRI日志分析专家。"},
                      {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2000
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI 分析失败: {e}"

def render_screenshot_content(analysis_md, log_highlight):
    return f"""
    <div id="screenshot-area" style="background:white; padding:20px; font-family:Arial; max-width:900px;">
        <h2>📋 日志分析报告</h2>
        <div style="margin:20px 0; line-height:1.6;">{analysis_md.replace(chr(10), '<br>')}</div>
        <h3>📄 关键日志上下文</h3>
        <pre style="background:#f5f5f5; padding:15px; border-radius:5px; overflow-x:auto; white-space:pre-wrap;">{log_highlight}</pre>
    </div>
    """

# ==================== 界面与交互逻辑 ====================
if "auto_fill_done" not in st.session_state:
    st.session_state.auto_fill_done = False
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False
if "extracted_kws" not in st.session_state:
    st.session_state.extracted_kws = []
if "user_query_full" not in st.session_state:
    st.session_state.user_query_full = ""
# 用于自动填充的字符串变量（独立于输入框的 key）
if "auto_keywords_str" not in st.session_state:
    st.session_state.auto_keywords_str = ""

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
    st.metric("用户反馈方案", len(kb))
    st.metric("错误码部件", len(error_manager.get_all_components()))

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 上传日志文件")
    uploaded_file = st.file_uploader("选择 .log 或 .txt", type=["log", "txt"])
    log_lines = []
    if uploaded_file:
        raw = uploaded_file.read()
        enc = chardet.detect(raw)['encoding'] or 'gbk'
        log_content = raw.decode(enc, errors='ignore')
        log_lines = log_content.splitlines()
        st.success(f"已加载 {len(log_lines)} 行日志")
        with st.expander("预览前100行"):
            st.text("\n".join(log_lines[:100]))

    st.subheader("❓ 问题描述")
    user_query = st.text_area("描述问题（支持 SSW_0x00000070）", height=120, placeholder="例如：频率校正信噪比异常(SSW_0x00000070)")

    st.subheader("🖼️ 图片上传 (可选)")
    uploaded_image = st.file_uploader("支持 PNG/JPG", type=["png", "jpg", "jpeg"])
    ocr_text = ""
    if uploaded_image:
        with st.spinner("OCR 识别中..."):
            ocr_result, method = perform_ocr(uploaded_image)
            if ocr_result:
                st.success(f"OCR 成功 ({method})")
                st.text_area("识别结果", ocr_result, height=80)
                ocr_text = ocr_result
            else:
                st.error("OCR 失败")

with col2:
    st.subheader("🔎 分析设置")
    context_lines = st.slider("上下文行数", 20, 500, 80)
    max_lines = st.slider("最大输出行数", 100, 5000, 1500)
    # 关键修改：value 绑定到 st.session_state.auto_keywords_str
    extra_keywords_input = st.text_input(
        "附加检索关键词 (用逗号分隔，可选)",
        key="extra_keywords_input",
        value=st.session_state.auto_keywords_str,
        help="可补充其他关键词，如 '频率校正'、'信噪比' 等。点击分析后会自动填充智能提取的关键词。"
    )
    analyze_btn = st.button("🚀 开始智能分析", type="primary", use_container_width=True)

def perform_analysis():
    full_query = st.session_state.user_query_full
    # 1. 错误码知识库查询
    error_kb_parts = []
    parsed = error_assistant.parse_and_query(full_query)
    if parsed:
        st.subheader("📖 错误码知识库匹配")
        for abbr, code, entry in parsed:
            if entry:
                full_name = error_manager.ABBREVIATION_MAP.get(abbr.upper(), abbr)
                st.markdown(f"**`{abbr}_{code}`** → 部件 `{full_name}`  \n- 严重程度: {entry.severity}  \n- 解决措施: {entry.service_solution}")
                error_kb_parts.append(
                    f"错误码 {abbr}_{code} (部件{full_name})：\n严重程度: {entry.severity}\n解决措施: {entry.service_solution}\n"
                )
            else:
                st.warning(f"未找到错误码 {abbr}_{code} 的定义")
        st.divider()
    error_kb_info = "\n".join(error_kb_parts)

    keywords = st.session_state.extracted_kws
    if not keywords:
        extra_kws = [k.strip() for k in extra_keywords_input.split(',') if k.strip()] if extra_keywords_input else []
        keywords = error_assistant.extract_keywords(full_query, extra_keywords=extra_kws)
        st.session_state.extracted_kws = keywords

    st.markdown("**🔑 使用的检索关键词:**")
    st.write(", ".join(keywords) if keywords else "（未提取到关键词，将使用原始描述子串匹配）")

    with st.spinner("检索日志中..."):
        indices, match_cnt = find_relevant_context(log_lines, keywords, context_lines)
        if not indices:
            st.warning("未在日志中找到相关行，将仅基于错误码定义和描述进行分析。")
            log_snippet = ""
            log_highlight_html = "（无相关日志）"
        else:
            st.success(f"找到 {match_cnt} 处匹配，提取 {len(indices)} 行上下文。")
            if len(indices) > max_lines:
                indices = indices[:max_lines]
            context_lines_list = [log_lines[i] for i in indices]
            log_snippet = "\n".join(context_lines_list)
            kw_lower = [k.lower() for k in keywords]
            highlighted = []
            for line in context_lines_list:
                if any(k in line.lower() for k in kw_lower):
                    highlighted.append(f"<span style='background-color:#ffcccc'>{line}</span>")
                else:
                    highlighted.append(line)
            log_highlight_html = "<br>".join(highlighted)
            with st.expander("查看高亮日志", expanded=True):
                st.markdown(f"<pre>{log_highlight_html}</pre>", unsafe_allow_html=True)

    similar_text = ""
    vec, vcts, meta = load_vectorizer_and_vectors()
    if vec and meta:
        sols = find_similar_solutions(full_query, vec, vcts, meta)
        if sols:
            similar_text = "\n".join(f"- {s}" for s in sols)

    with st.spinner("🤖 AI 分析中..."):
        analysis_result = generate_analysis(log_snippet, full_query, similar_text, error_kb_info)

    st.subheader("📊 综合分析报告")
    st.markdown(analysis_result)

    st.divider()
    st.subheader("📸 导出报告")
    html_str = render_screenshot_content(analysis_result, log_highlight_html.replace("<br>", "\n") if 'log_highlight_html' in locals() else "无")
    st.components.v1.html(f"""
    <html><head><script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script></head>
    <body>
        <div id="capture" style="display:none;">{html_str}</div>
        <button id="btn" style="padding:10px 20px; background:#4CAF50; color:white; border:none; border-radius:5px;">⬇️ 下载截图</button>
        <script>
            document.getElementById('btn').addEventListener('click', function(){{
                html2canvas(document.getElementById('capture'), {{ scale: 2, backgroundColor: '#ffffff' }}).then(canvas => {{
                    var link = document.createElement('a');
                    link.download = 'log_analysis_report.png';
                    link.href = canvas.toDataURL();
                    link.click();
                }});
            }});
        </script>
    </body></html>
    """, height=80)

    st.divider()
    st.subheader("📝 提交解决方案")
    user_solution = st.text_area("分享您的解决方案...")
    if st.button("提交"):
        if user_solution:
            kb = load_knowledge_base()
            new_id = f"sol_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            kb.append({"id": new_id, "query": full_query, "solution": user_solution, "timestamp": datetime.now().isoformat()})
            save_knowledge_base(kb)
            update_vectorizer_and_vectors(kb)
            st.success("感谢反馈！")
            st.rerun()
        else:
            st.warning("请输入内容")

    st.session_state.run_analysis = False

if analyze_btn and uploaded_file and user_query:
    if not openai.api_key:
        st.error("请先输入 DeepSeek API Key")
    else:
        full_q = user_query
        if ocr_text:
            full_q = f"{user_query}\n\n[图片OCR内容]\n{ocr_text}"
        st.session_state.user_query_full = full_q

        if not st.session_state.auto_fill_done:
            extra_kws = [k.strip() for k in extra_keywords_input.split(',') if k.strip()] if extra_keywords_input else []
            keywords = error_assistant.extract_keywords(full_q, extra_keywords=extra_kws)
            st.session_state.extracted_kws = keywords
            # 修改 auto_keywords_str 而不是直接修改 extra_keywords_input
            st.session_state.auto_keywords_str = ", ".join(keywords)
            st.session_state.auto_fill_done = True
            st.session_state.run_analysis = True
            st.rerun()
        else:
            st.session_state.run_analysis = True

if st.session_state.get("run_analysis", False) and uploaded_file and log_lines:
    perform_analysis()
    st.session_state.auto_fill_done = False
    # 分析完成后可清空自动填充字符串，以便下次分析可以重新填充（可选）
    st.session_state.auto_keywords_str = ""
elif not uploaded_file and analyze_btn:
    st.info("👆 请上传日志文件")

st.divider()
st.caption("Powered by Streamlit + DeepSeek + 本地错误码知识库")