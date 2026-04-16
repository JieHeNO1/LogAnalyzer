import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ErrorCodeEntry:
    """单个错误码条目"""
    code: str                # 错误码，如 0x20000035
    user_prompt_cn: str      # 用户提示信息（中文）
    user_prompt_en: str      # 用户提示信息（英文）
    severity: str            # 严重程度：Warning / Error / Info
    software_recoverable: bool  # 是否软件可恢复
    upload_to_remote: bool      # 是否上传至远程平台
    service_error_info: str     # 服务提示报错信息
    service_solution: str       # 服务提示解决措施

class ComponentErrorCodeManager:
    """
    多部件错误码管理器
    加载指定目录下所有符合格式的部件错误码定义文件（.txt）
    文件命名规范示例：{Component}_{Model/Version}.txt 或 {Component}.txt
    """

    # 部件缩写到全称的映射表（基于用户提供表格）
    ABBREVIATION_MAP = {
        'SSW': 'Software',
        'SCU': 'SCU',
        'GA': 'GA',
        'RFA': 'RFA',
        'Couch': 'Couch',
        'LCC': 'LCC',
        'MMU': 'MMU',
        'ACS': 'ACS',
        'HUB': 'HUB',
        'GC': 'GC',
        'VICP': 'VICP',
        'DRTH': 'DeviceRoomTRH',
        'MRTH': 'MagnetRoomTRH',
        'ECG': 'ECG',
        'RESP': 'RESP',
        'Finger': 'Finger'
    }

    def __init__(self, definitions_dir: str):
        """
        :param definitions_dir: 存放错误码定义文件的目录路径
        """
        self.definitions_dir = definitions_dir
        # 数据结构：{component_full_name: {error_code: ErrorCodeEntry}}
        self.db: Dict[str, Dict[str, ErrorCodeEntry]] = {}
        self._load_all_files()

    def _extract_component_from_filename(self, filename: str) -> Optional[str]:
        """
        从文件名提取部件全称，如 ACS_80E2P3.txt -> ACS
        也支持 Software.txt -> Software
        """
        base = os.path.basename(filename)
        name_without_ext = os.path.splitext(base)[0]
        # 使用下划线分割，取第一部分作为部件代码
        parts = name_without_ext.split('_')
        if parts:
            return parts[0]  # 保持原始大小写
        return None

    def _parse_bool_field(self, value: str) -> bool:
        """解析Yes/No字段为布尔值"""
        if not value:
            return False
        return value.strip().lower() == 'yes'

    def _parse_file(self, filepath: str, component_full_name: str) -> List[ErrorCodeEntry]:
        """解析单个文件，返回条目列表"""
        entries = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='gbk') as f:
                lines = f.readlines()

        if not lines:
            return entries

        header_line = lines[0].strip()
        headers = header_line.split('\t')
        headers = [h.strip() for h in headers]

        col_map = {name: idx for idx, name in enumerate(headers)}

        required_cols = ['Code', '用户提示信息（医生、技师）', '英文',
                         'Severity', 'Software Recoverable', '上传至远程平台',
                         '服务提示报错信息', '服务提示解决措施']
        for col in required_cols:
            if col not in col_map:
                raise ValueError(f"文件 {filepath} 缺少必要列: {col}")

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            while len(parts) < len(headers):
                parts.append('')

            code = parts[col_map['Code']].strip()
            if not code:
                continue

            user_prompt_cn = parts[col_map['用户提示信息（医生、技师）']].strip()
            user_prompt_en = parts[col_map['英文']].strip()
            severity = parts[col_map['Severity']].strip()
            sw_recoverable = self._parse_bool_field(parts[col_map['Software Recoverable']])
            upload_remote = self._parse_bool_field(parts[col_map['上传至远程平台']])
            service_info = parts[col_map['服务提示报错信息']].strip()
            service_solution = parts[col_map['服务提示解决措施']].strip()

            entry = ErrorCodeEntry(
                code=code,
                user_prompt_cn=user_prompt_cn,
                user_prompt_en=user_prompt_en,
                severity=severity,
                software_recoverable=sw_recoverable,
                upload_to_remote=upload_remote,
                service_error_info=service_info,
                service_solution=service_solution
            )
            entries.append(entry)

        return entries

    def _load_all_files(self):
        """加载目录下所有匹配的错误码文件，按部件全称索引"""
        if not os.path.isdir(self.definitions_dir):
            raise ValueError(f"目录不存在: {self.definitions_dir}")

        txt_files = [f for f in os.listdir(self.definitions_dir)
                     if f.lower().endswith('.txt')]

        for filename in txt_files:
            component_full = self._extract_component_from_filename(filename)
            if not component_full:
                continue
            filepath = os.path.join(self.definitions_dir, filename)
            try:
                entries = self._parse_file(filepath, component_full)
                if component_full not in self.db:
                    self.db[component_full] = {}
                for entry in entries:
                    self.db[component_full][entry.code] = entry
            except Exception as e:
                print(f"警告: 解析文件 {filename} 失败: {e}")

    def query_by_full_name(self, component_full: str, error_code: str) -> Optional[ErrorCodeEntry]:
        """
        根据部件全称和错误码查询
        :param component_full: 部件全称，如 'ACS', 'Software'
        :param error_code: 错误码，如 '0x20000035'
        """
        # 尝试直接匹配（区分大小写）
        if component_full in self.db:
            return self.db[component_full].get(error_code)
        # 尝试忽略大小写匹配
        for comp, entries in self.db.items():
            if comp.lower() == component_full.lower():
                return entries.get(error_code)
        return None

    def query_by_abbreviation(self, abbr: str, error_code: str) -> Optional[ErrorCodeEntry]:
        """
        根据部件缩写和错误码查询
        """
        full_name = self.ABBREVIATION_MAP.get(abbr.upper())
        if not full_name:
            print(f"警告: 未知的部件缩写 '{abbr}'")
            return None
        return self.query_by_full_name(full_name, error_code)

    def get_all_components(self) -> List[str]:
        return list(self.db.keys())

    def get_component_errors(self, component_full: str) -> Dict[str, ErrorCodeEntry]:
        for comp, entries in self.db.items():
            if comp.lower() == component_full.lower():
                return entries.copy()
        return {}

    def format_for_user(self, component: str, error_code: str, lang: str = 'cn') -> str:
        """格式化输出（支持全称或缩写）"""
        entry = None
        # 尝试作为全称查询
        entry = self.query_by_full_name(component, error_code)
        if not entry:
            # 尝试作为缩写查询
            entry = self.query_by_abbreviation(component, error_code)

        if not entry:
            if lang == 'cn':
                return f"未找到错误码 {error_code} (部件: {component}) 的定义。"
            else:
                return f"Error code {error_code} (Component: {component}) not found."

        if lang == 'cn':
            prompt = entry.user_prompt_cn
            severity = entry.severity
            solution = entry.service_solution.replace('\n', '\n  ')
            return f"[{component}] {error_code} - {severity}\n用户提示: {prompt}\n解决措施: {solution}"
        else:
            prompt = entry.user_prompt_en
            severity = entry.severity
            solution = entry.service_solution.replace('\n', '\n  ')
            return f"[{component}] {error_code} - {severity}\nUser prompt: {prompt}\nSolution: {solution}"


class ErrorDiagnosisAssistant:
    """
    错误诊断助手：集成错误码管理器，并加入智能分析能力
    """
    # 正则模式：匹配类似 SSW_0x00000070 的标识
    ERROR_PATTERN = re.compile(r'\b([A-Za-z]+)_(0x[0-9A-Fa-f]+)\b')

    def __init__(self, definitions_dir: str):
        self.manager = ComponentErrorCodeManager(definitions_dir)

    def parse_error_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """
        从文本中提取所有 (部件缩写, 错误码) 对
        返回列表，每个元素为 (缩写, 错误码, 匹配到的完整字符串)
        """
        matches = []
        for match in self.ERROR_PATTERN.finditer(text):
            abbr = match.group(1)
            code = match.group(2)
            full_match = match.group(0)
            matches.append((abbr, code, full_match))
        return matches

    def diagnose_text(self, text: str, lang: str = 'cn') -> str:
        """
        对输入文本进行错误诊断，返回综合分析结论
        """
        errors = self.parse_error_from_text(text)
        if not errors:
            if lang == 'cn':
                return "未检测到任何错误标识（格式如：SSW_0x00000070）。"
            else:
                return "No error identifier detected (format: SSW_0x00000070)."

        results = []
        for abbr, code, full_match in errors:
            entry = self.manager.query_by_abbreviation(abbr, code)
            if entry:
                # 获取基本信息
                if lang == 'cn':
                    prompt = entry.user_prompt_cn
                else:
                    prompt = entry.user_prompt_en
                severity = entry.severity
                solution = entry.service_solution
                service_info = entry.service_error_info

                # 调用 AI 分析函数（此处为模拟）
                ai_analysis = self._ai_analyze(entry, lang)

                result_block = f"""
========================================
错误标识: {full_match}
部件全称: {self.manager.ABBREVIATION_MAP.get(abbr.upper(), '未知')}
错误码: {code}
严重程度: {severity}
用户提示: {prompt}
服务信息: {service_info}
解决措施: {solution}

【AI 智能分析】
{ai_analysis}
========================================
"""
                results.append(result_block.strip())
            else:
                if lang == 'cn':
                    results.append(f"错误标识 {full_match} 未找到对应的定义信息，请检查部件缩写或错误码是否正确。")
                else:
                    results.append(f"Error identifier {full_match} not found in definition files.")

        return "\n\n".join(results)

    def _ai_analyze(self, entry: ErrorCodeEntry, lang: str = 'cn') -> str:
        """
        AI 智能分析函数（模拟实现）
        实际使用时，您可以将此函数替换为调用真实的 AI 服务（如 OpenAI、本地模型等）
        返回分析建议文本
        """
        severity = entry.severity
        recoverable = entry.software_recoverable
        solution = entry.service_solution

        # 基于规则的简单分析，可替换为 LLM 生成
        if lang == 'cn':
            if severity == 'Error':
                if recoverable:
                    analysis = "这是一个严重错误，但系统可能能够自动恢复。建议首先观察系统是否自行恢复正常，若持续出现则需人工介入。"
                else:
                    analysis = "这是一个严重错误且无法软件恢复，需要立即处理。请按照解决措施中的步骤排查，必要时联系服务工程师。"
            elif severity == 'Warning':
                analysis = "这是一个警告级别的问题，暂不影响系统核心功能，但需留意并尽快排查。"
            else:
                analysis = "错误级别未明确，建议按照解决措施进行处理。"

            # 附加基于解决措施的建议
            if "检查" in solution or "check" in solution.lower():
                analysis += " 解决措施中涉及多项检查，请逐一核实。"
            if "更换" in solution or "replace" in solution.lower():
                analysis += " 可能需要更换硬件部件，请提前准备好备件。"
        else:
            if severity == 'Error':
                if recoverable:
                    analysis = "This is a critical error, but the system may attempt auto-recovery. Monitor if the issue resolves itself; if persistent, manual intervention is required."
                else:
                    analysis = "This is a critical error and cannot be recovered by software. Immediate action is required. Follow the solution steps and contact service engineer if needed."
            elif severity == 'Warning':
                analysis = "This is a warning issue. It may not affect core functionality immediately, but should be investigated soon."
            else:
                analysis = "Error severity unspecified. Please follow the provided solution."

            if "check" in solution.lower():
                analysis += " The solution involves multiple checks; please verify each step."
            if "replace" in solution.lower():
                analysis += " Hardware replacement may be required; please have spare parts ready."

        return analysis


# ------------------------- 使用示例 -------------------------
if __name__ == '__main__':
    # 初始化助手，指定定义文件目录
    assistant = ErrorDiagnosisAssistant(definitions_dir='./error_defs')

    # 模拟用户输入的一段日志或描述
    user_input = """
    系统在扫描过程中出现异常，报错信息为 SSW_0x20000035，同时 ACS_0x2000003B 也有记录。
    请帮忙分析原因。
    """

    print("=== 智能诊断结果 ===\n")
    diagnosis_report = assistant.diagnose_text(user_input, lang='cn')
    print(diagnosis_report)

    # 也可以单独查询某个错误码
    print("\n--- 单独查询 ACS_0x2000003B ---")
    print(assistant.manager.format_for_user('ACS', '0x2000003B', 'cn'))