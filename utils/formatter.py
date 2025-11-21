from config import Config, logger
from storage.models import Asset
from decorators.decorators import cpu_bound

from typing import Dict, List, Tuple, Any, Match
import re

import telegramify_markdown
from telegramify_markdown.customize import get_runtime_config
from telegramify_markdown.render import escape_markdown


EXTENSION_TO_TELEGRAM_LANG = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "java": "java",
    "kt": "kotlin",
    "kts": "kotlin",
    "swift": "swift",
    "c": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "h": "c",
    "hpp": "cpp",
    "cs": "csharp",
    "php": "php",
    "rb": "ruby",
    "go": "go",
    "rs": "rust",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "html": "html",
    "htm": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "json": "json",
    "xml": "xml",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "sql": "sql",
    "md": "markdown",
    "markdown": "markdown",
    "tex": "latex",
    "bat": "batch",
    "ps1": "powershell",
    "dockerfile": "dockerfile",
    "r": "r",
    "lua": "lua",
    "perl": "perl",
    "pl": "perl",
}


def map_extension_to_lang(ext: str) -> str:
    """Map file extension to Telegram language identifier"""
    return EXTENSION_TO_TELEGRAM_LANG.get(ext.lower(), ext.lower())


class MessageFormatter:
    """Handle message formatting for Telegram using telegramify-markdown"""

    def __init__(self) -> None:
        # Configure telegramify-markdown for clean output
        get_runtime_config().markdown_symbol.head_level_1 = ""
        get_runtime_config().markdown_symbol.link = ""

    def _latex_to_unicode(self, latex: str) -> str:
        """Convert LaTeX to Unicode - comprehensive conversion"""
        latex = latex.strip()

        # Extended conversions map
        conversions = {
            # Greek letters (lowercase)
            r"\alpha": "α",
            r"\beta": "β",
            r"\gamma": "γ",
            r"\delta": "δ",
            r"\epsilon": "ε",
            r"\zeta": "ζ",
            r"\eta": "η",
            r"\theta": "θ",
            r"\iota": "ι",
            r"\kappa": "κ",
            r"\lambda": "λ",
            r"\mu": "μ",
            r"\nu": "ν",
            r"\xi": "ξ",
            r"\pi": "π",
            r"\rho": "ρ",
            r"\sigma": "σ",
            r"\tau": "τ",
            r"\upsilon": "υ",
            r"\phi": "φ",
            r"\chi": "χ",
            r"\psi": "ψ",
            r"\omega": "ω",
            # Greek letters (uppercase)
            r"\Gamma": "Γ",
            r"\Delta": "Δ",
            r"\Theta": "Θ",
            r"\Lambda": "Λ",
            r"\Xi": "Ξ",
            r"\Pi": "Π",
            r"\Sigma": "Σ",
            r"\Upsilon": "Υ",
            r"\Phi": "Φ",
            r"\Psi": "Ψ",
            r"\Omega": "Ω",
            # Arrows
            r"\rightarrow": "→",
            r"\leftarrow": "←",
            r"\uparrow": "↑",
            r"\downarrow": "↓",
            r"\Rightarrow": "⇒",
            r"\Leftarrow": "⇐",
            r"\Uparrow": "⇑",
            r"\Downarrow": "⇓",
            r"\leftrightarrow": "↔",
            r"\Leftrightarrow": "⇔",
            # Math operators
            r"\partial": "∂",
            r"\infty": "∞",
            r"\pm": "±",
            r"\mp": "∓",
            r"\times": "×",
            r"\div": "÷",
            r"\cdot": "·",
            r"\ast": "∗",
            r"\leq": "≤",
            r"\geq": "≥",
            r"\neq": "≠",
            r"\approx": "≈",
            r"\equiv": "≡",
            r"\sim": "∼",
            r"\simeq": "≃",
            r"\propto": "∝",
            # Set theory
            r"\subset": "⊂",
            r"\supset": "⊃",
            r"\subseteq": "⊆",
            r"\supseteq": "⊇",
            r"\in": "∈",
            r"\notin": "∉",
            r"\ni": "∋",
            r"\emptyset": "∅",
            r"\cup": "∪",
            r"\cap": "∩",
            # Logic
            r"\forall": "∀",
            r"\exists": "∃",
            r"\nexists": "∄",
            r"\therefore": "∴",
            r"\because": "∵",
            r"\land": "∧",
            r"\lor": "∨",
            r"\lnot": "¬",
            r"\neg": "¬",
            # Calculus
            r"\nabla": "∇",
            r"\sum": "∑",
            r"\prod": "∏",
            r"\int": "∫",
            r"\oint": "∮",
            r"\iint": "∬",
            r"\iiint": "∭",
            # Other
            r"\sqrt": "√",
            r"\angle": "∠",
            r"\perp": "⊥",
            r"\parallel": "∥",
            r"\degree": "°",
            r"\prime": "′",
            r"\hbar": "ℏ",
            r"\ell": "ℓ",
            r"\Re": "ℜ",
            r"\Im": "ℑ",
            r"\aleph": "ℵ",
        }

        # Apply symbol conversions
        for latex_cmd, unicode_char in conversions.items():
            latex = latex.replace(latex_cmd, unicode_char)

        # Handle fractions
        latex = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", latex)

        # Superscripts mapping
        superscript_map = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
            "+": "⁺",
            "-": "⁻",
            "=": "⁼",
            "(": "⁽",
            ")": "⁾",
            "n": "ⁿ",
            "i": "ⁱ",
        }

        # Subscripts mapping
        subscript_map = {
            "0": "₀",
            "1": "₁",
            "2": "₂",
            "3": "₃",
            "4": "₄",
            "5": "₅",
            "6": "₆",
            "7": "₇",
            "8": "₈",
            "9": "₉",
            "+": "₊",
            "-": "₋",
            "=": "₌",
            "(": "₍",
            ")": "₎",
            "a": "ₐ",
            "e": "ₑ",
            "i": "ᵢ",
            "o": "ₒ",
            "r": "ᵣ",
            "u": "ᵤ",
            "v": "ᵥ",
            "x": "ₓ",
        }

        # Process superscripts ^{...} or ^x
        def replace_superscript(match: Match[str]) -> str:
            text = match.group(1) if match.group(1) else match.group(2)
            result = ""
            for char in text:
                result += superscript_map.get(char, char)
            return result

        latex = re.sub(r"\^{([^}]+)}|\^([a-zA-Z0-9])", replace_superscript, latex)

        # Process subscripts _{...} or _x
        def replace_subscript(match: Match[str]) -> str:
            text = match.group(1) if match.group(1) else match.group(2)
            result = ""
            for char in text:
                result += subscript_map.get(char, char)
            return result

        latex = re.sub(r"_{([^}]+)}|_([a-zA-Z0-9])", replace_subscript, latex)

        # Clean up remaining LaTeX commands
        latex = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", latex)
        latex = re.sub(r"\\[a-zA-Z]+", "", latex)

        # Clean up braces and extra spaces
        latex = latex.replace("{", "").replace("}", "")
        latex = re.sub(r"\s+", " ", latex).strip()

        return latex

    @cpu_bound
    def _preprocess_latex_in_markdown(self, text: str) -> str:
        """Convert LaTeX to Unicode in markdown text while preserving structure"""
        code_blocks: List[str] = []
        code_pattern = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

        def store_code(match: Match[str]) -> str:
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{idx}__"

        text = code_pattern.sub(store_code, text)

        # Process display math ($$...$$)
        def replace_display(match: Match[str]) -> str:
            latex = self._latex_to_unicode(match.group(1))
            return f"**{latex}**"

        text = re.sub(r"\$\$(.*?)\$\$", replace_display, text, flags=re.DOTALL)

        # Process inline math ($...$)
        def replace_inline(match: Match[str]) -> str:
            latex = self._latex_to_unicode(match.group(1))
            return f"_{latex}_"

        text = re.sub(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", replace_inline, text)

        for idx, code_block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{idx}__", code_block)

        return text

    async def _extract_content_structure(self, raw_text: str) -> List[Any]:
        """Pre-process LaTeX then extract structure using telegramify"""
        processed_text = await self._preprocess_latex_in_markdown(raw_text)

        boxes = await telegramify_markdown.telegramify(
            content=processed_text,
            max_word_count=Config.SAFE_MESSAGE_LENGTH,
            latex_escape=False,
            normalize_whitespace=True,
        )
        return boxes if isinstance(boxes, list) else [boxes]

    def _process_text_box(self, box: telegramify_markdown.type.Text) -> str:
        """Process Text box - content is already escaped by telegramify"""
        return str(box.content)

    def _escape_code_content(self, code: str) -> str:
        """Properly escape code content for MarkdownV2 code blocks"""
        # Inside code blocks, only backticks need escaping
        return code.replace("`", "\\`")

    def _create_code_block(self, code: str, lang: str) -> str:
        """Create a properly formatted code block for Telegram MarkdownV2"""
        escaped_code = self._escape_code_content(code)
        return f"```{lang}\n{escaped_code}\n```"

    @cpu_bound
    def _process_file_box(
        self, box: telegramify_markdown.type.File, file_type_counter: Dict[str, int]
    ) -> Tuple[Asset, List[str]]:
        """Process File box"""

        ext = (
            box.file_name[box.file_name.rfind(".") + 1 :]
            if "." in box.file_name
            else "txt"
        )

        if ext not in file_type_counter:
            file_type_counter[ext] = 0
        file_type_counter[ext] += 1

        telegram_lang = map_extension_to_lang(ext)
        generated_filename = f"{telegram_lang}_code_{file_type_counter[ext]}.{ext}"

        asset = Asset(
            asset_id=f"asset_{sum(file_type_counter.values())}",
            file_name=generated_filename,
            file_data=box.file_data,
            language=telegram_lang,
            size=len(box.file_data),
        )

        code_str = box.file_data.decode("utf-8", errors="replace")

        fence_overhead = len(f"```{telegram_lang}\n\n```")
        max_code_size = Config.MAX_MESSAGE_LENGTH - fence_overhead - 100

        wrapped_parts = []

        if len(code_str) <= max_code_size:
            wrapped = self._create_code_block(code_str, telegram_lang)
            wrapped_parts.append(wrapped)
        else:
            logger.info(
                f"Code ({len(code_str)} chars) exceeds {max_code_size}, splitting..."
            )

            lines = code_str.split("\n")
            current_chunk_lines: List[str] = []
            current_chunk_size = 0

            for line in lines:
                line_size = len(line) + 1

                if (
                    current_chunk_size + line_size > max_code_size
                    and current_chunk_lines
                ):
                    chunk_content = "\n".join(current_chunk_lines)
                    wrapped_chunk = self._create_code_block(
                        chunk_content, telegram_lang
                    )
                    wrapped_parts.append(wrapped_chunk)

                    current_chunk_lines = [line]
                    current_chunk_size = line_size
                else:
                    current_chunk_lines.append(line)
                    current_chunk_size += line_size

            if current_chunk_lines:
                chunk_content = "\n".join(current_chunk_lines)
                wrapped_chunk = self._create_code_block(chunk_content, telegram_lang)
                wrapped_parts.append(wrapped_chunk)

        return (asset, wrapped_parts)

    async def _assemble_message_parts(
        self, boxes: List[Any]
    ) -> Tuple[List[str], List[Asset]]:
        """Process all boxes"""
        string_parts = []
        assets = []
        file_type_counter: Dict[str, int] = {}

        for box in boxes:
            if isinstance(box, telegramify_markdown.type.Text):
                text = self._process_text_box(box)
                if text:
                    string_parts.append(text)

            elif isinstance(box, telegramify_markdown.type.File):
                asset, wrapped_parts = await self._process_file_box(
                    box, file_type_counter
                )
                assets.append(asset)
                string_parts.extend(wrapped_parts)

            elif isinstance(box, telegramify_markdown.type.Photo):
                logger.info(f"Photo box found: {box}")

        return (string_parts, assets)

    @cpu_bound
    def _merge_into_messages(self, string_parts: List[str]) -> List[str]:
        """Merge string parts into messages"""
        if not string_parts:
            return []

        messages = []
        current_message = ""

        for string_part in string_parts:
            if not string_part:
                continue

            part_len = len(string_part)
            current_len = len(current_message)

            if part_len > Config.MAX_MESSAGE_LENGTH:
                logger.warning(
                    f"Part exceeds limit ({part_len} chars), will try to send anyway"
                )
                if current_message:
                    messages.append(current_message)
                    current_message = ""
                messages.append(string_part)
                continue

            if (
                current_message
                and current_len + 2 + part_len <= Config.MAX_MESSAGE_LENGTH
            ):
                current_message += "\n\n" + string_part
            elif not current_message:
                current_message = string_part
            else:
                messages.append(current_message)
                current_message = string_part

        if current_message:
            messages.append(current_message)

        return messages

    @cpu_bound
    def _validate_markdown_v2(self, text: str) -> Tuple[bool, List[str]]:
        """
        Validate that text is properly escaped for MarkdownV2
        Returns (is_valid, list_of_issues)
        """
        issues = []
        special_chars = r"_*[]()~`>#+-=|{}.!"

        # Skip validation inside code blocks
        code_ranges = []
        for match in re.finditer(r"```[\s\S]*?```", text):
            code_ranges.append((match.start(), match.end()))

        i = 0
        while i < len(text):
            # Skip if inside code block
            in_code = any(start <= i < end for start, end in code_ranges)

            if not in_code and i < len(text) and text[i] in special_chars:
                # Check if escaped
                if i == 0 or text[i - 1] != "\\":
                    issues.append(f"Unescaped '{text[i]}' at position {i}")

            i += 1

        return (len(issues) == 0, issues)

    async def format_response_for_telegram(
        self, raw_text: str
    ) -> Tuple[List[str], List[Asset]]:
        """
        Main entry point: Format LLM response for Telegram

        Process flow:
        1. Convert LaTeX to Unicode for better display
        2. Pass to telegramify for markdown processing and escaping
        3. Split into messages respecting size limits
        """
        logger.info("Formatting response for Telegram...")

        try:
            # Step 1: Extract structure (LaTeX pre-processing happens inside)
            boxes = await self._extract_content_structure(raw_text)
            logger.info(f"Extracted {len(boxes)} boxes")

            # Step 2: Process boxes into parts
            string_parts, assets = await self._assemble_message_parts(boxes)
            logger.info(
                f"Assembled {len(string_parts)} string parts, {len(assets)} assets"
            )

            # Step 3: Merge parts into messages
            messages = await self._merge_into_messages(string_parts)
            logger.info(f"Merged into {len(messages)} messages")

            # Step 4: Validate messages
            for i, msg in enumerate(messages):
                if len(msg) > Config.MAX_MESSAGE_LENGTH:
                    logger.warning(f"Message {i+1} exceeds limit: {len(msg)} chars")

                # Check for markdown issues
                is_valid, issues = await self._validate_markdown_v2(msg)
                if not is_valid and issues:
                    for issue in issues[:3]:
                        logger.warning(f"Message {i+1}: {issue}")

            return (messages, assets)

        except Exception as e:
            logger.error(f"Error formatting response: {e}", exc_info=True)

            # Fallback: escape and split plain text
            escaped = escape_markdown(raw_text)

            if len(escaped) <= Config.MAX_MESSAGE_LENGTH:
                return ([escaped], [])
            else:
                messages = []
                for i in range(0, len(escaped), Config.SAFE_MESSAGE_LENGTH):
                    messages.append(escaped[i : i + Config.SAFE_MESSAGE_LENGTH])
                return (messages, [])
