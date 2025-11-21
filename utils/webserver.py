from config import Config, logger
from storage import DatabaseManager

import asyncio
import jinja2
from typing import Optional
from pyngrok import ngrok
from aiohttp import web


class WebServer:
    """AIOHTTP server for hosting messages with correct format"""

    HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Answer</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        :root {
            --bg-base: #0f1117;
            --bg-raised: #1a1b26;
            --text-primary: #e8e9ed;
            --text-secondary: #9ca3af;
            --border-subtle: #27282f;
            --accent: #20808d;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "SF Pro Text", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background-color: var(--bg-base);
            font-size: 16px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        .container {
            max-width: 768px;
            margin: 0 auto;
            padding: 48px 24px;
        }

        .content {
            background: transparent;
            color: var(--text-primary);
        }

        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
            line-height: 1.3;
            margin: 24px 0 16px 0;
            color: var(--text-primary);
        }

        h1 { font-size: 32px; }
        h2 { font-size: 24px; margin-top: 32px; }
        h3 { font-size: 20px; }
        h4 { font-size: 18px; }

        p {
            margin: 16px 0;
            color: var(--text-primary);
            line-height: 1.7;
        }

        a {
            color: var(--accent);
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s;
        }

        a:hover {
            border-bottom-color: var(--accent);
        }

        pre {
            background: #1e1e2e;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 16px;
            overflow-x: auto;
            margin: 16px 0;
        }

        code {
            background: #1e1e2e;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "SF Mono", "Monaco", "Inconsolata", "Fira Code", "Consolas", monospace;
            font-size: 14px;
        }

        pre code {
            background: none;
            padding: 0;
            font-size: 14px;
            line-height: 1.5;
        }

        ul, ol {
            margin: 16px 0;
            padding-left: 24px;
        }

        li {
            margin: 8px 0;
            color: var(--text-primary);
        }

        table {
            border-collapse: collapse;
            margin: 24px 0;
            width: 100%;
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            overflow: hidden;
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border-subtle);
        }

        th {
            background: var(--bg-raised);
            font-weight: 600;
            color: var(--text-primary);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .katex {
            font-size: 1.1em;
            color: var(--text-primary);
        }

        .katex-display {
            margin: 24px 0;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 16px 0;
        }

        blockquote {
            border-left: 3px solid var(--accent);
            padding-left: 16px;
            margin: 16px 0;
            color: var(--text-secondary);
            font-style: italic;
        }

        hr {
            border: none;
            border-top: 1px solid var(--border-subtle);
            margin: 32px 0;
        }

        strong, b {
            font-weight: 600;
            color: var(--text-primary);
        }

        em, i {
            font-style: italic;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="content" id="content"></div>
    </div>
    <script>
        const rawContent = {{ content|tojson }};

        function protectLatex(text) {
            const latexBlocks = [];
            let counter = 0;

            text = text.replace(/\$\$([\s\S]*?)\$\$/g, function(match, latex) {
                const placeholder = `LATEXINLINE${counter}LATEXINLINE`;
                latexBlocks[counter] = { type: 'inline', content: latex };
                counter++;
                return placeholder;
            });

            text = text.replace(/\$([^$\n]+?)\$/g, function(match, latex) {
                const placeholder = `LATEXINLINE${counter}LATEXINLINE`;
                latexBlocks[counter] = { type: 'inline', content: latex };
                counter++;
                return placeholder;
            });

            return { text, latexBlocks };
        }

        function restoreLatex(html, latexBlocks) {
            for (let i = 0; i < latexBlocks.length; i++) {
                const block = latexBlocks[i];
                const placeholder = block.type === 'display'
                    ? `LATEXBLOCK${i}LATEXBLOCK`
                    : `LATEXINLINE${i}LATEXINLINE`;

                const latexHtml = block.type === 'display'
                    ? `<span class="katex-display">\\[${block.content}\\]</span>`
                    : `\\(${block.content}\\)`;

                html = html.replace(new RegExp(placeholder, 'g'), latexHtml);
            }
            return html;
        }

        const { text: protectedText, latexBlocks } = protectLatex(rawContent);

        marked.setOptions({
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                }
                return hljs.highlightAuto(code).value;
            },
            breaks: true,
            gfm: true
        });

        let htmlContent = marked.parse(protectedText);
        htmlContent = restoreLatex(htmlContent, latexBlocks);
        document.getElementById('content').innerHTML = htmlContent;

        renderMathInElement(document.body, {
            delimiters: [
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false},
            ],
            throwOnError: false,
            trust: true
        });

        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    </script>
</body>
</html>
    """

    def __init__(self, storage: DatabaseManager, host: str, port: int) -> None:
        self.storage = storage
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.public_url: Optional[str] = None

        self.template = jinja2.Template(self.HTML_TEMPLATE)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup AIOHTTP routes"""
        self.app.router.add_get("/answer/{page_id}", self.view_answer)

    async def view_answer(self, request: web.Request) -> web.Response:
        page_id = request.match_info["page_id"]
        content = await self.storage.load_web_page(page_id)

        if content is None:
            raise web.HTTPNotFound(text="Page not found")

        html = self.template.render(content=content)
        return web.Response(text=html, content_type="text/html")

    async def start(self) -> None:
        """Start web server and ngrok"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()

        logger.info(f"Web server running at http://{self.host}:{self.port}")

        await self._start_ngrok()

    async def _start_ngrok(self) -> None:
        """Initialize ngrok in a separate thread to avoid blocking the loop"""
        loop = asyncio.get_running_loop()

        if Config.NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(Config.NGROK_AUTH_TOKEN)

        ngrok_kwargs = {"addr": self.port}
        if Config.NGROK_DOMAIN:
            ngrok_kwargs["domain"] = Config.NGROK_DOMAIN

        try:
            tunnel = await loop.run_in_executor(
                None, lambda: ngrok.connect(**ngrok_kwargs)
            )
            self.public_url = tunnel.public_url
            logger.info(f"Ngrok tunnel established: {self.public_url}")
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")

    def get_answer_url(self, page_id: str) -> str:
        """Get public URL for answer page"""
        if not self.public_url:
            return f"http://{self.host}:{self.port}/answer/{page_id}"
        return f"{self.public_url}/answer/{page_id}"

    async def stop(self) -> None:
        """Cleanup method"""
        if self.runner:
            await self.runner.cleanup()
