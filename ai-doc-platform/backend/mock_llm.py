"""本地假大模型服务，只为验证 SSE 流式链路，不消耗任何 API 额度。

实现了 OpenAI 兼容的 /v1/chat/completions（stream=true），
按字符逐个吐出，模拟真实模型的 token 流。

用法（只在测试时用，正式跑请删掉）：
    python mock_llm.py            # 监听 9999
    # 另开终端：
    LLM_BASE_URL=http://127.0.0.1:9999/v1 LLM_API_KEY=sk-mock-key-for-local-test-only \
        python -m uvicorn main:app --port 8001
"""

import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

MOCK_ANSWER = "这份文档包含 3 行数据，共 4 列：姓名、年龄、城市、备注。其中「李四」的备注列有跨行内容。"


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    question = ""
    for m in body.get("messages", []):
        if m.get("role") == "user":
            question = m.get("content", "")

    def gen():
        # 先确认真的收到了文档内容，否则说明上下文拼装有问题
        if "文档内容" not in question:
            yield _chunk("【错误】没有收到文档内容")
        for ch in MOCK_ANSWER:
            yield _chunk(ch)
            time.sleep(0.02)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _chunk(text: str) -> str:
    payload = {
        "id": "mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "mock",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9999, log_level="warning")