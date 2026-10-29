// 复现 App.vue 里的流式解析逻辑，验证 chunk 切分和中文解码是否正确。
// 运行：node test_stream.mjs <doc_id> [端口，默认8000]
const API = `http://127.0.0.1:${process.argv[3] || 8000}`
const DOC_ID = process.argv[2]

if (!DOC_ID) {
  console.error('用法: node test_stream.mjs <doc_id> [port]')
  process.exit(1)
}

const res = await fetch(`${API}/api/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ doc_id: DOC_ID, question: '这个表格有几行' })
})

if (!res.ok || !res.body) {
  console.error('请求失败', res.status, await res.text())
  process.exit(1)
}

const reader = res.body.getReader()
const decoder = new TextDecoder('utf-8')
let buffer = ''
let answer = ''
let meta = null
let done = null
let deltas = 0
let rawChunks = 0

while (true) {
  const { value, done: finished } = await reader.read()
  if (finished) break
  rawChunks++

  buffer += decoder.decode(value, { stream: true })
  const parts = buffer.split('\n\n')
  buffer = parts.pop()

  for (const part of parts) {
    const line = part.trim()
    if (!line.startsWith('data:')) continue
    const payload = line.slice(5).trim()
    if (payload === '[DONE]') continue

    const evt = JSON.parse(payload)
    if (evt.type === 'meta') meta = evt
    else if (evt.type === 'delta') { answer += evt.content; deltas++ }
    else if (evt.type === 'done') done = evt
    else if (evt.type === 'error') { console.error('流内错误:', evt.message); process.exit(1) }
  }
}

console.log('原始 chunk 数  :', rawChunks)
console.log('meta 事件      :', JSON.stringify(meta))
console.log('delta 事件数   :', deltas)
console.log('done 事件      :', JSON.stringify(done))
console.log('拼接后回答     :', answer)
console.log('回答长度       :', answer.length)
console.log('中文是否乱码   :', /[\u4e00-\u9fa5]/.test(answer) ? '正常' : '异常')
