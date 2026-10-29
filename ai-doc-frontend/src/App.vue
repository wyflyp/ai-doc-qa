<script setup>
import { ref, computed, onMounted, nextTick } from 'vue'

const API = 'http://127.0.0.1:8000'

const docs = ref([])
const selected = ref([])          // 多选的文档 id
const question = ref('')
const answer = ref('')
const streaming = ref(false)
const sources = ref([])           // 本次检索到的片段
const mode = ref('')              // rag / fulltext
const elapsed = ref(0)
const error = ref('')
const uploading = ref(false)
const answerBox = ref(null)
const stats = ref(null)
const showSources = ref(true)
const retrievalMode = ref('hybrid')   // hybrid / vector / bm25
const useRerank = ref(true)    // 57 条题上 +8，默认开
const useRewrite = ref(false)  // 叠加精排后无增益且拖慢，默认关
const useDecompose = ref(true) // 跨文档题：拆子问题分别检索，CROSS 6/8→8/8
const subQuestions = ref([])   // 拆出的子问题
const rewritten = ref([])             // 查询改写结果
const rewriteMs = ref(0)

const canAsk = computed(() => question.value.trim() && !streaming.value)

onMounted(async () => {
  await Promise.all([loadDocs(), loadStats()])
})

async function loadDocs() {
  const res = await fetch(`${API}/api/documents`)
  if (res.ok) docs.value = (await res.json()).items
}

async function loadStats() {
  try {
    const res = await fetch(`${API}/api/rag/stats`)
    if (res.ok) stats.value = await res.json()
  } catch { /* 统计失败不影响主流程 */ }
}

async function onFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  uploading.value = true
  error.value = ''
  const fd = new FormData()
  fd.append('file', file)
  try {
    const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || '上传失败')
    docs.value = [data.document, ...docs.value]
    // 上传后自动选中这个文档
    if (!selected.value.includes(data.document.id)) {
      selected.value = [data.document.id]
    }
    if (data.index_error) {
      error.value = `文档已入库，但索引失败：${data.index_error}`
    }
    await loadStats()
  } catch (err) {
    error.value = err.message
  } finally {
    uploading.value = false
    e.target.value = ''
  }
}

function toggleDoc(id) {
  const i = selected.value.indexOf(id)
  if (i >= 0) selected.value.splice(i, 1)
  else selected.value.push(id)
}

function clearSelection() {
  selected.value = []
}

async function reindex() {
  error.value = ''
  try {
    const res = await fetch(`${API}/api/reindex`, { method: 'POST' })
    const data = await res.json()
    if (!res.ok) throw new Error(data.detail || '重建失败')
    await loadStats()
    if (data.failed?.length) {
      error.value = `部分文档索引失败：${data.failed.map(f => f.filename).join('、')}`
    } else {
      error.value = ''
    }
  } catch (err) {
    error.value = err.message
  }
}

// ★ 核心：fetch + ReadableStream 接收 POST 的 SSE 流
// EventSource 只支持 GET，传不了 body，所以这里必须用 fetch
async function send() {
  const q = question.value.trim()
  if (!q || streaming.value) return

  question.value = ''
  answer.value = ''
  error.value = ''
  sources.value = []
  mode.value = ''
  elapsed.value = 0
  streaming.value = true

  try {
    const body = { question: q, top_k: 4, use_rag: true, mode: retrievalMode.value, use_rerank: useRerank.value, use_rewrite: useRewrite.value, use_decompose: useDecompose.value }
    if (selected.value.length) body.doc_ids = selected.value

    const res = await fetch(`${API}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })

    if (!res.ok || !res.body) {
      const detail = await res.json().catch(() => ({}))
      throw new Error(detail.detail || `请求失败 ${res.status}`)
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    // SSE 的 chunk 边界是任意的，一条消息可能被切成两块，
    // 所以必须先 buffer 起来，攒够 "\n\n" 才算一条完整消息
    while (true) {
      const { value, done } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop()   // 最后一段可能不完整，留到下一轮

      for (const chunk of chunks) {
        const line = chunk.trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') continue

        const evt = JSON.parse(payload)
        if (evt.type === 'meta') {
          sources.value = evt.sources || []
          mode.value = evt.mode || ''
          rewritten.value = evt.rewritten || []
          rewriteMs.value = evt.rewrite_ms || 0
          subQuestions.value = evt.sub_questions || []
        } else if (evt.type === 'delta') {
          answer.value += evt.content
          await scrollToBottom()
        } else if (evt.type === 'done') {
          elapsed.value = evt.elapsed_ms
        } else if (evt.type === 'error') {
          error.value = evt.message
        }
      }
    }
  } catch (err) {
    error.value = err.message
  } finally {
    streaming.value = false
    loadStats()
  }
}

async function scrollToBottom() {
  await nextTick()
  if (answerBox.value) {
    answerBox.value.scrollTop = answerBox.value.scrollHeight
  }
}
</script>

<template>
  <div class="layout">
    <!-- 左侧：文档列表（可多选） -->
    <aside class="sidebar">
      <div class="side-head">
        <h2>知识库</h2>
        <span v-if="stats" class="badge">{{ stats.chunks }} 片段</span>
      </div>

      <label class="upload">
        <input type="file" accept=".txt,.csv,.md" @change="onFileChange" />
        <span>{{ uploading ? '上传并建索引中…' : '+ 上传 txt / csv / md' }}</span>
      </label>

      <div class="side-tools">
        <button class="ghost" @click="clearSelection">清空选择</button>
        <button class="ghost" @click="reindex">重建索引</button>
      </div>

      <p class="hint">
        不勾选任何文档 = 全库检索；勾选多个 = 跨文档检索
      </p>

      <ul class="doc-list">
        <li
          v-for="d in docs"
          :key="d.id"
          :class="{ active: selected.includes(d.id) }"
          @click="toggleDoc(d.id)"
        >
          <span class="check">{{ selected.includes(d.id) ? '✓' : '' }}</span>
          <div class="doc-body">
            <div class="name">{{ d.filename }}</div>
            <div class="meta">{{ d.summary || d.ext }}</div>
          </div>
        </li>
      </ul>

      <p v-if="!docs.length" class="empty">还没有文档，先上传一个</p>
    </aside>

    <!-- 右侧：问答区 -->
    <main class="main">
      <header>
        <div class="head-row">
          <h2>AI 文档问答 <small v-if="stats">· {{ stats.provider === 'local' ? '语义向量' : '词面向量' }}</small></h2>
          <div class="head-tools">
            <label class="rr-toggle" :class="{ on: useDecompose }" title="多子问题会被拆开分别检索，改善跨文档问答">
              <input type="checkbox" v-model="useDecompose" :disabled="streaming" />
              分解
            </label>
            <label class="rr-toggle" :class="{ on: useRewrite }" title="查询改写：把小模型生成的多角度查询一起检索">
              <input type="checkbox" v-model="useRewrite" :disabled="streaming" />
              改写
            </label>
            <label class="rr-toggle" :class="{ on: useRerank }">
              <input type="checkbox" v-model="useRerank" :disabled="streaming" />
              精排
            </label>
            <select v-model="retrievalMode" class="mode-select" :disabled="streaming">
              <option value="hybrid">混合检索（推荐）</option>
              <option value="vector">仅向量</option>
              <option value="bm25">仅 BM25</option>
            </select>
          </div>
        </div>
        <p class="sub">
          已选 {{ selected.length || '全部' }} 个文档
          <span v-if="mode" class="mode-tag" :class="mode">
            {{ mode === 'rag' ? 'RAG 检索问答' : '全文模式（未命中片段）' }}
          </span>
        </p>
      </header>

      <p v-if="error" class="error">⚠ {{ error }}</p>

      <!-- 查询改写结果：让用户看到系统是怎么理解问题的 -->
      <section v-if="rewritten.length" class="rewrite">
        <span class="rw-label">检索时还查了</span>
        <span v-for="(q, i) in rewritten" :key="i" class="rw-item">{{ q }}</span>
        <span class="rw-cost">+{{ rewriteMs }}ms</span>
      </section>

      <!-- 子问题拆解：跨文档题会拆成多个子问题分别检索 -->
      <section v-if="subQuestions.length" class="rewrite">
        <span class="rw-label">拆成子问题</span>
        <span v-for="(q, i) in subQuestions" :key="i" class="rw-item sub-q">
          {{ i + 1 }}. {{ q }}
        </span>
      </section>

      <div ref="answerBox" class="answer">
        <p v-if="streaming && !answer" class="placeholder small">
          正在检索片段并生成回答<span class="cursor">▍</span>
        </p>
        <pre v-else-if="answer" class="text">{{ answer }}<span v-if="streaming" class="cursor">▍</span></pre>
        <p v-else class="placeholder small">
          在左侧选择文档（可多选），然后在下面提问
        </p>

        <p v-if="elapsed" class="elapsed">耗时 {{ (elapsed / 1000).toFixed(2) }}s</p>
      </div>

      <!-- 检索依据：RAG 的可解释性就体现在这里 -->
      <section v-if="sources.length" class="sources">
        <div class="sources-head" @click="showSources = !showSources">
          <span>依据 {{ sources.length }} 个片段</span>
          <span class="toggle">{{ showSources ? '收起' : '展开' }}</span>
        </div>
        <div v-if="showSources" class="source-list">
          <div v-for="(s, i) in sources" :key="i" class="source">
            <div class="source-top">
              <span class="idx">片段{{ i + 1 }}</span>
              <span class="fname">{{ s.filename }}</span>
              <span class="score">融合分 {{ s.score }}</span>
              <span v-if="s.kind === 'table'" class="kind">表格</span>
            </div>
            <div class="hit-row">
              <span
                v-for="m in s.matched_by"
                :key="m"
                class="hit"
                :class="m"
              >{{ m === 'vector' ? '向量命中' : 'BM25 命中' }}</span>
              <span v-if="s.rerank_score !== null && s.rerank_score !== undefined" class="hit rerank">
                精排 {{ s.rerank_score }}
              </span>
              <span v-if="s.vector_score !== null" class="sub">向量 {{ s.vector_score }}</span>
              <span v-if="s.bm25_score !== null" class="sub">BM25 {{ s.bm25_score }}</span>
            </div>
            <p class="preview">{{ s.preview }}…</p>
          </div>
        </div>
      </section>

      <form class="composer" @submit.prevent="send">
        <textarea
          v-model="question"
          placeholder="基于所选文档提问，例如：年假有几天？出差住宿标准是多少？"
          rows="3"
          @keydown.enter.exact.prevent="send"
        ></textarea>
        <button type="submit" :disabled="!canAsk">
          {{ streaming ? '生成中…' : '发送' }}
        </button>
      </form>
    </main>
  </div>
</template>

<style scoped>
* { box-sizing: border-box; }

.layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  height: 100vh;
  font-family: -apple-system, "Microsoft YaHei", sans-serif;
  background: #f7f8fa;
  color: #1f2328;
}

/* --- 侧栏 --- */
.sidebar {
  background: #fff;
  border-right: 1px solid #e4e6eb;
  padding: 20px 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}
.side-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
.side-head h2 { margin: 0; font-size: 15px; color: #57606a; }
.badge { font-size: 11px; background: #ddf4ff; color: #0969da; padding: 2px 8px; border-radius: 10px; }

.upload input { display: none; }
.upload span {
  display: block;
  text-align: center;
  padding: 10px;
  border: 1px dashed #d0d7de;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  color: #0969da;
}
.upload span:hover { background: #f0f6ff; border-color: #0969da; }

.side-tools { display: flex; gap: 8px; margin-top: 10px; }
.ghost {
  flex: 1;
  padding: 6px;
  font-size: 12px;
  background: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  cursor: pointer;
  color: #57606a;
}
.ghost:hover { background: #eef1f4; }

.hint { font-size: 11px; color: #8b949e; line-height: 1.6; margin: 12px 0 8px; }

.doc-list { list-style: none; margin: 0; padding: 0; }
.doc-list li {
  display: flex;
  gap: 8px;
  padding: 10px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 4px;
  align-items: flex-start;
}
.doc-list li:hover { background: #f0f2f5; }
.doc-list li.active { background: #ddf4ff; }
.check {
  width: 16px;
  height: 16px;
  border: 1px solid #d0d7de;
  border-radius: 4px;
  font-size: 11px;
  line-height: 15px;
  text-align: center;
  color: #0969da;
  flex-shrink: 0;
  margin-top: 2px;
  background: #fff;
}
.doc-list li.active .check { border-color: #0969da; }
.doc-body { min-width: 0; }
.doc-list .name { font-size: 13px; font-weight: 500; word-break: break-all; }
.doc-list .meta { font-size: 11px; color: #6e7781; margin-top: 3px; }
.empty { font-size: 13px; color: #8b949e; }

/* --- 主区 --- */
.main { display: flex; flex-direction: column; padding: 20px 28px; overflow: hidden; }
.main header h2 { margin: 0 0 6px; font-size: 18px; }
.head-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.head-tools { display: flex; align-items: center; gap: 8px; }
.rr-toggle {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #8b949e;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  padding: 4px 9px;
  cursor: pointer;
  user-select: none;
  background: #fff;
}
.rr-toggle.on { color: #1a7f37; border-color: #8ce99a; background: #f0fff4; }
.rr-toggle input { margin: 0; cursor: pointer; }
.mode-select {
  padding: 5px 10px;
  font-size: 12px;
  border: 1px solid #d0d7de;
  border-radius: 6px;
  background: #fff;
  color: #57606a;
  cursor: pointer;
}
.mode-select:disabled { opacity: 0.6; cursor: not-allowed; }
.main header small { font-size: 12px; font-weight: 400; color: #8b949e; }
.main header .sub { margin: 0; font-size: 13px; color: #6e7781; }
.mode-tag { margin-left: 8px; padding: 1px 8px; border-radius: 10px; font-size: 11px; }
.mode-tag.rag { background: #dafbe1; color: #1a7f37; }
.mode-tag.fulltext { background: #fff8c5; color: #9a6700; }

.error {
  background: #ffebe9;
  border: 1px solid #ff8182;
  color: #cf222e;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 13px;
  margin: 10px 0 0;
}

.rewrite {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  margin: 8px 0 0;
  font-size: 11px;
}
.rw-label { color: #8b949e; }
.rw-item {
  background: #eef1f4;
  color: #57606a;
  padding: 2px 8px;
  border-radius: 10px;
  border: 1px solid #e4e6eb;
}
.rw-cost { color: #8b949e; }
.rw-item.sub-q { background: #eaf6ff; color: #0969da; border-color: #cfe8ff; }

.placeholder { color: #8b949e; display: flex; align-items: center; justify-content: center; flex: 1; }
.placeholder.small { font-size: 14px; }

.answer {
  flex: 1;
  overflow-y: auto;
  background: #fff;
  border: 1px solid #e4e6eb;
  border-radius: 10px;
  padding: 18px;
  margin: 16px 0 0;
  min-height: 120px;
}
.text { white-space: pre-wrap; font-family: inherit; font-size: 14px; line-height: 1.8; margin: 0; }
.cursor { color: #0969da; animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }
.elapsed { margin: 14px 0 0; font-size: 11px; color: #8b949e; }

/* --- 检索依据 --- */
.sources { margin-top: 12px; flex-shrink: 0; max-height: 32%; overflow-y: auto; }
.sources-head {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #57606a;
  cursor: pointer;
  padding: 6px 2px;
  user-select: none;
}
.sources-head .toggle { color: #0969da; }
.source-list { display: flex; flex-direction: column; gap: 6px; }
.source {
  background: #fff;
  border: 1px solid #e4e6eb;
  border-left: 3px solid #0969da33;
  border-radius: 6px;
  padding: 8px 10px;
}
.source-top { display: flex; gap: 8px; align-items: center; font-size: 11px; }
.source-top .idx { color: #0969da; font-weight: 600; }
.source-top .fname { color: #57606a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 40%; }
.source-top .score { color: #8b949e; }
.source-top .kind { background: #fff8c5; color: #9a6700; padding: 0 6px; border-radius: 8px; }
.hit-row { display: flex; gap: 6px; align-items: center; margin-top: 5px; flex-wrap: wrap; }
.hit {
  font-size: 10px;
  padding: 1px 7px;
  border-radius: 8px;
  border: 1px solid transparent;
}
.hit.vector { background: #ddf4ff; color: #0969da; border-color: #b6e3ff; }
.hit.bm25 { background: #fbefff; color: #8250df; border-color: #e9d4ff; }
.hit.rerank { background: #f0fff4; color: #1a7f37; border-color: #8ce99a; }
.sub { font-size: 10px; color: #8b949e; }
.preview { margin: 4px 0 0; font-size: 12px; color: #6e7781; line-height: 1.5; }

/* --- 输入区 --- */
.composer { display: flex; gap: 12px; align-items: flex-end; margin-top: 14px; }
.composer textarea {
  flex: 1;
  padding: 10px 12px;
  border: 1px solid #d0d7de;
  border-radius: 10px;
  font-size: 14px;
  font-family: inherit;
  resize: none;
}
.composer textarea:focus { outline: 2px solid #0969da33; border-color: #0969da; }
.composer button {
  padding: 12px 26px;
  border: none;
  border-radius: 10px;
  background: #0969da;
  color: #fff;
  font-size: 14px;
  cursor: pointer;
}
.composer button:disabled { background: #afb8c1; cursor: not-allowed; }
</style>
