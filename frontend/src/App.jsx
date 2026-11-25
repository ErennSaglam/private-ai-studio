import { useEffect, useRef, useState } from 'react'
import './App.css'

const API = 'http://localhost:3001/api'

const SUGGESTIONS = [
  'Kendini kısaca tanıt.',
  'Bana bir Python fonksiyonu yaz: listedeki tek sayıları toplasın.',
  'Fine-tuning nedir? Basitçe anlat.',
  'Verimli çalışmak için 3 öneri ver.',
]

const DEFAULT_SETTINGS = {
  // Written in English on purpose: the 1B model copies the language of the instruction itself.
  // A Turkish instruction made it answer English questions in Turkish; this one keeps the user's language.
  systemPrompt: "You are a helpful assistant. Always reply in the same language as the user's message. Keep answers short and clear.",
  temperature: 0.7,
  maxTokens: 512,
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [info, setInfo] = useState(null)
  const [online, setOnline] = useState(null)
  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [showSettings, setShowSettings] = useState(false)
  const [copied, setCopied] = useState(null)
  const chatEnd = useRef(null)
  const textarea = useRef(null)

  useEffect(() => {
    fetch(`${API}/model-info`)
      .then(r => r.json())
      .then(data => { setInfo(data); setOnline(true) })
      .catch(() => setOnline(false))
  }, [])

  useEffect(() => {
    chatEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    const el = textarea.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [input])

  async function send(text) {
    const content = text.trim()
    if (!content || loading) return
    const history = [...messages, { role: 'user', content }]
    setMessages(history)
    setInput('')
    setLoading(true)

    const payload = history.map(({ role, content }) => ({ role, content }))
    if (settings.systemPrompt.trim()) {
      payload.unshift({ role: 'system', content: settings.systemPrompt.trim() })
    }

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: payload,
          temperature: Number(settings.temperature),
          max_tokens: Number(settings.maxTokens),
        }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setMessages([...history, { role: 'assistant', content: data.reply, stats: data }])
      setOnline(true)
    } catch (err) {
      setMessages([...history, { role: 'assistant', content: `Hata: ${err.message}`, error: true }])
      if (err instanceof TypeError) setOnline(false)
    } finally {
      setLoading(false)
      textarea.current?.focus()
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      send(input)
    }
  }

  async function copy(text, i) {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(i)
      setTimeout(() => setCopied(null), 1500)
    } catch { /* clipboard unavailable */ }
  }

  function update(key, value) {
    setSettings(s => ({ ...s, [key]: value }))
  }

  const baseName = info?.model_path?.split('/').pop() ?? 'Model'
  const adapterName = info?.adapter_path?.split('/').pop()
  const modelName = adapterName ? `${baseName} + LoRA: ${adapterName}` : baseName

  return (
    <div className="layout">
      <header className="topbar">
        <div className="brand">
          <div className="logo">AI</div>
          <div>
            <div className="title">Private AI Studio</div>
            <div className="subtitle">
              <span className={`dot ${online === false ? 'off' : online ? 'on' : ''}`} />
              {online === false ? 'Backend bağlantısı yok' : `${modelName} · yerel · çevrimdışı`}
            </div>
          </div>
        </div>
        <div className="actions">
          <button className="ghost" onClick={() => setShowSettings(v => !v)} aria-expanded={showSettings}>
            Ayarlar
          </button>
          <button className="ghost" onClick={() => setMessages([])} disabled={loading || !messages.length}>
            Yeni sohbet
          </button>
        </div>
      </header>

      {showSettings && (
        <section className="settings">
          <label className="field wide">
            <span>Sistem mesajı</span>
            <textarea
              rows={2}
              value={settings.systemPrompt}
              onChange={e => update('systemPrompt', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Sıcaklık <b>{Number(settings.temperature).toFixed(1)}</b></span>
            <input
              type="range" min="0" max="2" step="0.1"
              value={settings.temperature}
              onChange={e => update('temperature', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Maks. token <b>{settings.maxTokens}</b></span>
            <input
              type="range" min="32" max="2048" step="32"
              value={settings.maxTokens}
              onChange={e => update('maxTokens', e.target.value)}
            />
          </label>
          {info && (
            <div className="model-facts wide">
              <span>{info.num_layers} katman</span>
              <span>{info.hidden_size} gizli boyut</span>
              <span>{info.num_attention_heads} attention head</span>
              <span>{info.quantization ? `${info.quantization.bits}-bit` : 'tam hassasiyet'}</span>
              <span>{info.weights_size_gb} GB</span>
            </div>
          )}
        </section>
      )}

      <main className="chat">
        <div className="thread">
          {!messages.length && (
            <div className="empty">
              <div className="logo big">AI</div>
              <h1>Bugün ne yapalım?</h1>
              <p>Her şey bu bilgisayarda çalışır. Mesajların hiçbir sunucuya gitmez.</p>
              <div className="suggestions">
                {SUGGESTIONS.map(s => (
                  <button key={s} onClick={() => send(s)} disabled={loading || online === false}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`row ${m.role}`}>
              {m.role === 'assistant' && <div className="avatar">AI</div>}
              <div className="bubble-wrap">
                <div className={`bubble ${m.error ? 'error' : ''}`}>{m.content}</div>
                {m.role === 'assistant' && !m.error && (
                  <div className="meta">
                    {m.stats && (
                      <span>
                        {m.stats.generated_tokens} token · {m.stats.tokens_per_second} tok/sn · {m.stats.elapsed_seconds} sn · {m.stats.peak_memory_gb} GB
                      </span>
                    )}
                    <button className="link" onClick={() => copy(m.content, i)}>
                      {copied === i ? 'Kopyalandı' : 'Kopyala'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="row assistant">
              <div className="avatar">AI</div>
              <div className="bubble typing"><span /><span /><span /></div>
            </div>
          )}
          <div ref={chatEnd} />
        </div>
      </main>

      <footer className="composer-area">
        <form
          className="composer"
          onSubmit={e => { e.preventDefault(); send(input) }}
        >
          <textarea
            ref={textarea}
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Mesajını yaz…"
            autoFocus
          />
          <button className="send" disabled={loading || !input.trim()} aria-label="Gönder">
            <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
              <path d="M12 19V5M5 12l7-7 7 7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </form>
        <div className="hint">Enter gönderir · Shift+Enter yeni satır</div>
      </footer>
    </div>
  )
}
