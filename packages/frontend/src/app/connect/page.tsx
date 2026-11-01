'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { api, type Chat, type IndexingStatus } from '@/lib/api'
import { useAuth } from '@/lib/auth'

type Step = 'loading' | 'invitation' | 'telegram' | 'chats' | 'plugin'
const button = 'rounded-full bg-primary px-6 py-3 text-sm font-medium text-surface transition-opacity hover:opacity-80 disabled:opacity-40'
const input = 'w-full rounded-xl border bg-transparent px-4 py-3 text-primary outline-none focus:ring-2 focus:ring-blue-500'
const mcpUrl = 'https://telegram.waiwai.is/mcp'
const assistantInstructions = `Подключи Wai Telegram через MCP: ${mcpUrl}. Авторизация: Bearer-токен. Запроси мой личный ключ через защищённое поле, не в чате. Если нужен локальный транспорт, используй npx -y mcp-remote ${mcpUrl} --header "Authorization:Bearer \${WAI_TELEGRAM_TOKEN}" --transport http-only. После подключения вызови list_chats и проверь доступ. Ищи только в моих данных. Отправляй сообщения только по моему прямому поручению; добавляй в конце пустую строку и __Отправлено AI-помощником.__ и проверяй доставку через get_message.`

export default function ConnectPage() {
  const { user, isLoading, acceptInvitation } = useAuth()
  const booted = useRef(false)
  const [step, setStep] = useState<Step>('loading')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [phone, setPhone] = useState('')
  const [codeHash, setCodeHash] = useState('')
  const [code, setCode] = useState('')
  const [password, setPassword] = useState('')
  const [needsPassword, setNeedsPassword] = useState(false)
  const [chats, setChats] = useState<Chat[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [query, setQuery] = useState('')
  const [key, setKey] = useState('')
  const [copied, setCopied] = useState('')
  const [coverage, setCoverage] = useState<IndexingStatus | null>(null)
  const [pluginUsed, setPluginUsed] = useState(false)
  const [autoSince, setAutoSince] = useState(user?.auto_index_since ?? null)

  async function loadChats(refresh = false) {
    const page = refresh ? await api.refreshChats() : await api.getChats(undefined, 1000)
    let all = page.chats
    let cursor = page.next_cursor
    while (cursor) {
      const next = await api.getChats(undefined, 1000, cursor)
      all = [...all, ...next.chats]
      cursor = next.next_cursor
    }
    setChats(all)
    setSelected(new Set(all.filter(chat => chat.indexing_enabled).map(chat => chat.id)))
    setStep('chats')
  }

  useEffect(() => {
    if (isLoading || booted.current) return
    booted.current = true
    const invite = new URLSearchParams(window.location.hash.slice(1)).get('invite')
    if (invite) window.history.replaceState(null, '', window.location.pathname)
    async function boot() {
      try {
        if (invite) await acceptInvitation(invite)
        else if (!user) { setStep('invitation'); return }
        const since = useAuth.getState().user?.auto_index_since ?? null
        setAutoSince(since)
        const session = await api.getTelegramSession()
        if (session && since) setStep('plugin')
        else if (session) await loadChats()
        else setStep('telegram')
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Не удалось открыть подключение')
        setStep('invitation')
      }
    }
    void boot()
  }, [isLoading, user, acceptInvitation])

  useEffect(() => { window.scrollTo(0, 0) }, [step])

  useEffect(() => {
    if (step !== 'plugin') return
    let cancelled = false
    const refresh = async () => {
      try {
        const [status, keys] = await Promise.all([api.getIndexingStatus(), api.listApiKeys()])
        if (!cancelled) {
          setCoverage(status)
          setPluginUsed(keys.some(item => item.name === 'Wai Telegram plugin' && item.is_active && item.last_used_at))
        }
      } catch { /* Keep the last confirmed progress during a transient outage. */ }
    }
    void refresh()
    const timer = window.setInterval(refresh, 10000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [step])

  async function perform(action: () => Promise<void>) {
    setBusy(true)
    setError('')
    try { await action() }
    catch (err) { setError(err instanceof Error ? err.message : 'Не удалось выполнить действие') }
    finally { setBusy(false) }
  }

  async function loginTelegram(event: React.FormEvent) {
    event.preventDefault()
    await perform(async () => {
      if (!codeHash) {
        const result = await api.requestCode(phone.trim())
        setCodeHash(result.phone_code_hash)
        return
      }
      try {
        const result = await api.verifyCode(phone.trim(), codeHash, code.trim(), password || undefined)
        if (result.access_token && result.refresh_token) {
          useAuth.getState().completeTelegramLogin(result.access_token, result.refresh_token)
        }
      }
      catch (err) {
        if (err instanceof Error && /two-factor|password required/i.test(err.message)) {
          setNeedsPassword(true)
          return
        }
        if (err instanceof Error && /истёк|expired|no pending authentication/i.test(err.message)) {
          setCodeHash(''); setCode(''); setPassword(''); setNeedsPassword(false)
        }
        throw err
      }
      setCode(''); setPassword('')
      if (autoSince) setStep('plugin')
      else await loadChats(true)
    })
  }

  async function startIndexing() {
    await perform(async () => {
      await api.selectIndexedChats([...selected])
      setStep('plugin')
    })
  }

  async function copy(value: string, label: string) {
    await perform(async () => { await navigator.clipboard.writeText(value); setCopied(label) })
  }

  const steps = autoSince ? ['Telegram', 'Плагин'] : ['Telegram', 'Чаты', 'Плагин']
  const active = step === 'telegram' ? 0 : step === 'chats' ? 1 : step === 'plugin' ? steps.length - 1 : -1
  const period = autoSince ? new Date(`${autoSince.slice(0, 10)}T00:00:00Z`).toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC' }).replace(/\s*г\.$/, '') : null
  const visible = chats.filter(chat => `${chat.title} ${chat.username ?? ''}`.toLocaleLowerCase().includes(query.toLocaleLowerCase()))
  const totals = coverage?.chats.reduce((sum, chat) => ({
    text: sum.text + chat.text_messages, indexed: sum.indexed + chat.indexed_texts,
    media: sum.media + chat.pending_media,
    failed: sum.failed + (chat.failed_media ?? 0),
  }), { text: 0, indexed: 0, media: 0, failed: 0 })
  const historyReady = Boolean(coverage?.chats.length && coverage.chats.every(chat => chat.history_complete))

  return (
    <main className="mx-auto min-h-screen max-w-2xl px-6 py-10 sm:py-16">
      <header className="mb-14 flex items-center justify-between">
        <Link href="/" className="text-lg font-semibold tracking-tight">wai<span className="text-blue-500">.</span> telegram</Link>
        <span className="text-xs text-secondary">Твой Telegram для AI</span>
      </header>
      {active >= 0 && <nav aria-label="Шаги подключения" className="mb-10 flex gap-6 text-xs sm:gap-10">
        {steps.map((label, index) => <span key={label} aria-current={index === active ? 'step' : undefined} className={index === active ? 'font-semibold text-primary' : 'text-secondary'}>
          <span className={index <= active ? 'mr-2 text-blue-500' : 'mr-2'}>{index < active ? '✓' : `0${index + 1}`}</span>{label}
        </span>)}
      </nav>}
      {error && <p role="alert" className="mb-6 rounded-xl border border-red-300 p-4 text-sm text-red-600">{error}</p>}
      {step === 'loading' && <p role="status" className="text-secondary">Открываем подключение…</p>}
      {step === 'invitation' && <>
        <h1 className="mb-4 text-4xl font-medium tracking-tight">Начни с приглашения</h1>
        <p className="max-w-md leading-relaxed text-secondary">Открой персональную ссылку от Мика. Она подключит твой аккаунт без отдельной регистрации.</p>
        <Link href="/login" className="mt-8 inline-block text-sm underline underline-offset-4">Уже есть аккаунт WAI</Link>
      </>}
      {step === 'telegram' && <>
        <h1 className="mb-4 text-4xl font-medium tracking-tight">Подключи свой Telegram</h1>
        <p className="mb-8 leading-relaxed text-secondary">{autoSince ? `Все чаты подключатся автоматически. История с ${period} и новые сообщения — для поиска по смыслу.` : 'Войди в свой аккаунт. На следующем шаге выберешь переписки для поиска.'}</p>
        <form onSubmit={loginTelegram} className="max-w-sm space-y-5">
          {!codeHash ? <label className="block text-sm">Номер телефона
            <input className={`${input} mt-2`} type="tel" autoComplete="tel" placeholder="+7 999 123-45-67" value={phone} onChange={event => setPhone(event.target.value)} required />
          </label> : <>
            <label className="block text-sm">Код из Telegram
              <input className={`${input} mt-2 tracking-[0.3em]`} inputMode="numeric" autoComplete="one-time-code" value={code} onChange={event => setCode(event.target.value)} required />
            </label>
            {needsPassword && <label className="block text-sm">Пароль двухэтапной проверки
              <input className={`${input} mt-2`} type="password" autoComplete="current-password" value={password} onChange={event => setPassword(event.target.value)} required />
            </label>}
          </>}
          <button className={button} disabled={busy}>{busy ? 'Подключаем…' : codeHash ? 'Войти' : 'Получить код'}</button>
          {codeHash && <button type="button" disabled={busy} className="ml-4 text-sm text-secondary" onClick={() => { setCodeHash(''); setCode(''); setPassword(''); setNeedsPassword(false) }}>Другой номер</button>}
        </form>
      </>}
      {step === 'chats' && <>
        <h1 className="mb-4 text-4xl font-medium tracking-tight">Что будем помнить?</h1>
        <p className="mb-7 leading-relaxed text-secondary">Выбери нужные переписки. Сохраним всю доступную историю для поиска по смыслу — вместе с текстом голосовых и документов.</p>
        <input aria-label="Найти чат" className={input} type="search" placeholder="Найти человека или чат" value={query} onChange={event => setQuery(event.target.value)} />
        <div className="my-4 flex items-center justify-between gap-4 text-xs text-secondary">
          <span>Выбрано: {selected.size}</span>
          <button disabled={busy} onClick={() => void perform(() => loadChats(true))}>Обновить список</button>
        </div>
        <div className="max-h-[44vh] overflow-auto border-y">
          {visible.map(chat => <label key={chat.id} className="flex cursor-pointer items-center gap-4 border-b px-1 py-4 last:border-b-0 hover:bg-surface-hover">
            <input type="checkbox" checked={selected.has(chat.id)} onChange={event => setSelected(previous => {
              const next = new Set(previous)
              if (event.target.checked) next.add(chat.id); else next.delete(chat.id)
              return next
            })} className="h-5 w-5 shrink-0 accent-blue-500" />
            <span className="min-w-0"><span className="block truncate text-sm font-medium">{chat.title}</span>
              <span className="mt-1 block text-xs text-secondary">{chat.chat_type === 'private' ? 'Личная переписка' : chat.chat_type === 'channel' ? 'Канал' : 'Группа'}</span>
            </span>
          </label>)}
          {!visible.length && <p className="py-10 text-center text-sm text-secondary">{chats.length ? 'Таких чатов нет' : 'Нажми «Обновить список», чтобы загрузить чаты.'}</p>}
        </div>
        <div className="mt-7 flex flex-wrap items-center gap-4">
          <button className={button} disabled={busy || (!selected.size && !chats.some(chat => chat.indexing_enabled))} onClick={() => void startIndexing()}>{busy ? 'Запускаем…' : 'Продолжить'}</button>
          <span className="text-xs text-secondary">История загрузится в фоне.</span>
        </div>
      </>}
      {step === 'plugin' && <>
        <h1 className="mb-4 text-4xl font-medium tracking-tight">Теперь — в твоём AI</h1>
        <p className="mb-8 leading-relaxed text-secondary">Подключи Wai Telegram один раз. Помощник сможет искать сообщения, читать контекст и писать по твоему поручению.</p>
        <div className="space-y-7">
          <section>
            <h2 className="mb-3 text-sm font-medium">1. Попроси помощника подключить плагин</h2>
            <button className={button} disabled={busy} onClick={() => void copy(assistantInstructions, 'instruction')}>{copied === 'instruction' ? 'Инструкция скопирована' : 'Скопировать инструкцию для AI'}</button>
          </section>
          <section>
            <h2 className="mb-3 text-sm font-medium">2. Вставь личный ключ в поле подключения</h2>
            {key ? <div className="flex items-center gap-3">
              <input aria-label="Личный ключ плагина" type="password" value={key} readOnly className={`${input} min-w-0 font-mono text-sm`} />
              <button disabled={busy} className="shrink-0 text-sm font-medium text-blue-500" onClick={() => void copy(key, 'key')}>{copied === 'key' ? 'Скопирован' : 'Копировать'}</button>
            </div> : <button className={button} disabled={busy} onClick={() => void perform(async () => {
              const result = await api.createApiKey('Wai Telegram plugin')
              setKey(result.api_key)
            })}>{busy ? 'Создаём…' : 'Создать личный ключ'}</button>}
            <p className="mt-3 text-xs leading-relaxed text-secondary">Ключ вводится в защищённое поле подключения, а не в чат.</p>
          </section>
          <details className="text-sm text-secondary">
            <summary className="cursor-pointer">Адрес для ручного подключения</summary>
            <p className="mt-3 break-all font-mono text-xs">{mcpUrl}</p>
            <p className="mt-2 text-xs">Авторизация: Authorization: Bearer &lt;личный ключ&gt;</p>
          </details>
        </div>
        <div className="mt-10 border-t pt-6 text-sm" aria-live="polite">
          <p className="font-medium">{pluginUsed ? '✓ Плагин подключён' : 'Ждём первого обращения из плагина'}</p>
          {autoSince && <p className="mt-2 leading-relaxed text-secondary">Все чаты · история с {period}. Загрузка идёт автоматически.</p>}
          {coverage?.pause_reason === 'disk_space' && <p className="mt-3 text-secondary">Для продолжения индексации нужно дополнительное место на сервере. Поиск по готовой части истории работает.</p>}
          {totals && <p className="mt-2 leading-relaxed text-secondary">
            {historyReady && totals.indexed === totals.text && !totals.media && !totals.failed ? 'История готова к поиску.' : 'Поиск работает по готовой части истории.'}
            {' '}{totals.indexed.toLocaleString('ru-RU')} из {totals.text.toLocaleString('ru-RU')} загруженных текстов в индексе.
            {totals.media > 0 && ` Вложений в обработке: ${totals.media.toLocaleString('ru-RU')}.`}
            {totals.failed > 0 && ` Не удалось обработать вложений: ${totals.failed.toLocaleString('ru-RU')}.`}
          </p>}
          {coverage?.sync_status === 'failed' && (autoSince ? <p className="mt-3 text-secondary">Загрузка временно прервалась. Повторим автоматически.</p> : <button className="mt-3 text-blue-500" disabled={busy} onClick={() => void startIndexing()}>Продолжить загрузку истории</button>)}
          {!autoSince && <button className="mt-5 text-xs text-secondary underline underline-offset-4" onClick={() => setStep('chats')}>Изменить выбор чатов</button>}
        </div>
      </>}
    </main>
  )
}
