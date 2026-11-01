import { StrictMode } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ConnectPage from './page'
import { api } from '@/lib/api'

const auth = vi.hoisted(() => ({ user: null as null | { id: string }, isLoading: false, acceptInvitation: vi.fn(), completeTelegramLogin: vi.fn() }))
vi.mock('@/lib/auth', () => ({ useAuth: Object.assign(() => auth, { getState: () => auth }) }))
vi.mock('@/lib/api', () => ({ api: {
  getTelegramSession: vi.fn(), refreshChats: vi.fn(), getChats: vi.fn(), requestCode: vi.fn(),
  verifyCode: vi.fn(), selectIndexedChats: vi.fn(), getIndexingStatus: vi.fn(),
  createApiKey: vi.fn(), listApiKeys: vi.fn(),
} }))

beforeEach(() => {
  vi.resetAllMocks()
  vi.stubGlobal('scrollTo', vi.fn())
  auth.user = null
  window.history.replaceState({}, '', '/connect')
  vi.mocked(api.getTelegramSession).mockResolvedValue(null)
  vi.mocked(api.refreshChats).mockResolvedValue({ chats: [
    { id: 'chat-a', title: 'Мик', chat_type: 'private', indexing_enabled: false },
    { id: 'chat-b', title: 'Школа', chat_type: 'group', indexing_enabled: false },
  ] } as Awaited<ReturnType<typeof api.refreshChats>>)
  vi.mocked(api.requestCode).mockResolvedValue({ phone_code_hash: 'test-hash', code_type: 'app' })
  vi.mocked(api.selectIndexedChats).mockResolvedValue({ selected: 1, job_id: 'job-a' })
  vi.mocked(api.listApiKeys).mockResolvedValue([])
  vi.mocked(api.getIndexingStatus).mockResolvedValue({ chats: [], sync_status: 'pending' })
})

describe('Personal Telegram connection', () => {
  it('uses the invite once, handles Telegram 2FA and only indexes chosen chats', async () => {
    window.history.replaceState({}, '', '/connect#invite=one-use-test-invitation')
    const user = userEvent.setup()
    const clipboard = vi.spyOn(navigator.clipboard, 'writeText')
    render(<StrictMode><ConnectPage /></StrictMode>)
    await screen.findByRole('heading', { name: 'Подключи свой Telegram' })
    expect(auth.acceptInvitation).toHaveBeenCalledExactlyOnceWith('one-use-test-invitation')
    expect(window.location.hash).toBe('')
    expect(screen.queryByLabelText(/email/i)).not.toBeInTheDocument()
    await user.type(screen.getByLabelText('Номер телефона'), '+79991234567')
    await user.click(screen.getByRole('button', { name: 'Получить код' }))
    await user.type(await screen.findByLabelText('Код из Telegram'), '12345')
    vi.mocked(api.verifyCode).mockRejectedValueOnce(new Error('Two-factor authentication is enabled. Password required.'))
    await user.click(screen.getByRole('button', { name: 'Войти' }))
    await user.type(await screen.findByLabelText('Пароль двухэтапной проверки'), 'private-password')
    vi.mocked(api.verifyCode).mockResolvedValueOnce({ success: true, telegram_user_id: 42, access_token: 'full-access', refresh_token: 'full-refresh' })
    await user.click(screen.getByRole('button', { name: 'Войти' }))
    await user.click(await screen.findByRole('checkbox', { name: /Мик/ }))
    expect(auth.completeTelegramLogin).toHaveBeenCalledExactlyOnceWith('full-access', 'full-refresh')
    await user.click(screen.getByRole('button', { name: 'Продолжить' }))
    await screen.findByRole('heading', { name: 'Теперь — в твоём AI' })
    expect(api.selectIndexedChats).toHaveBeenCalledExactlyOnceWith(['chat-a'])
    vi.mocked(api.createApiKey).mockResolvedValue({ api_key: 'wai_private_plugin_key' } as Awaited<ReturnType<typeof api.createApiKey>>)
    await user.click(screen.getByRole('button', { name: 'Создать личный ключ' }))
    expect(await screen.findByLabelText('Личный ключ плагина')).toHaveAttribute('type', 'password')
    await user.click(screen.getByRole('button', { name: 'Скопировать инструкцию для AI' }))
    expect(clipboard).toHaveBeenCalled()
    expect(clipboard.mock.lastCall?.[0]).not.toContain('wai_private_plugin_key')
    expect(clipboard.mock.lastCall?.[0]).not.toContain('private-password')
  })

  it('shows an invitation failure without entering someone else’s account', async () => {
    window.history.replaceState({}, '', '/connect#invite=expired-link')
    auth.acceptInvitation.mockRejectedValue(new Error('Ссылка уже использована'))
    render(<ConnectPage />)
    expect(await screen.findByRole('alert')).toHaveTextContent('Ссылка уже использована')
    expect(api.getTelegramSession).not.toHaveBeenCalled()
  })

  it('requires a personal link when signed out', async () => {
    render(<ConnectPage />)
    await screen.findByRole('heading', { name: 'Начни с приглашения' })
    expect(auth.acceptInvitation).not.toHaveBeenCalled()
  })

  it('keeps existing chat choices on return and shows confirmed plugin use', async () => {
    auth.user = { id: 'guest' }
    vi.mocked(api.getTelegramSession).mockResolvedValue({ id: 'session', phone_number: '+79991234567', is_active: true })
    vi.mocked(api.getChats).mockResolvedValue({ chats: [
      { id: 'chat-a', title: 'Мик', chat_type: 'private', indexing_enabled: true },
    ], has_more: false } as Awaited<ReturnType<typeof api.getChats>>)
    vi.mocked(api.listApiKeys).mockResolvedValue([{ name: 'Wai Telegram plugin', is_active: true, last_used_at: '2026-11-01' }] as Awaited<ReturnType<typeof api.listApiKeys>>)
    vi.mocked(api.getIndexingStatus).mockResolvedValue({ sync_status: 'completed', chats: [
      { id: 'chat-a', title: 'Мик', history_complete: true, messages: 12, text_messages: 10, indexed_texts: 10, pending_media: 0 },
    ] })
    const user = userEvent.setup()
    render(<ConnectPage />)
    expect(await screen.findByRole('checkbox', { name: /Мик/ })).toBeChecked()
    await user.click(screen.getByRole('button', { name: 'Продолжить' }))
    await waitFor(() => expect(screen.getByText('✓ Плагин подключён')).toBeVisible())
    expect(screen.getByText(/История готова к поиску/)).toBeVisible()
  })
})
