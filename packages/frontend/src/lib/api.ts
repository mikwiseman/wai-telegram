function configuredApiUrl() {
  return (process.env.NEXT_PUBLIC_API_URL ?? '').trim()
}

export function resolveApiUrl(value: string) {
  const baseUrl = configuredApiUrl()
    || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000')
  try {
    const url = new URL(value, baseUrl)
    return ['http:', 'https:'].includes(url.protocol) ? url.toString() : null
  } catch {
    return null
  }
}

class ApiClient {
  private accessToken: string | null = null
  private refreshToken: string | null = null
  private onTokenRefresh: ((accessToken: string, refreshToken: string) => void) | null = null
  private onAuthFailure: (() => void) | null = null
  private refreshPromise: Promise<boolean> | null = null

  setAccessToken(token: string | null) {
    this.accessToken = token
  }

  setRefreshToken(token: string | null) {
    this.refreshToken = token
  }

  setTokenRefreshCallback(cb: (accessToken: string, refreshToken: string) => void) {
    this.onTokenRefresh = cb
  }

  setAuthFailureCallback(cb: () => void) {
    this.onAuthFailure = cb
  }

  private getBaseUrl() {
    const apiUrl = configuredApiUrl()
    if (apiUrl) return apiUrl
    if (typeof window !== 'undefined') return window.location.origin
    return 'http://localhost:3000'
  }

  private async attemptTokenRefresh(): Promise<boolean> {
    if (!this.refreshToken) return false

    // Deduplicate concurrent refresh attempts
    if (this.refreshPromise) return this.refreshPromise

    this.refreshPromise = (async () => {
      try {
        const response = await fetch(new URL('/api/v1/auth/refresh', this.getBaseUrl()).toString(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: this.refreshToken }),
        })

        if (!response.ok) return false

        const tokens = await response.json()
        this.accessToken = tokens.access_token
        this.refreshToken = tokens.refresh_token
        this.onTokenRefresh?.(tokens.access_token, tokens.refresh_token)
        return true
      } catch {
        return false
      } finally {
        this.refreshPromise = null
      }
    })()

    return this.refreshPromise
  }

  private async request<T>(
    method: string,
    path: string,
    options: { body?: unknown; params?: Record<string, string> } = {}
  ): Promise<T> {
    const url = new URL(path, this.getBaseUrl())
    if (options.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, value)
      })
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (this.accessToken) {
      headers['Authorization'] = `Bearer ${this.accessToken}`
    }

    let response = await fetch(url.toString(), {
      method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    })

    // 401 interceptor: try token refresh once, then retry
    if (response.status === 401 && this.refreshToken) {
      const refreshed = await this.attemptTokenRefresh()
      if (refreshed) {
        headers['Authorization'] = `Bearer ${this.accessToken}`
        response = await fetch(url.toString(), {
          method,
          headers,
          body: options.body ? JSON.stringify(options.body) : undefined,
        })
      } else {
        this.onAuthFailure?.()
      }
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Request failed' }))
      throw new Error(error.detail || 'Request failed')
    }

    if (response.status === 204) {
      return undefined as T
    }

    return response.json()
  }

  // Auth
  async acceptInvitation(token: string) {
    const response = await fetch(new URL('/api/v1/onboarding/accept', this.getBaseUrl()), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
    const body = await response.json()
    if (!response.ok) throw new Error(body.detail || 'Не удалось открыть приглашение')
    return body as { access_token: string; refresh_token: string }
  }

  async selectIndexedChats(chatIds: string[]) {
    return this.request<{ selected: number; job_id: string | null }>('PUT', '/api/v1/indexing/chats', {
      body: { chat_ids: chatIds },
    })
  }

  async getIndexingStatus() {
    return this.request<IndexingStatus>('GET', '/api/v1/indexing/status')
  }

  async login(email: string, password: string) {
    const formData = new URLSearchParams()
    formData.append('username', email)
    formData.append('password', password)

    const response = await fetch(new URL('/api/v1/auth/login', this.getBaseUrl()).toString(), {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Login failed' }))
      throw new Error(error.detail || 'Login failed')
    }

    return response.json()
  }

  async getMe() {
    return this.request<{ id: string; email: string }>(
      'GET',
      '/api/v1/auth/me'
    )
  }

  // API Key Management
  async listApiKeys() {
    return this.request<ApiKeyInfo[]>('GET', '/api/v1/auth/api-keys')
  }

  async createApiKey(name: string, expiresInDays: number | null = 365, scopes: string[] = ['read', 'write']) {
    return this.request<ApiKeyCreateResponse>('POST', '/api/v1/auth/api-keys', {
      body: { name, expires_in_days: expiresInDays, scopes },
    })
  }

  async revokeApiKey(keyId: string) {
    return this.request<void>('DELETE', `/api/v1/auth/api-keys/${keyId}`)
  }

  async toggleApiKey(keyId: string, isActive: boolean) {
    return this.request<ApiKeyInfo>('PATCH', `/api/v1/auth/api-keys/${keyId}`, {
      body: { is_active: isActive },
    })
  }

  async testMcpConnection() {
    return this.request<{ success: boolean; message: string; chat_count?: number; message_count?: number }>(
      'POST',
      '/api/v1/auth/api-keys/test'
    )
  }

  // Telegram
  async requestCode(phone_number: string) {
    return this.request<{ phone_code_hash: string; code_type: string }>(
      'POST',
      '/api/v1/telegram/request-code',
      { body: { phone_number } }
    )
  }

  async verifyCode(
    phone_number: string,
    phone_code_hash: string,
    code: string,
    password?: string
  ) {
    return this.request<{ success: boolean; telegram_user_id: number; access_token?: string | null; refresh_token?: string | null }>(
      'POST',
      '/api/v1/telegram/verify-code',
      { body: { phone_number, phone_code_hash, code, password } }
    )
  }

  async getTelegramSession() {
    return this.request<{ id: string; phone_number: string; is_active: boolean } | null>(
      'GET',
      '/api/v1/telegram/session'
    )
  }

  async deleteTelegramSession() {
    return this.request<{ message: string }>('DELETE', '/api/v1/telegram/session')
  }

  // Chats
  async getChats(chatType?: string, limit = 100, cursor?: string) {
    const params: Record<string, string> = {}
    if (chatType) params.chat_type = chatType
    params.limit = String(limit)
    if (cursor) params.cursor = cursor
    return this.request<ChatListPage>(
      'GET',
      '/api/v1/chats',
      { params }
    )
  }

  async refreshChats() {
    return this.request<ChatListPage>(
      'POST',
      '/api/v1/chats/refresh'
    )
  }

  async getChatMessages(chatId: string, limit = 50, before?: string, after?: string) {
    const params: Record<string, string> = { limit: String(limit) }
    if (before) params.before = before
    if (after) params.after = after
    return this.request<MessageListPage>(
      'GET',
      `/api/v1/chats/${chatId}/messages`,
      { params }
    )
  }

  async getMessageContent(chatId: string, telegramMessageId: number) {
    return this.request<MessageContent>(
      'GET',
      `/api/v1/chats/${chatId}/messages/${telegramMessageId}/content`
    )
  }

  async prepareMessageMedia(chatId: string, telegramMessageId: number) {
    return this.request<MediaPrepareResult>(
      'POST',
      `/api/v1/chats/${chatId}/messages/${telegramMessageId}/prepare`
    )
  }

  async getTranscriptSegments(chatId: string, telegramMessageId: number, cursor = 0) {
    return this.request<TranscriptSegmentPage>(
      'GET',
      `/api/v1/chats/${chatId}/messages/${telegramMessageId}/transcript`,
      { params: { cursor: String(cursor), limit: '100' } }
    )
  }

  // Sync
  async syncChat(chatId: string, limit?: number) {
    const params: Record<string, string> = {}
    if (limit !== undefined) params.limit = String(limit)
    return this.request<SyncJob>('POST', `/api/v1/sync/chats/${chatId}`, { params })
  }

  async syncAll(limitPerChat = 0) {
    return this.request<SyncJob>('POST', '/api/v1/sync/all', {
      params: { limit_per_chat: String(limitPerChat) },
    })
  }

  async getSyncJob(jobId: string) {
    return this.request<SyncJobProgress>('GET', `/api/v1/sync/jobs/${jobId}`)
  }

  async getSyncJobs(limit = 20) {
    return this.request<SyncJob[]>('GET', '/api/v1/sync/jobs', {
      params: { limit: String(limit) },
    })
  }

  // Search
  async search(query: string, chatIds?: string[], limit = 20, cursor?: string) {
    return this.request<{ results: SearchResult[]; total: number; has_more: boolean; next_cursor: string | null }>(
      'POST',
      '/api/v1/search',
      { body: { query, chat_ids: chatIds, limit, cursor } }
    )
  }

  // Digests
  async getDigests(limit = 30) {
    return this.request<Digest[]>('GET', '/api/v1/digests', {
      params: { limit: String(limit) },
    })
  }

  async generateDigest(date?: string) {
    return this.request<Digest>('POST', '/api/v1/digests/generate', {
      body: date ? { date } : {},
    })
  }

  // Settings
  async getSettings() {
    return this.request<UserSettings>('GET', '/api/v1/settings')
  }

  async updateSettings(settings: Partial<UserSettingsUpdate>) {
    return this.request<UserSettings>('PUT', '/api/v1/settings', {
      body: settings,
    })
  }

  async testBot() {
    return this.request<{ success: boolean; message: string }>(
      'POST',
      '/api/v1/settings/test-bot'
    )
  }
}

export const api = new ApiClient()

// Types
export interface Chat {
  indexing_enabled?: boolean
  history_synced_at?: string | null
  id: string
  telegram_chat_id: number
  chat_type: 'private' | 'group' | 'supergroup' | 'channel'
  title: string
  username: string | null
  last_sync_at: string | null
  last_activity_at: string | null
  total_messages_synced: number
  last_message_text: string | null
  last_message_sender_name: string | null
  unread_count: number
}

export interface IndexingStatus {
  pause_reason?: 'disk_space' | null
  chats: { id: string; title: string; history_complete: boolean; messages: number; text_messages: number; indexed_texts: number; pending_media: number; failed_media?: number }[]
  sync_status: string | null
}

export interface Message {
  id: string
  telegram_message_id: number
  text: string | null
  has_media: boolean
  media_type: string | null
  sender_id: number | null
  sender_name: string | null
  is_outgoing: boolean
  sent_at: string
  transcribed_at: string | null
  media_file_name?: string | null
  media_mime_type?: string | null
  media_file_size?: number | null
  media_duration_seconds?: number | null
  content_summary?: string | null
  content_preview?: string | null
  media_processing_status?: string | null
  media_processing_error_code?: string | null
  telegram_message_url?: string | null
  media_download_url?: string | null
  media_cache_status?: string | null
  media_cache_stage?: string | null
  media_cached_bytes?: number | null
  media_sha256?: string | null
  visible_urls?: string[]
  hidden_urls?: string[]
  edited_at?: string | null
  deleted_at?: string | null
}

export interface SearchResult {
  id: string
  chat_id: string
  chat_title: string
  text: string | null
  sender_name: string | null
  is_outgoing: boolean
  sent_at: string
  similarity: number
  has_media: boolean
  media_type: string | null
  transcribed_at: string | null
  telegram_message_id?: number
  content_summary?: string | null
  content_preview?: string | null
  media_file_name?: string | null
  media_mime_type?: string | null
  media_file_size?: number | null
  visible_urls?: string[]
  hidden_urls?: string[]
  deleted_at?: string | null
  telegram_message_url?: string | null
  media_download_url?: string | null
}

export interface MessageContent {
  id: string
  telegram_message_id: number
  text: string | null
  media_type: string | null
  media_file_name: string | null
  media_mime_type: string | null
  media_file_size: number | null
  media_duration_seconds: number | null
  content_text: string | null
  content_summary: string | null
  media_processing_status: string | null
  media_processing_error_code: string | null
  media_processing_error: string | null
  transcribed_at: string | null
  media_processed_at: string | null
  content_model: string | null
  summary_model: string | null
  telegram_message_url: string | null
  media_download_url: string | null
  media_cache_status: string | null
  media_cache_stage: string | null
  media_sha256: string | null
  media_cached_bytes: number | null
  next_action: string | null
}

export interface MediaPrepareResult {
  message_id: string
  status: string
  stage: string
  byte_offset: number
  size_bytes: number | null
  sha256: string | null
  retry_after: string | null
  error_code: string | null
  error_detail: string | null
  media_download_url: string | null
  next_action: string
}

export interface TranscriptSegment {
  sequence: number
  start_ms: number
  end_ms: number
  speaker: string | null
  confidence: number | null
  language: string | null
  text: string
}

export interface TranscriptSegmentPage {
  segments: TranscriptSegment[]
  has_more: boolean
  next_cursor: number | null
}

export interface ChatListPage {
  chats: Chat[]
  has_more: boolean
  next_cursor: string | null
  total: number | null
}

export interface MessageListPage {
  messages: Message[]
  total: number | null
  has_more: boolean
  next_cursor: string | null
  newest_cursor: string | null
  total_messages_synced: number | null
  last_sync_at: string | null
}

export interface SyncJob {
  id: string
  chat_id: string | null
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  messages_processed: number
  error_message?: string | null
}

export interface SyncJobProgress {
  job_id: string
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled'
  messages_processed: number
  current_chat: string | null
  progress_percent: number | null
  error_message: string | null
  retry_after_seconds: number | null
  chats_completed: number | null
  total_chats: number | null
  messages_total: number | null
  messages_seen: number | null
}

export interface Digest {
  id: string
  digest_date: string
  content: string
  summary_stats: Record<string, unknown>
}

export interface UserSettings {
  digest_enabled: boolean
  digest_hour_utc: number
  digest_timezone: string
  digest_telegram_enabled: boolean
  realtime_sync_enabled: boolean
  listener_active: boolean
}

export interface UserSettingsUpdate {
  digest_enabled?: boolean
  digest_hour_utc?: number
  digest_timezone?: string
  digest_telegram_enabled?: boolean
  realtime_sync_enabled?: boolean
}

export interface ApiKeyInfo {
  id: string
  name: string
  key_hint: string
  is_active: boolean
  created_at: string
  last_used_at: string | null
  expires_at: string | null
  scopes: string[]
}

export interface ApiKeyCreateResponse {
  id: string
  name: string
  api_key: string
  key_hint: string
  expires_at: string | null
  scopes: string[]
  message: string
}
