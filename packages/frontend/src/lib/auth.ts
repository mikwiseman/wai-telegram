import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { api } from './api'

interface User {
  id: string
  email: string
}

interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  acceptInvitation: (token: string) => Promise<void>
  completeTelegramLogin: (accessToken: string, refreshToken: string) => void
  logout: () => void
  checkAuth: () => Promise<void>
}

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => {
      // Wire up token refresh callback so API client can update store
      api.setTokenRefreshCallback((accessToken, refreshToken) => {
        set({ accessToken, refreshToken })
      })

      // Wire up auth failure callback to force logout
      api.setAuthFailureCallback(() => {
        api.setAccessToken(null)
        api.setRefreshToken(null)
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
        })
      })

      return {
        user: null,
        accessToken: null,
        refreshToken: null,
        isLoading: true,

        completeTelegramLogin: (accessToken: string, refreshToken: string) => {
          api.setAccessToken(accessToken)
          api.setRefreshToken(refreshToken)
          set({ accessToken, refreshToken })
        },

        acceptInvitation: async (token: string) => {
          const tokens = await api.acceptInvitation(token)
          api.setAccessToken(tokens.access_token)
          api.setRefreshToken(tokens.refresh_token)
          const user = await api.getMe()
          set({ user, accessToken: tokens.access_token, refreshToken: tokens.refresh_token, isLoading: false })
        },

        login: async (email: string, password: string) => {
          const tokens = await api.login(email, password)
          api.setAccessToken(tokens.access_token)
          api.setRefreshToken(tokens.refresh_token)
          const user = await api.getMe()
          set({
            user,
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
          })
        },

        logout: () => {
          api.setAccessToken(null)
          api.setRefreshToken(null)
          set({
            user: null,
            accessToken: null,
            refreshToken: null,
          })
        },

        checkAuth: async () => {
          const { accessToken, refreshToken } = get()
          if (!accessToken) {
            set({ isLoading: false })
            return
          }

          api.setAccessToken(accessToken)
          api.setRefreshToken(refreshToken)

          try {
            const user = await api.getMe()
            set({ user, isLoading: false })
          } catch {
            set({
              user: null,
              accessToken: null,
              refreshToken: null,
              isLoading: false,
            })
          }
        },
      }
    },
    {
      name: 'auth-storage',
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
      }),
      onRehydrateStorage: () => (state) => {
        state?.checkAuth()
      },
    }
  )
)
