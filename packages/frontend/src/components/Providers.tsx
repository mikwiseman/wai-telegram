'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ThemeProvider } from 'next-themes'
import { useState } from 'react'
import { useAuth } from '@/lib/auth'

function AccountQueries({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            retry: 1,
          },
        },
      })
  )

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}

export function Providers({ children }: { children: React.ReactNode }) {
  const userId = useAuth(state => state.user?.id)
  return <ThemeProvider attribute="class" defaultTheme="light" disableTransitionOnChange>
    <AccountQueries key={userId ?? 'signed-out'}>{children}</AccountQueries>
  </ThemeProvider>
}
