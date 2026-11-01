import { describe, it, expect, vi } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { Providers } from '../Providers'
import { useAuth } from '@/lib/auth'
import { useQueryClient } from '@tanstack/react-query'

vi.mock('@/lib/auth', async () => {
  const { create } = await import('zustand')
  return { useAuth: create(() => ({ user: null })) }
})

describe('Providers', () => {
  it('renders children', () => {
    render(
      <Providers>
        <div data-testid="child">Hello</div>
      </Providers>
    )
    expect(screen.getByTestId('child')).toBeInTheDocument()
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})


it('discards cached messages when the account changes', () => {
  function CachedMessage() {
    const cache = useQueryClient()
    const message = cache.getQueryData<string>(['messages'])
    if (!message) cache.setQueryData(['messages'], 'private from first account')
    return <div data-testid="cached-message">{message ?? 'empty'}</div>
  }
  act(() => useAuth.setState({ user: { id: 'owner', email: 'owner@example.test' } }))
  const page = render(<Providers><CachedMessage /></Providers>)
  page.rerender(<Providers><CachedMessage /></Providers>)
  expect(screen.getByTestId('cached-message')).toHaveTextContent('private from first account')
  act(() => useAuth.setState({ user: { id: 'guest', email: 'guest@example.test' } }))
  expect(screen.getByTestId('cached-message')).toHaveTextContent('empty')
  act(() => useAuth.setState({ user: null }))
})
