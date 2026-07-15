import { describe, it, expect } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useClimatState } from './useClimatState'

describe('useClimatState', () => {
  it('défaut = spi, fenêtre 3, isIndex vrai', () => {
    const { result } = renderHook(() => useClimatState())
    expect(result.current.variable).toBe('spi')
    expect(result.current.window).toBe(3)
    expect(result.current.isIndex).toBe(true)
    expect(result.current.isDaily).toBe(false)
  })
  it('passer à tmax rend isDaily vrai', () => {
    const { result } = renderHook(() => useClimatState())
    act(() => result.current.setVariable('tmax'))
    expect(result.current.isDaily).toBe(true)
    expect(result.current.isIndex).toBe(false)
  })
})
