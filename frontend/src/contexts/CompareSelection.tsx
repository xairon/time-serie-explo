import { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react'

export type StationType = 'piezo' | 'hydro'
export interface CompareItem { code: string; type: StationType }

const STORAGE_KEY = 'junon.compare.selection.v1'
const MAX_STATIONS = 5

interface ContextValue {
  items: CompareItem[]
  /** Active comparison type, derived from items. null when empty. */
  type: StationType | null
  /** Number of stations selected. */
  count: number
  /** Add a station. No-op if already present, type-locked, or already at MAX. */
  add: (item: CompareItem) => void
  /** Remove a station by code. */
  remove: (code: string) => void
  /** Reset the selection to empty. */
  clear: () => void
  /** Whether a station is currently in the selection. */
  has: (code: string) => boolean
  /** Whether a station of the given type can still be added. */
  canAdd: (type: StationType) => boolean
  /** Reason `canAdd` returned false (for tooltips). null if it would succeed. */
  blockReason: (type: StationType) => string | null
}

const CompareContext = createContext<ContextValue | null>(null)

function loadFromStorage(): CompareItem[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((x) => x && typeof x.code === 'string' && (x.type === 'piezo' || x.type === 'hydro'))
      .slice(0, MAX_STATIONS)
  } catch {
    return []
  }
}

export function CompareSelectionProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<CompareItem[]>(() => loadFromStorage())

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
    } catch {
      // Storage quota / private mode — selection just won't persist this session.
    }
  }, [items])

  const type: StationType | null = items.length > 0 ? items[0].type : null

  const add = useCallback((item: CompareItem) => {
    setItems((prev) => {
      if (prev.length >= MAX_STATIONS) return prev
      if (prev.some((p) => p.code === item.code)) return prev
      if (prev.length > 0 && prev[0].type !== item.type) return prev
      return [...prev, item]
    })
  }, [])

  const remove = useCallback((code: string) => {
    setItems((prev) => prev.filter((p) => p.code !== code))
  }, [])

  const clear = useCallback(() => setItems([]), [])

  const has = useCallback((code: string) => items.some((p) => p.code === code), [items])

  const canAdd = useCallback((t: StationType) => {
    if (items.length >= MAX_STATIONS) return false
    if (items.length > 0 && items[0].type !== t) return false
    return true
  }, [items])

  const blockReason = useCallback((t: StationType) => {
    if (items.length >= MAX_STATIONS) return `Maximum atteint (${MAX_STATIONS} stations). Retirez-en une avant d'en ajouter.`
    if (items.length > 0 && items[0].type !== t) {
      return `Comparaison ${items[0].type === 'piezo' ? 'piézométrique' : 'hydrométrique'} en cours. Videz-la pour mélanger les types.`
    }
    return null
  }, [items])

  const value = useMemo<ContextValue>(() => ({
    items, type, count: items.length, add, remove, clear, has, canAdd, blockReason,
  }), [items, type, add, remove, clear, has, canAdd, blockReason])

  return <CompareContext.Provider value={value}>{children}</CompareContext.Provider>
}

export function useCompareSelection(): ContextValue {
  const ctx = useContext(CompareContext)
  if (!ctx) throw new Error('useCompareSelection must be used inside <CompareSelectionProvider>')
  return ctx
}

export { MAX_STATIONS }
