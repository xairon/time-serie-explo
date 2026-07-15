import { useState } from 'react'
import type { ClimatVariable } from '@/lib/climat-colors'
import { isClimatIndexVariable, isClimatDailyVariable } from '@/lib/climat-colors'

/** Extracts ClimatPage's variable/window/month/day state + derived isIndex/isDaily
 *  flags. Pure state — the overlays wiring (data hooks, selected-cell param,
 *  default month/day effects) stays in ClimatPage. */
export function useClimatState() {
  const [variable, setVariable] = useState<ClimatVariable>('spi')
  const [window, setWindow] = useState(3)
  const [month, setMonth] = useState('')
  const [day, setDay] = useState('')
  return {
    variable, setVariable, window, setWindow, month, setMonth, day, setDay,
    isIndex: isClimatIndexVariable(variable),
    isDaily: isClimatDailyVariable(variable),
  }
}
