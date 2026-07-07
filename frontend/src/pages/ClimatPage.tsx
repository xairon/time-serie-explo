import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { ClimatMap } from '@/components/climat/ClimatMap'
import { VariablePicker } from '@/components/climat/VariablePicker'
import { MonthStepper } from '@/components/climat/MonthStepper'
import { ClimatLegend } from '@/components/climat/ClimatLegend'
import { SituationBanner } from '@/components/climat/SituationBanner'
import { useClimatGridMonthly, useClimatGridIndices, useClimatSituationSummary } from '@/hooks/useClimat'
import { useERA5Range } from '@/hooks/useObservatory'
import { CLIMAT_VARIABLES, isClimatIndexVariable } from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'

/** Climat page — vue Situation (Lot 2, Task B1): full-screen map of SPI/STI or a raw
 *  monthly variable, month/window pickers, and a territory-wide synthesis banner.
 *  All data comes from the read-only marts endpoints in api/routers/observatory_climat.py
 *  (fct_era5_monthly_grid / fct_era5_indices_grid) — no client-side stats. */
export default function ClimatPage() {
  const { t } = useTranslation()
  const [variable, setVariable] = useState<ClimatVariable>('spi')
  const [window, setWindow] = useState(3)
  const [month, setMonth] = useState<string>('')

  // The Climat marts share the same ERA5 grid coverage as the Observatory overlay,
  // so its /era5/range endpoint gives valid month bounds without a dedicated one.
  const { data: range } = useERA5Range(true)
  useEffect(() => { if (range?.max_date && !month) setMonth(range.max_date.slice(0, 7)) }, [range, month])

  const isIndex = isClimatIndexVariable(variable)
  const monthlyParam = CLIMAT_VARIABLES[variable].monthlyParam

  const { data: monthlyPoints, isLoading: monthlyLoading } = useClimatGridMonthly(
    month, monthlyParam ?? '', !isIndex && !!month,
  )
  const { data: indexPoints, isLoading: indexLoading } = useClimatGridIndices(
    month, window, variable as 'spi' | 'sti', isIndex && !!month,
  )
  const { data: summary, isLoading: summaryLoading } = useClimatSituationSummary(month, window, !!month)

  const gridLoading = isIndex ? indexLoading : monthlyLoading

  if (!month) {
    return <div className="flex items-center justify-center h-full text-text-secondary">{t('common.loading')}</div>
  }

  return (
    <div className="relative h-full">
      <ClimatMap variable={variable} window={window} monthlyPoints={monthlyPoints} indexPoints={indexPoints} />
      <SituationBanner summary={summary} isLoading={summaryLoading} />
      <div className="absolute top-16 left-3 z-10 flex flex-col gap-2">
        <VariablePicker variable={variable} onVariableChange={setVariable} window={window} onWindowChange={setWindow} />
        <MonthStepper month={month} onChange={setMonth} minMonth={range?.min_date} maxMonth={range?.max_date} />
      </div>
      <ClimatLegend variable={variable} window={window} month={month} />
      {gridLoading && (
        <div className="absolute bottom-4 right-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-1.5 shadow-lg text-[11px] text-text-secondary">
          {t('climat.loadingGrid')}
        </div>
      )}
    </div>
  )
}
