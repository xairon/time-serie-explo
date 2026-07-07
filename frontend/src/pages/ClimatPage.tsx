import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { ClimatMap } from '@/components/climat/ClimatMap'
import { VariablePicker } from '@/components/climat/VariablePicker'
import { MonthStepper } from '@/components/climat/MonthStepper'
import { ClimatLegend } from '@/components/climat/ClimatLegend'
import { SituationBanner } from '@/components/climat/SituationBanner'
import { PointPanel } from '@/components/climat/PointPanel'
import { useClimatGridMonthly, useClimatGridIndices, useClimatSituationSummary, useClimatRange, useSelectedCellParam } from '@/hooks/useClimat'
import { CLIMAT_VARIABLES, isClimatIndexVariable } from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'

/** Climat page — vue Situation (Lot 2, Task B1) + vue Point (Task B2): full-screen map
 *  of SPI/STI or a raw monthly variable, month/window pickers, a territory-wide
 *  synthesis banner, and a Point panel opened by clicking a cell (or via a shareable
 *  ?lat&lon deep link, see useSelectedCellParam). All data comes from the read-only
 *  marts endpoints in api/routers/observatory_climat.py — no client-side stats. */
export default function ClimatPage() {
  const { t } = useTranslation()
  const [variable, setVariable] = useState<ClimatVariable>('spi')
  const [window, setWindow] = useState(3)
  const [month, setMonth] = useState<string>('')
  const { selectedCell, selectCell, clearSelectedCell } = useSelectedCellParam()

  // Default month + stepper bounds come from the Climat range endpoint, NOT
  // /era5/range (the daily grid): the daily grid's max is the partial current
  // month, which has no SPI/STI yet — landing there by default emptied the map
  // and made the drought banner show a misleading 0%.
  const { data: range } = useClimatRange()
  useEffect(() => {
    if (range?.max_indices_month && !month) setMonth(range.max_indices_month.slice(0, 7))
  }, [range, month])

  const isIndex = isClimatIndexVariable(variable)
  const monthlyParam = CLIMAT_VARIABLES[variable].monthlyParam
  // SPI/STI never exist past max_indices_month — capping the stepper there
  // prevents users from stepping into a dead-end empty map. Raw variables may
  // still step into the partial current month (max_monthly_month); it gets
  // flagged in the legend via ClimatMonthlyPoint.mois_complet.
  const stepperMaxMonth = isIndex ? range?.max_indices_month : range?.max_monthly_month
  const stepperMinMonth = range?.min_month

  const { data: monthlyPoints, isLoading: monthlyLoading } = useClimatGridMonthly(
    month, monthlyParam ?? '', !isIndex && !!month,
  )
  const { data: indexPoints, isLoading: indexLoading } = useClimatGridIndices(
    month, window, variable as 'spi' | 'sti', isIndex && !!month,
  )
  const { data: summary, isLoading: summaryLoading } = useClimatSituationSummary(month, window, !!month)

  const gridLoading = isIndex ? indexLoading : monthlyLoading
  // At least one cell came back flagged mois_complet=false (the raw-variable
  // grid-monthly response) — the value shown is a partial-month reading.
  const monthIncomplete = !isIndex && (monthlyPoints ?? []).some((p) => p.mois_complet === false)

  if (!month) {
    return <div className="flex items-center justify-center h-full text-text-secondary">{t('common.loading')}</div>
  }

  return (
    <div className="relative h-full">
      <ClimatMap
        variable={variable}
        monthlyPoints={monthlyPoints}
        indexPoints={indexPoints}
        onCellClick={selectCell}
        selectedCell={selectedCell}
      />
      <SituationBanner summary={summary} isLoading={summaryLoading} />
      <div className="absolute top-16 left-3 z-10 flex flex-col gap-2">
        <VariablePicker variable={variable} onVariableChange={setVariable} window={window} onWindowChange={setWindow} />
        <MonthStepper month={month} onChange={setMonth} minMonth={stepperMinMonth ?? undefined} maxMonth={stepperMaxMonth ?? undefined} />
      </div>
      <ClimatLegend variable={variable} window={window} month={month} incomplete={monthIncomplete} />
      {gridLoading && (
        <div className="absolute bottom-4 right-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-1.5 shadow-lg text-[11px] text-text-secondary">
          {t('climat.loadingGrid')}
        </div>
      )}
      {selectedCell && <PointPanel lat={selectedCell.lat} lon={selectedCell.lon} onClose={clearSelectedCell} />}
    </div>
  )
}
