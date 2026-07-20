import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { ClimatMap } from '@/components/climat/ClimatMap'
import { VariablePicker } from '@/components/climat/VariablePicker'
import { MonthStepper } from '@/components/climat/MonthStepper'
import { DayStepper } from '@/components/climat/DayStepper'
import { ClimatLegend } from '@/components/climat/ClimatLegend'
import { PointPanel } from '@/components/climat/PointPanel'
import {
  useClimatGridMonthly, useClimatGridIndices, useClimatRange, useSelectedCellParam,
  useClimatDailyTempRange, useClimatDailyTemp, useClimatDailyPrecip, useClimatDailyPrecipRange,
} from '@/hooks/useClimat'
import { useClimatState } from '@/hooks/useClimatState'
import { CLIMAT_VARIABLES } from '@/lib/climat-colors'
import { resolveDefaultDay } from '@/lib/climat-day-stepper'

/** Climat page — vue Situation (Lot 2, Task B1) + vue Point (Task B2): full-screen map
 *  of SPI/STI or a raw monthly variable, month/window pickers, a territory-wide
 *  synthesis banner, and a Point panel opened by clicking a cell (or via a shareable
 *  ?lat&lon deep link, see useSelectedCellParam). All data comes from the read-only
 *  marts endpoints in api/routers/observatory_climat.py — no client-side stats. */
export default function ClimatPage() {
  const { t } = useTranslation()
  const s = useClimatState()
  const { selectedCell, selectCell, clearSelectedCell } = useSelectedCellParam()

  // Default month + stepper bounds come from the Climat range endpoint, NOT
  // /era5/range (the daily grid): the daily grid's max is the partial current
  // month, which has no SPI/STI yet — landing there by default emptied the map
  // and made the drought banner show a misleading 0%.
  const { data: range } = useClimatRange()
  useEffect(() => {
    if (range?.max_indices_month && !s.month) s.setMonth(range.max_indices_month.slice(0, 7))
  }, [range, s.month])

  // Les deux couches journalières ont des couvertures DIFFÉRENTES (mesuré :
  // température -> 2026-07-10, pluie -> 2026-07-12) : chacune sa plage, sinon le
  // DayStepper masquerait les jours de pluie les plus récents.
  const isPrecipDaily = s.variable === 'precip_daily'
  const { data: tempRange } = useClimatDailyTempRange()
  const { data: precipRange } = useClimatDailyPrecipRange()
  const dailyRange = isPrecipDaily ? precipRange : tempRange
  useEffect(() => {
    if (!dailyRange?.max_date) return
    if (!s.day) {
      s.setDay(resolveDefaultDay(dailyRange.max_date))
      return
    }
    // La plage change quand l'utilisateur bascule pluie <-> température : un jour
    // valide dans l'ancienne plage peut se retrouver hors de la nouvelle (couvertures
    // différentes), auquel cas on le ramène dans les bornes plutôt que de le laisser
    // échoué (carte vide, « Suivant » grisé).
    const maxDate = dailyRange.max_date.slice(0, 10)
    const minDate = dailyRange.min_date?.slice(0, 10)
    if (s.day > maxDate) s.setDay(maxDate)
    else if (minDate && s.day < minDate) s.setDay(minDate)
  }, [dailyRange, s.day])

  const monthlyParam = CLIMAT_VARIABLES[s.variable].monthlyParam
  const dailyParam = CLIMAT_VARIABLES[s.variable].dailyParam
  // SPI/STI never exist past max_indices_month — capping the stepper there
  // prevents users from stepping into a dead-end empty map. Raw variables may
  // still step into the partial current month (max_monthly_month); it gets
  // flagged in the legend via ClimatMonthlyPoint.mois_complet.
  const stepperMaxMonth = s.isIndex ? range?.max_indices_month : range?.max_monthly_month
  const stepperMinMonth = range?.min_month

  const { data: monthlyPoints, isLoading: monthlyLoading } = useClimatGridMonthly(
    s.month, monthlyParam ?? '', !s.isIndex && !s.isDaily && !!s.month,
  )
  const { data: indexPoints, isLoading: indexLoading } = useClimatGridIndices(
    s.month, s.window, s.variable as 'spi' | 'sti', s.isIndex && !!s.month,
  )
  // Les deux couches journalières lisent des tables différentes (mart température
  // vs stg_era5_timeseries) : deux hooks, dont un seul est activé à la fois.
  const { data: tempPoints, isLoading: tempLoading } = useClimatDailyTemp(
    s.day, dailyParam ?? 'tmax', s.isDaily && !isPrecipDaily && !!s.day,
  )
  const { data: precipPoints, isLoading: precipLoading } = useClimatDailyPrecip(
    s.day, s.isDaily && isPrecipDaily,
  )
  const dailyPoints = isPrecipDaily ? precipPoints : tempPoints
  const dailyLoading = isPrecipDaily ? precipLoading : tempLoading

  const gridLoading = s.isDaily ? dailyLoading : s.isIndex ? indexLoading : monthlyLoading
  // At least one cell came back flagged mois_complet=false (the raw-variable
  // grid-monthly response) — the value shown is a partial-month reading.
  const monthIncomplete = !s.isIndex && !s.isDaily && (monthlyPoints ?? []).some((p) => p.mois_complet === false)

  if (!s.month) {
    return <div className="flex items-center justify-center h-full text-text-secondary">{t('common.loading')}</div>
  }

  return (
    <div className="relative h-full">
      <ClimatMap
        variable={s.variable}
        monthlyPoints={monthlyPoints}
        indexPoints={indexPoints}
        dailyPoints={dailyPoints}
        onCellClick={selectCell}
        selectedCell={selectedCell}
      />
      <div className="absolute top-16 left-3 z-10 flex flex-col gap-2">
        <VariablePicker variable={s.variable} onVariableChange={s.setVariable} window={s.window} onWindowChange={s.setWindow} />
        {s.isDaily ? (
          <DayStepper day={s.day} onChange={s.setDay} minDay={dailyRange?.min_date ?? undefined} maxDay={dailyRange?.max_date ?? undefined} />
        ) : (
          <MonthStepper month={s.month} onChange={s.setMonth} minMonth={stepperMinMonth ?? undefined} maxMonth={stepperMaxMonth ?? undefined} />
        )}
      </div>
      <ClimatLegend variable={s.variable} window={s.window} month={s.isDaily ? s.day : s.month} incomplete={monthIncomplete} />
      {gridLoading && (
        <div className="absolute bottom-4 right-3 z-10 bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg px-3 py-1.5 shadow-lg text-[11px] text-text-secondary">
          {t('climat.loadingGrid')}
        </div>
      )}
      {/* `month` suit la période active, comme ClimatLegend juste au-dessus : en mode
          journalier c'est le mois du jour affiché, sinon le mois du MonthStepper. */}
      {selectedCell && (
        <PointPanel
          lat={selectedCell.lat}
          lon={selectedCell.lon}
          month={s.isDaily ? s.day.slice(0, 7) : s.month}
          onClose={clearSelectedCell}
        />
      )}
    </div>
  )
}
