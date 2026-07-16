import { useTranslation } from 'react-i18next'
import {
  CLIMAT_VARIABLES, CLIMAT_WINDOWS, DAILY_TEMP_VARIABLE_ORDER,
  isClimatIndexVariable,
} from '@/lib/climat-colors'
import type { ClimatVariable } from '@/lib/climat-colors'
import { InfoTip } from '@/components/pastas/InfoTip'

interface Props {
  variable: ClimatVariable
  onVariableChange: (v: ClimatVariable) => void
  window: number
  onWindowChange: (w: number) => void
}

/** SPI/STI/bilan hydrique are deviations from a 1991-2020 normal — "is this month
 *  unusual?". Ce sont les seules couches mensuelles : les valeurs absolues
 *  (précipitation/température/ETP) sont des CHIFFRES dans le PointPanel, pas des
 *  cartes (cf. spec 2026-07-16 — une carte porte un indicateur, un nombre porte
 *  la valeur). Les journalières ci-dessous font exception : domaine absolu fixe. */
const ANOMALY_VARS: ClimatVariable[] = ['spi', 'sti', 'bilan_hydrique']

/** Variable + window picker for the Climat Situation view. Window selector only
 *  applies to SPI/STI (the raw variables have no rolling-window concept here). */
export function VariablePicker({ variable, onVariableChange, window, onWindowChange }: Props) {
  const { t } = useTranslation()
  const showWindow = isClimatIndexVariable(variable)
  const WINDOW_LABELS: Record<number, string> = {
    1: t('climat.picker.window1'), 3: t('climat.picker.window3'),
    6: t('climat.picker.window6'), 12: t('climat.picker.window12'),
  }

  const renderVariableGroup = (vars: ClimatVariable[], legendKey: string) => (
    <fieldset className="flex flex-wrap items-center gap-1">
      <legend className="text-[10px] text-text-secondary mr-1">{t(legendKey)}</legend>
      <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={t(legendKey)}>
        {vars.map((v) => (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={v === variable}
            onClick={() => onVariableChange(v)}
            className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
              v === variable
                ? 'bg-accent-cyan/20 text-accent-cyan'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
            }`}
          >
            {t(CLIMAT_VARIABLES[v].labelKey)}
          </button>
        ))}
      </div>
    </fieldset>
  )

  return (
    <div className="bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg p-2 shadow-lg flex flex-col gap-2">
      {renderVariableGroup(ANOMALY_VARS, 'climat.picker.familyAnomaly')}
      <div
        className="flex flex-wrap items-center gap-1 border-t border-white/10 pt-2"
        role="radiogroup"
        aria-label={t('climat.picker.dailyTempLabel')}
      >
        <span className="text-[10px] text-text-secondary mr-1 inline-flex items-center gap-1">
          {t('climat.picker.dailyTempLabel')}
          <InfoTip text={t('climat.picker.dailyTempInfo')} />
        </span>
        {DAILY_TEMP_VARIABLE_ORDER.map((v) => (
          <button
            key={v}
            type="button"
            role="radio"
            aria-checked={v === variable}
            onClick={() => onVariableChange(v)}
            className={`text-xs px-2.5 py-1 rounded-md transition-colors ${
              v === variable
                ? 'bg-accent-cyan/20 text-accent-cyan'
                : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
            }`}
          >
            {t(CLIMAT_VARIABLES[v].labelKey)}
          </button>
        ))}
      </div>
      {showWindow && (
        <div className="flex items-center gap-1 border-t border-white/10 pt-2" role="radiogroup" aria-label={t('climat.picker.windowLabel')}>
          <span className="text-[10px] text-text-secondary mr-1">{t('climat.picker.windowLabel')}</span>
          {CLIMAT_WINDOWS.map((w) => (
            <button
              key={w}
              type="button"
              role="radio"
              aria-checked={w === window}
              aria-label={`${w} — ${WINDOW_LABELS[w]}`}
              onClick={() => onWindowChange(w)}
              className={`text-xs px-2 py-0.5 rounded-md transition-colors ${
                w === window
                  ? 'bg-accent-cyan/20 text-accent-cyan'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-hover'
              }`}
            >
              {w}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
