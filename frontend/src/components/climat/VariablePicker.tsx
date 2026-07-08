import { useTranslation } from 'react-i18next'
import {
  CLIMAT_VARIABLE_ORDER, CLIMAT_VARIABLES, CLIMAT_WINDOWS, DAILY_TEMP_VARIABLE_ORDER,
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

/** Variable + window picker for the Climat Situation view. Window selector only
 *  applies to SPI/STI (the raw variables have no rolling-window concept here). */
export function VariablePicker({ variable, onVariableChange, window, onWindowChange }: Props) {
  const { t } = useTranslation()
  const showWindow = isClimatIndexVariable(variable)

  return (
    <div className="bg-bg-card/90 backdrop-blur-md border border-white/10 rounded-lg p-2 shadow-lg flex flex-col gap-2">
      <div className="flex flex-wrap gap-1" role="radiogroup" aria-label={t('climat.picker.variableLabel')}>
        {CLIMAT_VARIABLE_ORDER.map((v) => (
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
