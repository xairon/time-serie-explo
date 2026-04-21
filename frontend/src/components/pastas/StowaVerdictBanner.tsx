import type { StowaResult } from '@/lib/types'

interface Props {
  stowa: StowaResult | null
}

interface CriterionProps {
  label: string
  pass: boolean
  detail: string
}

function Criterion({ label, pass, detail }: CriterionProps) {
  return (
    <div
      className={`flex-1 min-w-0 flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg border ${
        pass
          ? 'bg-green-500/10 border-green-500/20 text-green-400'
          : 'bg-red-500/10 border-red-500/20 text-red-400'
      }`}
    >
      <span className="text-base leading-none">{pass ? '✓' : '✗'}</span>
      <span className="text-xs font-medium truncate w-full text-center">{label}</span>
      <span className="text-xs opacity-75 truncate w-full text-center">{detail}</span>
    </div>
  )
}

export function StowaVerdictBanner({ stowa }: Props) {
  if (!stowa) return null

  const criteria: CriterionProps[] = [
    {
      label: `EVP ${stowa.evp_value.toFixed(1)}%`,
      pass: stowa.evp_pass,
      detail: stowa.evp_pass ? 'Explained variance OK' : 'Low explained variance',
    },
    {
      label: 'Autocorrelation',
      pass: stowa.autocorrelation_pass,
      detail: `p = ${stowa.runs_test_pvalue.toFixed(3)}`,
    },
    {
      label: `t95 = ${stowa.t95_days.toFixed(0)}d`,
      pass: stowa.t95_pass,
      detail: `threshold ${stowa.t95_threshold.toFixed(0)}d`,
    },
    {
      label: 'Gain',
      pass: stowa.gain_pass,
      detail: `significance ${stowa.gain_significance.toFixed(2)}`,
    },
  ]

  return (
    <div className="space-y-2">
      <div className="flex items-stretch gap-2">
        {criteria.map((c) => (
          <Criterion key={c.label} {...c} />
        ))}

        <div
          className={`flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-3 py-2 rounded-lg border font-semibold ${
            stowa.overall_pass
              ? 'bg-green-500/10 border-green-500/20 text-green-400'
              : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
          }`}
        >
          <span className="text-base leading-none">{stowa.overall_pass ? '✓' : '!'}</span>
          <span className="text-xs text-center">
            {stowa.overall_pass ? 'Model accepted' : 'Needs attention'}
          </span>
        </div>
      </div>

      {stowa.suggestions.length > 0 && (
        <ul className="space-y-0.5">
          {stowa.suggestions.map((s, i) => (
            <li key={i} className="text-xs text-text-muted leading-snug">
              • {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
