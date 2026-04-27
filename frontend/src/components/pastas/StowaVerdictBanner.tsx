import type { StowaResult } from '@/lib/types'

interface Props {
  stowa: StowaResult | null
}

interface CriterionProps {
  label: string
  pass: boolean | null
  detail: string
  tooltip: string
}

function Criterion({ label, pass: criterion, detail, tooltip }: CriterionProps) {
  return (
    <div
      title={tooltip}
      className={`flex-1 min-w-0 flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg border cursor-help ${
        criterion === null
          ? 'bg-white/5 border-white/10 text-text-muted'
          : criterion
            ? 'bg-green-500/10 border-green-500/20 text-green-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
      }`}
    >
      {criterion === null ? (
        <span className="text-text-muted text-[10px]">&mdash;</span>
      ) : criterion ? (
        <span className="text-base leading-none">&#x2713;</span>
      ) : (
        <span className="text-base leading-none">&#x2717;</span>
      )}
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
      tooltip: 'Explained Variance Percentage — how much of the water level variation the model captures. Must be ≥ 70% to pass.',
    },
    {
      label: 'Autocorrelation',
      pass: stowa.autocorrelation_pass,
      detail: stowa.runs_test_pvalue != null ? `p = ${stowa.runs_test_pvalue.toFixed(3)}` : 'Not evaluated',
      tooltip: 'Wald-Wolfowitz Runs Test — checks if model residuals are random (no systematic patterns left). A low p-value means the model is missing some signal.',
    },
    {
      label: stowa.t95_days != null ? `t95 = ${stowa.t95_days.toFixed(0)}d` : 't95',
      pass: stowa.t95_pass,
      detail: stowa.t95_threshold != null ? `threshold ${stowa.t95_threshold.toFixed(0)}d` : 'Not evaluated',
      tooltip: 'Step response time t95 — how many days it takes for the groundwater to reach 95% of its response to a rainfall event. Must be less than half the calibration period.',
    },
    {
      label: 'Gain',
      pass: stowa.gain_pass,
      detail: stowa.gain_significance != null ? `significance ${stowa.gain_significance.toFixed(2)}` : 'Not evaluated',
      tooltip: 'Gain significance — tests whether the recharge parameter (A) is statistically significant. |optimal/stderr| must be > 1.96 (95% confidence level).',
    },
  ]

  return (
    <div className="space-y-2">
      <div className="flex items-stretch gap-2">
        {criteria.map((c) => (
          <Criterion key={c.label} {...c} />
        ))}

        <div
          title="STOWA verdict — Dutch standard (STOWA 2012) for Transfer Function Noise model quality. All 4 criteria must pass for the model to be accepted."
          className={`flex-1 min-w-0 flex flex-col items-center justify-center gap-0.5 px-3 py-2 rounded-lg border font-semibold cursor-help ${
            stowa.overall_pass === null
              ? 'bg-white/5 border-white/10 text-text-muted'
              : stowa.overall_pass
                ? 'bg-green-500/10 border-green-500/20 text-green-400'
                : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
          }`}
        >
          {stowa.overall_pass === null ? (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium border border-white/10 text-text-muted">
              Partiel
            </span>
          ) : (
            <>
              <span className="text-base leading-none">{stowa.overall_pass ? '✓' : '!'}</span>
              <span className="text-xs text-center">
                {stowa.overall_pass ? 'Model accepted' : 'Needs attention'}
              </span>
            </>
          )}
        </div>
      </div>

      {stowa.suggestions.length > 0 && (
        <ul className="space-y-0.5">
          {stowa.suggestions.map((s, i) => (
            <li key={i} className="text-xs text-text-muted leading-snug">
              &bull; {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
