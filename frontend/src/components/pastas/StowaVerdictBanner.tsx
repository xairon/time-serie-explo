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
      detail: stowa.evp_pass ? 'Variance expliquée OK' : 'Variance expliquée faible',
      tooltip: 'Pourcentage de variance expliquée — proportion de la variation du niveau d\'eau capturée par le modèle. Doit être ≥ 70% pour réussir.',
    },
    {
      label: 'Autocorrélation',
      pass: stowa.autocorrelation_pass,
      detail: stowa.runs_test_pvalue != null ? `p = ${stowa.runs_test_pvalue.toFixed(3)}` : 'Non évalué',
      tooltip: 'Test des suites de Wald-Wolfowitz — vérifie si les résidus du modèle sont aléatoires (aucun motif systématique restant). Un p faible signifie que le modèle manque du signal.',
    },
    {
      label: stowa.t95_days != null ? `t95 = ${stowa.t95_days.toFixed(0)} j` : 't95',
      pass: stowa.t95_pass,
      detail: stowa.t95_threshold != null ? `seuil ${stowa.t95_threshold.toFixed(0)} j` : 'Non évalué',
      tooltip: 'Temps de réponse indicielle t95 — nombre de jours pour que la nappe atteigne 95% de sa réponse à un événement pluvieux. Doit être inférieur à la moitié de la période de calibration.',
    },
    {
      label: 'Gain',
      pass: stowa.gain_pass,
      detail: stowa.gain_significance != null ? `significativité ${stowa.gain_significance.toFixed(2)}` : 'Non évalué',
      tooltip: 'Significativité du gain — teste si le paramètre de recharge (A) est statistiquement significatif. |optimal/stderr| doit être > 1,96 (niveau de confiance 95%).',
    },
  ]

  return (
    <div className="space-y-2">
      <div className="flex items-stretch gap-2">
        {criteria.map((c) => (
          <Criterion key={c.label} {...c} />
        ))}

        <div
          title="Verdict STOWA — norme néerlandaise (STOWA 2012) pour la qualité des modèles à fonction de transfert et bruit (TFN). Les 4 critères doivent réussir pour que le modèle soit accepté."
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
                {stowa.overall_pass ? 'Modèle accepté' : 'Attention requise'}
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
