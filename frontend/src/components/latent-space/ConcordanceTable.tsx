interface ConcordanceTableProps {
  concordance: Record<string, unknown>[]
}

const KEY_LABELS: Record<string, string> = {
  milieu_eh: 'Milieu',
  theme_eh: 'Thème',
  etat_eh: 'État',
  nature_eh: 'Nature',
  departement: 'Département',
  nom_cours_eau: 'Cours d\'eau',
}

function metricColor(value: number): string {
  if (value > 0.3) return 'bg-green-500/20 text-green-400'
  if (value > 0.1) return 'bg-amber-500/20 text-amber-400'
  return 'bg-red-500/20 text-red-400'
}

const TOOLTIPS: Record<string, string> = {
  ari: 'Adjusted Rand Index: agreement between two clusterings, adjusted for chance. 1.0 = perfect, 0.0 = random.',
  nmi: 'Normalized Mutual Information: shared information between clusterings. 1.0 = identical, 0.0 = independent.',
  cramers_v: "Cramér's V: association strength between categorical variables. 1.0 = perfect association, 0.0 = none.",
}

export function ConcordanceTable({ concordance }: ConcordanceTableProps) {
  const rows = concordance as { key: string; ari: number; nmi: number; cramers_v: number }[]

  if (rows.length === 0) return null

  return (
    <div className="bg-bg-card rounded-xl border border-white/5 p-4">
      <h3 className="text-text-primary text-sm font-medium mb-1">
        Concordance with Known Labels
      </h3>
      <p className="text-text-muted text-xs mb-3">How well clusters align with known metadata. Green ({'>'}0.3) = strong agreement, red ({'<'}0.1) = no meaningful alignment.</p>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-text-muted border-b border-white/5">
            <th className="text-left py-2 pr-4">Variable</th>
            {['ARI', 'NMI', "Cramér's V"].map((label, i) => (
              <th key={label} className="text-center py-2 px-2" title={TOOLTIPS[['ari', 'nmi', 'cramers_v'][i]]}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-white/5 last:border-0">
              <td className="py-2 pr-4 text-text-secondary">{KEY_LABELS[row.key] ?? row.key}</td>
              {(['ari', 'nmi', 'cramers_v'] as const).map((metric) => (
                <td key={metric} className="text-center py-2 px-2">
                  <span className={`inline-block px-2 py-0.5 rounded text-xs font-mono ${metricColor(row[metric])}`}>
                    {row[metric].toFixed(3)}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
