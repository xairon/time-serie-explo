interface Props {
  valSplit: number | null
  onChange: (v: number | null) => void
}

export function CalValToggle({ valSplit, onChange }: Props) {
  const enabled = valSplit !== null
  const pct = (valSplit ?? 0.3) * 100

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <label className="text-sm font-medium text-text-secondary">Validation</label>
          <span className="ml-1 text-text-muted cursor-help" title="Réserver une partie des données pour tester le modèle sur des données non vues pendant la calibration. Vérifie que le modèle généralise bien.">ⓘ</span>
        </div>
        <button
          onClick={() => onChange(enabled ? null : 0.3)}
          className={`text-xs px-2 py-0.5 rounded-full border transition-colors ${
            enabled
              ? 'border-accent-cyan text-accent-cyan bg-accent-cyan/10'
              : 'border-white/10 text-text-muted'
          }`}
        >
          {enabled ? 'Activée' : 'Désactivée'}
        </button>
      </div>
      {!enabled && (
        <p className="text-xs text-text-muted">
          Le modèle sera calibré sur la période complète. Activez pour réserver une portion pour les tests.
        </p>
      )}
      {enabled && (
        <div>
          <input
            type="range"
            min={10}
            max={50}
            step={5}
            value={pct}
            onChange={(e) => onChange(+e.target.value / 100)}
            className="w-full accent-accent-cyan"
          />
          <div className="flex justify-between text-xs text-text-muted">
            <span>Calibration : {(100 - pct).toFixed(0)}% (premières années)</span>
            <span>Test : {pct.toFixed(0)}% (dernières années)</span>
          </div>
          <p className="text-xs text-text-muted mt-1">
            Les métriques de test montrent la qualité du modèle sur des données non vues.
          </p>
        </div>
      )}
    </div>
  )
}
