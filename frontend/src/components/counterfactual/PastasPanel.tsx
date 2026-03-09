interface PastasPanelProps {
  className?: string
}

export function PastasPanel({ className = '' }: PastasPanelProps) {
  return (
    <div className={`space-y-4 ${className}`}>
      <h4 className="text-sm font-semibold text-text-primary">Validation Pastas</h4>

      <p className="text-xs text-text-secondary">
        La validation Pastas compare les prédictions du modèle TFT avec un modèle
        hydrogéologique Pastas calibré indépendamment.
      </p>

      <div className="bg-bg-card rounded-xl border border-white/5 p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-text-secondary">Statut</span>
          <span className="text-xs text-text-primary italic">Non lancé</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-text-secondary">Corrélation TFT vs Pastas</span>
          <span className="text-xs text-text-primary">—</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-text-secondary">RMSE comparatif</span>
          <span className="text-xs text-text-primary">—</span>
        </div>
      </div>

      <button
        disabled
        className="w-full bg-bg-hover text-text-primary px-4 py-2 rounded-lg border border-white/10 text-sm opacity-50"
      >
        Lancer la validation Pastas (nécessite un contrefactuel)
      </button>
    </div>
  )
}
