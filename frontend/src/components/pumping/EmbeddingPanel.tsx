export function EmbeddingPanel() {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4 text-center">
      <div className="w-16 h-16 rounded-full bg-accent-cyan/5 border border-white/10 flex items-center justify-center">
        <svg
          className="w-8 h-8 text-text-secondary opacity-40"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <circle cx="6" cy="12" r="2" />
          <circle cx="18" cy="6" r="2" />
          <circle cx="18" cy="18" r="2" />
          <circle cx="12" cy="8" r="1.5" />
          <circle cx="10" cy="16" r="1.5" />
          <line x1="6" y1="12" x2="12" y2="8" strokeWidth="0.8" opacity="0.4" />
          <line x1="6" y1="12" x2="10" y2="16" strokeWidth="0.8" opacity="0.4" />
          <line x1="18" y1="6" x2="12" y2="8" strokeWidth="0.8" opacity="0.4" />
          <line x1="18" y1="18" x2="10" y2="16" strokeWidth="0.8" opacity="0.4" />
        </svg>
      </div>

      <div className="space-y-1">
        <p className="text-text-primary font-medium text-sm">Embedding analysis not available</p>
        <p className="text-text-secondary text-xs max-w-xs">
          The SoftCLT / TS2Vec encoder has not yet been trained on this dataset.
          UMAP visualization will be available after a contrastive pre-training phase.
        </p>
      </div>

      <div
        className="w-48 h-28 rounded-lg bg-bg-primary/40 border border-white/5 flex items-center justify-center opacity-30"
        aria-hidden="true"
      >
        <span className="text-xs text-text-secondary">UMAP — coming soon</span>
      </div>
    </div>
  )
}
