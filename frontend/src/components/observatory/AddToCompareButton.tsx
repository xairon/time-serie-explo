import { GitCompare, Check } from 'lucide-react'
import { useCompareSelection, type StationType } from '@/contexts/CompareSelection'

interface Props {
  code: string
  type: StationType
  /** Compact pill style (for StationDrawer); default is full button (StationPage header). */
  variant?: 'full' | 'compact'
}

export function AddToCompareButton({ code, type, variant = 'full' }: Props) {
  const { has, add, remove, canAdd, blockReason } = useCompareSelection()
  const inSelection = has(code)
  const reason = inSelection ? null : blockReason(type)
  const disabled = !inSelection && !canAdd(type)

  const onClick = () => {
    if (inSelection) remove(code)
    else if (canAdd(type)) add({ code, type })
  }

  const label = inSelection ? 'Dans la comparaison' : 'Comparer'
  const Icon = inSelection ? Check : GitCompare

  if (variant === 'compact') {
    return (
      <button
        onClick={onClick}
        disabled={disabled}
        title={reason ?? label}
        aria-label={label}
        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium transition-colors ${
          inSelection
            ? 'bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30'
            : disabled
              ? 'bg-white/5 text-text-muted/40 cursor-not-allowed'
              : 'bg-white/5 text-text-secondary hover:bg-white/10 hover:text-text-primary'
        }`}
      >
        <Icon className="w-3 h-3" />
        {label}
      </button>
    )
  }

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={reason ?? label}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
        inSelection
          ? 'bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30'
          : disabled
            ? 'bg-white/5 text-text-muted/40 cursor-not-allowed'
            : 'bg-white/5 text-text-secondary hover:bg-white/10 hover:text-text-primary'
      }`}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  )
}
