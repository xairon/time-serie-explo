import type { LucideIcon } from 'lucide-react'

interface StatusCardProps {
  label: string
  value: string | number
  icon?: LucideIcon
  status?: 'ok' | 'error' | 'warning' | 'neutral'
}

const statusColors = {
  ok: 'text-accent-green',
  error: 'text-accent-red',
  warning: 'text-accent-amber',
  neutral: 'text-text-primary',
}

export function StatusCard({ label, value, icon: Icon, status = 'neutral' }: StatusCardProps) {
  return (
    <div className="bg-bg-card rounded-xl p-4 border border-white/5">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-text-secondary uppercase tracking-wide">{label}</span>
        {Icon && <Icon className="w-4 h-4 text-text-secondary" />}
      </div>
      <p className={`text-2xl font-bold ${statusColors[status]}`}>{value}</p>
    </div>
  )
}
