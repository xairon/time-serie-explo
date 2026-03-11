import { useMemo } from 'react'
import { classifyZ } from '@/lib/ips'
import type { MonthConcordance } from '@/lib/ips'

interface IPSMonthlyGridProps {
  predDates: string[]
  predValues: number[]
  gtValues: number[]
  refStats: Record<string, [number, number]>
  ipsLabels: Record<string, string>
  ipsColors: Record<string, string>
  label?: string
  concordance?: MonthConcordance[]
}

interface MonthData {
  key: string
  monthLabel: string
  monthNumber: number
  predMean: number
  gtMean: number
  predZ: number | null
  gtZ: number | null
  predClass: string | null
  gtClass: string | null
}

function groupByMonth(
  dates: string[],
  predValues: number[],
  gtValues: number[],
  refStats: Record<string, [number, number]>,
): MonthData[] {
  const monthMap = new Map<
    string,
    { preds: number[]; gts: number[]; monthNumber: number }
  >()

  for (let i = 0; i < dates.length; i++) {
    const d = new Date(dates[i])
    const yearMonth = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    const monthNumber = d.getMonth() + 1

    if (!monthMap.has(yearMonth)) {
      monthMap.set(yearMonth, { preds: [], gts: [], monthNumber })
    }
    const entry = monthMap.get(yearMonth)!
    if (!isNaN(predValues[i])) entry.preds.push(predValues[i])
    if (!isNaN(gtValues[i])) entry.gts.push(gtValues[i])
  }

  const MONTH_NAMES = [
    '', 'Jan', 'Fev', 'Mar', 'Avr', 'Mai', 'Jun',
    'Jul', 'Aou', 'Sep', 'Oct', 'Nov', 'Dec',
  ]

  const result: MonthData[] = []

  for (const [key, { preds, gts, monthNumber }] of monthMap) {
    const year = key.split('-')[0]
    const monthLabel = `${MONTH_NAMES[monthNumber]} ${year}`

    const predMean = preds.length > 0 ? preds.reduce((a, b) => a + b, 0) / preds.length : NaN
    const gtMean = gts.length > 0 ? gts.reduce((a, b) => a + b, 0) / gts.length : NaN

    const stats = refStats[String(monthNumber)]
    let predZ: number | null = null
    let gtZ: number | null = null
    let predClass: string | null = null
    let gtClass: string | null = null

    if (stats && !isNaN(predMean)) {
      const [mu, sigma] = stats
      if (sigma > 0) {
        predZ = (predMean - mu) / sigma
        predClass = classifyZ(predZ)
      }
    }

    if (stats && !isNaN(gtMean)) {
      const [mu, sigma] = stats
      if (sigma > 0) {
        gtZ = (gtMean - mu) / sigma
        gtClass = classifyZ(gtZ)
      }
    }

    result.push({ key, monthLabel, monthNumber, predMean, gtMean, predZ, gtZ, predClass, gtClass })
  }

  return result
}

function IPSBadge({
  cls,
  ipsLabels,
  ipsColors,
}: {
  cls: string
  ipsLabels: Record<string, string>
  ipsColors: Record<string, string>
}) {
  const color = ipsColors[cls] || '#888'
  const label = ipsLabels[cls] || cls

  return (
    <span
      className="px-2 py-0.5 rounded-full text-xs font-medium"
      style={{
        backgroundColor: `${color}33`,
        color: color,
      }}
    >
      {label}
    </span>
  )
}

export default function IPSMonthlyGrid({
  predDates,
  predValues,
  gtValues,
  refStats,
  ipsLabels,
  ipsColors,
  label,
  concordance,
}: IPSMonthlyGridProps) {
  const months = useMemo(
    () => groupByMonth(predDates, predValues, gtValues, refStats),
    [predDates, predValues, gtValues, refStats],
  )

  const concordanceMap = useMemo(() => {
    if (!concordance) return null
    return new Map(concordance.map((c) => [c.yearMonth, c]))
  }, [concordance])

  if (months.length === 0) return null

  return (
    <div>
      {label && (
        <h4 className="text-sm font-medium text-text-secondary mb-2">{label}</h4>
      )}
      <div className="flex gap-2 overflow-x-auto pb-1">
        {months.map((m) => (
          <div
            key={m.key}
            className="bg-bg-card rounded-lg border border-white/5 p-3 flex-shrink-0 min-w-[150px] flex flex-col items-center gap-1.5"
          >
            <span className="text-xs text-text-secondary">{m.monthLabel}</span>

            <div className="flex gap-3 w-full justify-center">
              {/* GT column */}
              {m.gtClass && (
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-[10px] font-medium text-text-secondary">Observe</span>
                  <IPSBadge cls={m.gtClass} ipsLabels={ipsLabels} ipsColors={ipsColors} />
                  {m.gtZ !== null && (
                    <span className="text-[10px] text-text-secondary">
                      z = {m.gtZ.toFixed(2)}
                    </span>
                  )}
                </div>
              )}

              {/* Pred column */}
              {m.predClass && (
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-[10px] font-medium text-text-secondary">Modele</span>
                  <IPSBadge cls={m.predClass} ipsLabels={ipsLabels} ipsColors={ipsColors} />
                  {m.predZ !== null && (
                    <span className="text-[10px] text-text-secondary">
                      z = {m.predZ.toFixed(2)}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Concordance indicator */}
            {concordanceMap?.has(m.key) && (() => {
              const c = concordanceMap.get(m.key)!
              const icon = c.status === 'exact' ? '\u2713' : c.status === 'adjacent' ? '\u2248' : '\u2717'
              const color = c.status === 'exact' ? 'text-emerald-400' : c.status === 'adjacent' ? 'text-amber-400' : 'text-red-400'
              return <span className={`text-xs font-bold ${color}`}>{icon}</span>
            })()}
          </div>
        ))}
      </div>
    </div>
  )
}
