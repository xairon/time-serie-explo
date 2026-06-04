import { Activity, ExternalLink, Plus } from 'lucide-react'
import Plot from 'react-plotly.js'
import { useTranslation } from 'react-i18next'
import { usePastasModels, usePastasModel } from '@/hooks/usePastas'
import { Link } from 'react-router-dom'

interface Props { codeBss: string }

export function PastasSection({ codeBss }: Props) {
  const { t, i18n } = useTranslation()
  const localeTag = i18n.language?.startsWith('en') ? 'en-US' : 'fr-FR'
  const { data: models, isLoading } = usePastasModels(codeBss)
  const bestModel = models?.slice().sort((a, b) => (b.nse ?? 0) - (a.nse ?? 0))[0]
  const { data: bestFit } = usePastasModel(bestModel?.run_id ?? null)

  if (isLoading) {
    return (
      <section className="bg-gray-900/50 rounded-xl border border-white/5 p-5 animate-pulse">
        <div className="h-4 bg-white/10 rounded w-1/3 mb-4" />
        <div className="h-24 bg-white/5 rounded" />
      </section>
    )
  }

  if (!bestModel) {
    return (
      <section className="bg-gray-900/50 rounded-xl border border-white/5 p-5 space-y-4">
        <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Activity className="w-4 h-4" />{t('observatory.pastas.title')}
        </h2>
        <div className="flex flex-col items-center justify-center py-6 text-center">
          <p className="text-sm text-gray-400 mb-4">{t('observatory.pastas.noModel')}</p>
          <Link
            to={`/pastas/calibrate?station=${encodeURIComponent(codeBss)}`}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
          >
            <Plus className="w-4 h-4" />
            {t('observatory.pastas.calibrate')}
          </Link>
        </div>
      </section>
    )
  }

  const metrics = [
    { label: 'NSE', value: bestModel.nse, fmt: (v: number) => v.toFixed(3) },
    { label: 'EVP', value: bestModel.evp, fmt: (v: number) => `${v.toFixed(1)}%` },
    { label: 'RMSE', value: bestModel.rmse, fmt: (v: number) => v.toFixed(3) },
  ]
  const valMetrics = bestModel.has_validation ? [
    { label: 'NSE val.', value: bestModel.val_nse, fmt: (v: number) => v.toFixed(3) },
    { label: 'EVP val.', value: bestModel.val_evp, fmt: (v: number) => `${v.toFixed(1)}%` },
  ] : []

  return (
    <section className="bg-gray-900/50 rounded-xl border border-white/5 p-5 space-y-4">
      <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
        <Activity className="w-4 h-4" />{t('observatory.pastas.title')}
      </h2>

      <div className="bg-bg-card border border-white/5 rounded-xl p-4 space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm font-medium text-gray-200">{bestModel.name}</p>
            <p className="text-xs text-gray-500 mt-0.5">
              {t('observatory.pastas.responseRecharge', { response: bestModel.response_type, recharge: bestModel.recharge_type })}
              {bestModel.include_temp && ` · ${t('observatory.pastas.withTemperature')}`}
            </p>
          </div>
          <span className="text-[10px] text-gray-500 font-mono">
            {new Date(bestModel.created_at).toLocaleDateString(localeTag)}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-3">
          {[...metrics, ...valMetrics].map(m => (
            m.value != null && (
              <div key={m.label} className="text-center">
                <p className="text-[10px] text-gray-500 uppercase tracking-wide">{m.label}</p>
                <p className="text-sm font-mono text-gray-200 mt-0.5">{m.fmt(m.value)}</p>
              </div>
            )
          ))}
        </div>

        {(bestFit?.observed?.index?.length ?? 0) > 0 && (
          <div className="pt-1">
            <p className="text-xs text-gray-400 mb-1">{t('observatory.pastas.obsVsModel', 'Observé vs modèle')}</p>
            <Plot
              data={[
                { x: bestFit!.observed.index, y: bestFit!.observed.values, name: 'Observé', type: 'scatter', mode: 'markers', marker: { color: '#9ca3af', size: 2 }, hoverinfo: 'skip' },
                { x: bestFit!.simulated.index, y: bestFit!.simulated.values, name: 'Modèle', type: 'scatter', mode: 'lines', line: { color: '#06b6d4', width: 1.5 } },
              ]}
              layout={{
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                height: 200, margin: { t: 6, r: 10, b: 24, l: 44 },
                font: { color: '#9ca3af', size: 9 },
                showlegend: true, legend: { orientation: 'h', y: 1.18, x: 0 },
                xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
                yaxis: { gridcolor: 'rgba(255,255,255,0.05)', title: { text: 'm NGF' } },
              }}
              config={{ displayModeBar: false }}
              useResizeHandler className="w-full"
            />
          </div>
        )}
        <div className="flex items-center gap-2 pt-2 border-t border-white/5">
          <Link
            to={`/pastas/results?model=${bestModel.run_id}&station=${encodeURIComponent(codeBss)}`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/20 transition-colors"
          >
            <ExternalLink className="w-3 h-3" />
            {t('observatory.pastas.viewInLab')}
          </Link>
          <Link
            to={`/pastas/calibrate?station=${encodeURIComponent(codeBss)}`}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-300 transition-colors"
          >
            <Plus className="w-3 h-3" />
            {t('observatory.pastas.calibrateAnother')}
          </Link>
        </div>
      </div>

      {models && models.length > 1 && (
        <p className="text-[11px] text-gray-500 text-center">
          {t('observatory.pastas.modelsAvailable', { count: models.length })}
        </p>
      )}
    </section>
  )
}
