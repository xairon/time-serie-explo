import { useRef, useEffect, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { useTranslation } from 'react-i18next'

function createPopupContent(title: string, subtitle?: string): HTMLDivElement {
  const div = document.createElement('div')
  const titleEl = document.createElement('div')
  titleEl.style.cssText = 'color:#0e1117;font-size:12px;font-weight:600'
  titleEl.textContent = title
  div.appendChild(titleEl)
  if (subtitle) {
    const subEl = document.createElement('div')
    subEl.style.cssText = 'color:#6b7280;font-size:11px'
    subEl.textContent = subtitle
    div.appendChild(subEl)
  }
  return div
}

interface Props {
  lat: number | null
  lon: number | null
  label: string
  metadata?: Record<string, unknown>
  siblings?: { code_bss: string; lat: number; lon: number; nom_commune?: string }[]
  onSiblingClick?: (codeBss: string) => void
}

export function StationMap({ lat, lon, label, metadata, siblings, onSiblingClick }: Props) {
  const { t } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const [style, setStyle] = useState<'dark' | 'satellite'>('dark')

  useEffect(() => {
    if (!containerRef.current || lat == null || lon == null) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: style === 'dark'
        ? 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
        : 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',
      center: [lon, lat],
      zoom: 10,
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')

    map.on('load', () => {
      const commune = metadata?.nom_commune ? String(metadata.nom_commune) : ''
      const popup = new maplibregl.Popup({ offset: 12, closeButton: false })
        .setDOMContent(createPopupContent(label, commune || undefined))

      new maplibregl.Marker({ color: '#22d3ee' })
        .setLngLat([lon, lat])
        .setPopup(popup)
        .addTo(map)
        .togglePopup()

      if (siblings && siblings.length > 0) {
        for (const sib of siblings) {
          const el = document.createElement('div')
          el.style.cssText = 'width:10px;height:10px;border-radius:50%;background:#a78bfa;border:2px solid #1e1b4b;cursor:pointer'

          const sibPopup = new maplibregl.Popup({ offset: 8, closeButton: false })
            .setDOMContent(createPopupContent(sib.code_bss, sib.nom_commune || undefined))

          new maplibregl.Marker({ element: el })
            .setLngLat([sib.lon, sib.lat])
            .setPopup(sibPopup)
            .addTo(map)

          if (onSiblingClick) {
            el.addEventListener('click', (e) => {
              e.stopPropagation()
              onSiblingClick(sib.code_bss)
            })
          }
        }
      }
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null }
  }, [lat, lon, label, style, siblings, metadata, onSiblingClick])

  if (lat == null || lon == null) return null

  const bdlisaCode = metadata?.codes_bdlisa ? String(metadata.codes_bdlisa) : null

  return (
    <div className="bg-bg-card rounded-lg border border-white/5 overflow-hidden relative">
      <div ref={containerRef} style={{ height: 200 }} />
      <button
        onClick={() => setStyle(s => s === 'dark' ? 'satellite' : 'dark')}
        className="absolute top-2 left-2 bg-bg-card/80 backdrop-blur-sm border border-white/10 rounded px-2 py-1 text-[10px] text-text-secondary hover:text-text-primary transition-colors z-10"
      >
        {style === 'dark' ? t('pastas.map.lightMap') : t('pastas.map.darkMap')}
      </button>
      {bdlisaCode && (
        <div className="absolute bottom-2 left-2 bg-bg-card/80 backdrop-blur-sm border border-white/10 rounded px-2 py-1 text-[10px] text-text-muted z-10">
          BDLISA: {bdlisaCode}
        </div>
      )}
    </div>
  )
}
