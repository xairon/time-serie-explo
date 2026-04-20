import type { DatasetSummary } from '@/lib/types'

interface StationPickerProps {
  datasets: DatasetSummary[]
  datasetId: string
  onDatasetChange: (id: string) => void
  stationId: string
  onStationChange: (id: string) => void
  precipColumn: string
  onPrecipChange: (col: string) => void
  evapColumn: string
  onEvapChange: (col: string) => void
}

export function StationPicker({
  datasets,
  datasetId,
  onDatasetChange,
  stationId,
  onStationChange,
  precipColumn,
  onPrecipChange,
  evapColumn,
  onEvapChange,
}: StationPickerProps) {
  const selectedDataset = datasets.find((d) => d.id === datasetId)
  const stations = selectedDataset?.stations ?? []
  const covariates = selectedDataset?.covariates ?? []
  const isMultiStation = (selectedDataset?.station_column ?? null) !== null && stations.length > 1

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-text-secondary mb-1">Dataset</label>
        <select
          value={datasetId}
          onChange={(e) => onDatasetChange(e.target.value)}
          className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50"
        >
          <option value="">-- Select a dataset --</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {isMultiStation && (
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1">Station</label>
          <select
            value={stationId}
            onChange={(e) => onStationChange(e.target.value)}
            className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50"
          >
            <option value="">-- Select a station --</option>
            {stations.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
      )}

      <div>
        <label className="block text-sm font-medium text-text-secondary mb-1">
          Precipitation column
        </label>
        <select
          value={precipColumn}
          onChange={(e) => onPrecipChange(e.target.value)}
          className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50"
          disabled={!selectedDataset}
        >
          <option value="">-- Select column --</option>
          {covariates.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-text-secondary mb-1">
          Evapotranspiration column
        </label>
        <select
          value={evapColumn}
          onChange={(e) => onEvapChange(e.target.value)}
          className="w-full bg-bg-primary border border-white/10 rounded-md px-3 py-2 text-text-primary text-sm focus:outline-none focus:border-accent-cyan/50"
          disabled={!selectedDataset}
        >
          <option value="">-- Select column --</option>
          {covariates.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
