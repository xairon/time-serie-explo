import { useState, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import { api } from '@/lib/api'

export function ImportCSVForm() {
  const { t } = useTranslation()
  const qc = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [name, setName] = useState('')
  const [dragActive, setDragActive] = useState(false)

  const mutation = useMutation({
    mutationFn: (body: FormData) => api.datasets.create(body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['datasets'] })
      setFile(null)
      setName('')
    },
  })

  const handleFile = (f: File) => {
    setFile(f)
    if (!name) setName(f.name.replace(/\.[^.]+$/, ''))
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    const fd = new FormData()
    fd.append('name', name)
    fd.append('source', 'csv')
    fd.append('file', file)
    mutation.mutate(fd)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <h3 className="text-sm font-semibold text-text-primary">{t('sharedComponents.import.csvTitle')}</h3>

      <div>
        <label className="block text-xs text-text-secondary mb-1">{t('sharedComponents.import.datasetName')}</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          required
          className="w-full bg-bg-input text-text-primary border border-white/10 rounded-lg px-3 py-2 text-sm"
          placeholder={t('sharedComponents.import.datasetNamePlaceholder')}
        />
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragActive
            ? 'border-accent-cyan bg-accent-cyan/5'
            : 'border-white/10 hover:border-white/20'
        }`}
      >
        <Upload className="w-8 h-8 text-text-secondary mx-auto mb-2" />
        {file ? (
          <p className="text-sm text-text-primary">{file.name}</p>
        ) : (
          <p className="text-sm text-text-secondary">
            {t('sharedComponents.import.dragDrop')}
          </p>
        )}
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.tsv"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) handleFile(f)
          }}
        />
      </div>

      <button
        type="submit"
        disabled={mutation.isPending || !file}
        className="w-full bg-accent-cyan text-white px-4 py-2 rounded-lg hover:bg-accent-cyan/80 disabled:opacity-50 transition-colors text-sm font-medium"
      >
        {mutation.isPending ? t('sharedComponents.import.uploading') : t('sharedComponents.import.importCsv')}
      </button>

      {mutation.isError && (
        <p className="text-xs text-accent-red">{t('sharedComponents.import.errorPrefix')} : {(mutation.error as Error).message}</p>
      )}
      {mutation.isSuccess && (
        <p className="text-xs text-accent-green">{t('sharedComponents.import.importSuccess')}</p>
      )}
    </form>
  )
}
