/**
 * UI Component Library - Modern glassmorphism design
 */

import React from 'react'

// Card component with glass effect
export function Card({
    children,
    className = '',
    hover = false,
}: {
    children: React.ReactNode
    className?: string
    hover?: boolean
}) {
    return (
        <div
            className={`
        bg-slate-800/40 backdrop-blur-xl rounded-2xl border border-slate-700/50
        ${hover ? 'hover:border-blue-500/50 hover:bg-slate-800/60 transition-all cursor-pointer' : ''}
        ${className}
      `}
        >
            {children}
        </div>
    )
}

// Button variants
type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({
    children,
    variant = 'primary',
    size = 'md',
    disabled = false,
    loading = false,
    className = '',
    onClick,
    type = 'button',
}: {
    children: React.ReactNode
    variant?: ButtonVariant
    size?: 'sm' | 'md' | 'lg'
    disabled?: boolean
    loading?: boolean
    className?: string
    onClick?: () => void
    type?: 'button' | 'submit'
}) {
    const baseStyles = 'font-medium rounded-xl transition-all flex items-center justify-center gap-2'

    const variants: Record<ButtonVariant, string> = {
        primary: 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 shadow-lg shadow-blue-500/25',
        secondary: 'bg-slate-700/50 text-slate-200 hover:bg-slate-700 border border-slate-600/50',
        ghost: 'bg-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/50',
        danger: 'bg-red-600/20 text-red-400 hover:bg-red-600/30 border border-red-500/30',
    }

    const sizes = {
        sm: 'px-3 py-1.5 text-sm',
        md: 'px-4 py-2',
        lg: 'px-6 py-3 text-lg',
    }

    return (
        <button
            type={type}
            disabled={disabled || loading}
            onClick={onClick}
            className={`
        ${baseStyles}
        ${variants[variant]}
        ${sizes[size]}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        ${className}
      `}
        >
            {loading && (
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
            )}
            {children}
        </button>
    )
}

// Input field
export function Input({
    label,
    type = 'text',
    value,
    onChange,
    placeholder,
    error,
    disabled = false,
    className = '',
}: {
    label?: string
    type?: string
    value: string | number
    onChange: (value: string) => void
    placeholder?: string
    error?: string
    disabled?: boolean
    className?: string
}) {
    return (
        <div className={className}>
            {label && (
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                    {label}
                </label>
            )}
            <input
                type={type}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                disabled={disabled}
                className={`
          w-full px-4 py-2.5 bg-slate-900/50 border rounded-xl
          text-slate-100 placeholder-slate-500
          focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500
          transition-all
          ${error ? 'border-red-500' : 'border-slate-700/50'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
            />
            {error && <p className="mt-1 text-sm text-red-400">{error}</p>}
        </div>
    )
}

// Select dropdown
export function Select({
    label,
    value,
    onChange,
    options,
    placeholder,
    className = '',
}: {
    label?: string
    value: string
    onChange: (value: string) => void
    options: { value: string; label: string }[]
    placeholder?: string
    className?: string
}) {
    return (
        <div className={className}>
            {label && (
                <label className="block text-sm font-medium text-slate-300 mb-1.5">
                    {label}
                </label>
            )}
            <select
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full px-4 py-2.5 bg-slate-900/50 border border-slate-700/50 rounded-xl
          text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500
          transition-all appearance-none cursor-pointer"
            >
                {placeholder && <option value="">{placeholder}</option>}
                {options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                        {opt.label}
                    </option>
                ))}
            </select>
        </div>
    )
}

// Tabs
export function Tabs({
    tabs,
    activeTab,
    onChange,
}: {
    tabs: { id: string; label: string; icon?: React.ReactNode }[]
    activeTab: string
    onChange: (id: string) => void
}) {
    return (
        <div className="flex gap-1 p-1 bg-slate-800/30 rounded-xl">
            {tabs.map((tab) => (
                <button
                    key={tab.id}
                    onClick={() => onChange(tab.id)}
                    className={`
            flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all
            ${activeTab === tab.id
                            ? 'bg-gradient-to-r from-blue-600 to-cyan-600 text-white'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                        }
          `}
                >
                    {tab.icon}
                    {tab.label}
                </button>
            ))}
        </div>
    )
}

// Progress bar
export function Progress({
    value,
    max = 100,
    label,
    showValue = true,
    size = 'md',
}: {
    value: number
    max?: number
    label?: string
    showValue?: boolean
    size?: 'sm' | 'md' | 'lg'
}) {
    const percentage = Math.min((value / max) * 100, 100)
    const heights = { sm: 'h-1.5', md: 'h-2.5', lg: 'h-4' }

    return (
        <div>
            {(label || showValue) && (
                <div className="flex justify-between mb-1.5 text-sm">
                    <span className="text-slate-300">{label}</span>
                    {showValue && <span className="text-slate-400">{percentage.toFixed(0)}%</span>}
                </div>
            )}
            <div className={`w-full bg-slate-800 rounded-full ${heights[size]}`}>
                <div
                    className={`bg-gradient-to-r from-blue-500 to-cyan-500 ${heights[size]} rounded-full transition-all duration-300`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    )
}

// Metric card
export function Metric({
    label,
    value,
    trend,
    icon,
}: {
    label: string
    value: string | number
    trend?: { value: number; positive: boolean }
    icon?: React.ReactNode
}) {
    return (
        <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-700/30">
            <div className="flex items-center justify-between">
                <span className="text-slate-400 text-sm">{label}</span>
                {icon && <span className="text-slate-500">{icon}</span>}
            </div>
            <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-white">{value}</span>
                {trend && (
                    <span className={trend.positive ? 'text-green-400 text-sm' : 'text-red-400 text-sm'}>
                        {trend.positive ? '↑' : '↓'} {Math.abs(trend.value)}%
                    </span>
                )}
            </div>
        </div>
    )
}

// Badge
export function Badge({
    children,
    variant = 'default',
}: {
    children: React.ReactNode
    variant?: 'default' | 'success' | 'warning' | 'error' | 'info'
}) {
    const variants = {
        default: 'bg-slate-700 text-slate-300',
        success: 'bg-green-500/20 text-green-400',
        warning: 'bg-yellow-500/20 text-yellow-400',
        error: 'bg-red-500/20 text-red-400',
        info: 'bg-blue-500/20 text-blue-400',
    }

    return (
        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${variants[variant]}`}>
            {children}
        </span>
    )
}

// Toast notifications
export function Toast({
    message,
    type = 'info',
    onClose,
}: {
    message: string
    type?: 'success' | 'error' | 'warning' | 'info'
    onClose?: () => void
}) {
    const types = {
        success: 'border-green-500/50 bg-green-500/10 text-green-400',
        error: 'border-red-500/50 bg-red-500/10 text-red-400',
        warning: 'border-yellow-500/50 bg-yellow-500/10 text-yellow-400',
        info: 'border-blue-500/50 bg-blue-500/10 text-blue-400',
    }

    return (
        <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${types[type]}`}>
            <span className="flex-1">{message}</span>
            {onClose && (
                <button onClick={onClose} className="text-current opacity-60 hover:opacity-100">
                    ✕
                </button>
            )}
        </div>
    )
}

// Empty state
export function EmptyState({
    icon,
    title,
    description,
    action,
}: {
    icon: React.ReactNode
    title: string
    description: string
    action?: React.ReactNode
}) {
    return (
        <div className="flex flex-col items-center justify-center py-16 text-center">
            <div className="text-6xl mb-6 text-slate-500">{icon}</div>
            <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
            <p className="text-slate-400 max-w-md mb-6">{description}</p>
            {action}
        </div>
    )
}

// Data table (simple version)
export function DataTable({
    columns,
    data,
    maxRows = 100,
}: {
    columns: { key: string; label: string }[]
    data: Record<string, unknown>[]
    maxRows?: number
}) {
    const displayData = data.slice(0, maxRows)

    return (
        <div className="overflow-x-auto">
            <table className="w-full">
                <thead>
                    <tr className="border-b border-slate-700/50">
                        {columns.map((col) => (
                            <th key={col.key} className="px-3 py-1.5 text-left text-sm font-medium text-slate-400">
                                {col.label}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {displayData.map((row, i) => (
                        <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                            {columns.map((col) => (
                                <td key={col.key} className="px-3 py-1.5 text-sm text-slate-300">
                                    {String(row[col.key] ?? '-')}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
            {data.length > maxRows && (
                <p className="text-center text-sm text-slate-500 py-3">
                    Showing {maxRows} of {data.length} rows
                </p>
            )}
        </div>
    )
}
