/**
 * Layout component with sidebar navigation
 */

import React from 'react'
import { useAppStore } from '../store/appStore'
import {
    Database,
    BarChart3,
    BrainCircuit,
    LineChart,
    Settings,
    ChevronLeft,
    ChevronRight,
    Activity,
} from 'lucide-react'

const navItems = [
    { id: 'datasets', name: 'Datasets', icon: Database, description: 'Prepare & explore data' },
    { id: 'training', name: 'Training', icon: BrainCircuit, description: 'Train AI models' },
    { id: 'forecasting', name: 'Forecasting', icon: LineChart, description: 'Generate predictions' },
    { id: 'models', name: 'Models', icon: BarChart3, description: 'Manage models' },
]

interface LayoutProps {
    children: React.ReactNode
}

export function Layout({ children }: LayoutProps) {
    const { activePage, setActivePage, sidebarCollapsed, toggleSidebar, dbConnection, activeJobs } = useAppStore()

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex">
            {/* Sidebar */}
            <aside
                className={`fixed left-0 top-0 h-full bg-slate-900/80 backdrop-blur-xl border-r border-slate-700/50 transition-all z-50
          ${sidebarCollapsed ? 'w-20' : 'w-64'}`}
            >
                {/* Logo */}
                <div className="p-6 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                        <Activity className="w-6 h-6 text-white" />
                    </div>
                    {!sidebarCollapsed && (
                        <div>
                            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                                Junon
                            </h1>
                            <p className="text-xs text-slate-500">Time Series Platform</p>
                        </div>
                    )}
                </div>

                {/* Navigation */}
                <nav className="mt-4 px-3">
                    {navItems.map((item) => {
                        const Icon = item.icon
                        const isActive = activePage === item.id
                        return (
                            <button
                                key={item.id}
                                onClick={() => setActivePage(item.id)}
                                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl mb-1 transition-all group
                  ${isActive
                                        ? 'bg-gradient-to-r from-blue-600/20 to-cyan-600/20 text-blue-400 border border-blue-500/30'
                                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
                                    }`}
                                title={sidebarCollapsed ? item.name : undefined}
                            >
                                <Icon className={`w-5 h-5 ${isActive ? 'text-blue-400' : ''}`} />
                                {!sidebarCollapsed && (
                                    <>
                                        <div className="flex-1 text-left">
                                            <span className="font-medium">{item.name}</span>
                                            <p className="text-xs text-slate-500 group-hover:text-slate-400">{item.description}</p>
                                        </div>
                                        {item.id === 'training' && activeJobs.filter(j => j.status === 'running').length > 0 && (
                                            <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                                        )}
                                    </>
                                )}
                            </button>
                        )
                    })}
                </nav>

                {/* Collapse toggle */}
                <button
                    onClick={toggleSidebar}
                    className="absolute bottom-24 right-0 translate-x-1/2 w-6 h-6 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center text-slate-400 hover:text-white transition-colors"
                >
                    {sidebarCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
                </button>

                {/* Status footer */}
                <div className={`absolute bottom-6 left-0 right-0 px-4 ${sidebarCollapsed ? 'px-2' : 'px-4'}`}>
                    <div className={`bg-slate-800/50 rounded-xl p-3 border border-slate-700/50 ${sidebarCollapsed ? 'text-center' : ''}`}>
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${dbConnection?.connected ? 'bg-green-500' : 'bg-slate-500'}`} />
                            {!sidebarCollapsed && (
                                <span className="text-sm text-slate-400">
                                    {dbConnection?.connected ? `${dbConnection.database}` : 'Not connected'}
                                </span>
                            )}
                        </div>
                        {!sidebarCollapsed && (
                            <p className="text-xs text-slate-500 mt-1">v0.1.0</p>
                        )}
                    </div>
                </div>
            </aside>

            {/* Main content */}
            <main className={`flex-1 min-h-screen transition-all ${sidebarCollapsed ? 'ml-20' : 'ml-64'}`}>
                <div className="p-8">
                    <div className="max-w-7xl mx-auto">
                        {children}
                    </div>
                </div>
            </main>
        </div>
    )
}
