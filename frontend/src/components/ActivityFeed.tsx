'use client'

import React, { useState, useEffect, useCallback } from 'react'

interface ActivityItem {
  id: string
  type: 'query' | 'decision' | 'biomarker' | 'treatment' | 'alert' | 'guideline' | 'system'
  title: string
  description?: string
  timestamp: Date
  metadata?: Record<string, unknown>
  status?: 'info' | 'success' | 'warning' | 'error'
  relatedPatientId?: string
}

interface ActivityFeedProps {
  activities?: ActivityItem[]
  maxItems?: number
  onActivityClick?: (activity: ActivityItem) => void
  onClearAll?: () => void
  showFilters?: boolean
  className?: string
  compact?: boolean
}

export function ActivityFeed({
  activities: externalActivities,
  maxItems = 50,
  onActivityClick,
  onClearAll,
  showFilters = true,
  className = '',
  compact = false
}: ActivityFeedProps) {
  const [activities, setActivities] = useState<ActivityItem[]>(externalActivities || [])
  const [filter, setFilter] = useState<string>('all')
  const [isExpanded, setIsExpanded] = useState(!compact)

  // Sync with external activities
  useEffect(() => {
    if (externalActivities) {
      setActivities(externalActivities.slice(0, maxItems))
    }
  }, [externalActivities, maxItems])

  // Add new activity
  const addActivity = useCallback((activity: Omit<ActivityItem, 'id' | 'timestamp'>) => {
    const newActivity: ActivityItem = {
      ...activity,
      id: `act-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    }

    setActivities(prev => [newActivity, ...prev].slice(0, maxItems))
  }, [maxItems])

  // Clear all activities
  const handleClearAll = () => {
    setActivities([])
    onClearAll?.()
  }

  // Filter activities
  const filteredActivities = filter === 'all'
    ? activities
    : activities.filter(a => a.type === filter)

  // Get icon for activity type
  const getActivityIcon = (type: ActivityItem['type']) => {
    switch (type) {
      case 'query':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        )
      case 'decision':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
      case 'biomarker':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        )
      case 'treatment':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
          </svg>
        )
      case 'alert':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        )
      case 'guideline':
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        )
      case 'system':
      default:
        return (
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        )
    }
  }

  // Get color classes for activity type
  const getActivityColors = (type: ActivityItem['type'], status?: ActivityItem['status']) => {
    if (status === 'error') return 'bg-red-500/20 text-red-400 border-red-500/30'
    if (status === 'warning') return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
    if (status === 'success') return 'bg-green-500/20 text-green-400 border-green-500/30'

    switch (type) {
      case 'query': return 'bg-blue-500/20 text-blue-400 border-blue-500/30'
      case 'decision': return 'bg-purple-500/20 text-purple-400 border-purple-500/30'
      case 'biomarker': return 'bg-green-500/20 text-green-400 border-green-500/30'
      case 'treatment': return 'bg-violet-500/20 text-violet-400 border-violet-500/30'
      case 'alert': return 'bg-orange-500/20 text-orange-400 border-orange-500/30'
      case 'guideline': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
      case 'system':
      default: return 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30'
    }
  }

  // Format relative time
  const formatRelativeTime = (date: Date) => {
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    const seconds = Math.floor(diff / 1000)
    const minutes = Math.floor(seconds / 60)
    const hours = Math.floor(minutes / 60)
    const days = Math.floor(hours / 24)

    if (seconds < 60) return 'just now'
    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    if (days < 7) return `${days}d ago`
    return date.toLocaleDateString()
  }

  // Add mock activities for demo
  useEffect(() => {
    if (!externalActivities && activities.length === 0) {
      const mockActivities: Omit<ActivityItem, 'id' | 'timestamp'>[] = [
        { type: 'decision', title: 'Treatment Recommendation', description: 'Osimertinib recommended for EGFR+ patient', status: 'success' },
        { type: 'biomarker', title: 'Biomarker Result', description: 'EGFR L858R mutation detected', status: 'info' },
        { type: 'query', title: 'Patient Query', description: 'Retrieved treatment history', status: 'info' },
        { type: 'guideline', title: 'Guideline Check', description: 'NCCN guideline applied', status: 'info' },
        { type: 'alert', title: 'Drug Interaction', description: 'Check concurrent medications', status: 'warning' }
      ]

      const now = Date.now()
      const newActivities = mockActivities.map((a, i) => ({
        ...a,
        id: `mock-${i}`,
        timestamp: new Date(now - i * 300000) // 5 min apart
      }))

      setActivities(newActivities)
    }
  }, [externalActivities, activities.length])

  if (compact && !isExpanded) {
    return (
      <button
        onClick={() => setIsExpanded(true)}
        className={`flex items-center gap-2 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg hover:bg-zinc-700 transition-colors ${className}`}
      >
        <svg className="w-4 h-4 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-sm text-zinc-300">Activity</span>
        {activities.length > 0 && (
          <span className="px-1.5 py-0.5 bg-violet-500/20 text-violet-400 text-xs rounded-full">
            {activities.length}
          </span>
        )}
      </button>
    )
  }

  return (
    <div className={`flex flex-col bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-sm font-semibold text-zinc-100">Activity Feed</h3>
          <span className="text-xs text-zinc-500">({filteredActivities.length})</span>
        </div>
        <div className="flex items-center gap-2">
          {compact && (
            <button
              onClick={() => setIsExpanded(false)}
              className="p-1 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
          {activities.length > 0 && (
            <button
              onClick={handleClearAll}
              className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="flex gap-1 p-2 border-b border-zinc-800 overflow-x-auto">
          {['all', 'query', 'decision', 'biomarker', 'treatment', 'alert', 'guideline'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-1 rounded text-xs whitespace-nowrap transition-colors ${
                filter === f
                  ? 'bg-violet-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
              }`}
            >
              {f === 'all' ? 'All' : f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      )}

      {/* Activity List */}
      <div className="flex-1 overflow-y-auto min-h-0" style={{ maxHeight: compact ? '300px' : '400px' }}>
        {filteredActivities.length > 0 ? (
          <div className="divide-y divide-zinc-800">
            {filteredActivities.map(activity => (
              <div
                key={activity.id}
                onClick={() => onActivityClick?.(activity)}
                className={`p-3 hover:bg-zinc-800/50 transition-colors ${
                  onActivityClick ? 'cursor-pointer' : ''
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg border ${getActivityColors(activity.type, activity.status)}`}>
                    {getActivityIcon(activity.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium text-zinc-100 truncate">
                        {activity.title}
                      </span>
                      <span className="text-xs text-zinc-500 whitespace-nowrap">
                        {formatRelativeTime(activity.timestamp)}
                      </span>
                    </div>
                    {activity.description && (
                      <p className="text-xs text-zinc-400 mt-1 line-clamp-2">
                        {activity.description}
                      </p>
                    )}
                    {activity.relatedPatientId && (
                      <span className="inline-block mt-1 px-1.5 py-0.5 bg-zinc-700 text-zinc-400 text-xs rounded">
                        Patient: {activity.relatedPatientId}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center text-zinc-500">
            <svg className="w-8 h-8 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-sm">No activity yet</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ActivityFeed
