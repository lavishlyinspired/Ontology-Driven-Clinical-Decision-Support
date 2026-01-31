'use client'

import { useState } from 'react'
import { MessageSquare, ChevronLeft, ChevronRight, X } from 'lucide-react'
import ChatInterface from './ChatInterface'
import { useApp } from '@/contexts/AppContext'

export default function ChatPanel() {
  const { isChatPanelOpen, setIsChatPanelOpen, setGraphData } = useApp()
  const [isCollapsed, setIsCollapsed] = useState(false)

  const handleToggle = () => {
    const newCollapsed = !isCollapsed
    setIsCollapsed(newCollapsed)
    setIsChatPanelOpen(!newCollapsed)
  }

  const handleGraphDataChange = (data: any) => {
    setGraphData(data)
  }

  const handleDecisionNodesChange = (nodes: any[]) => {
    // Handle decision nodes change if needed
    console.log('Decision nodes changed:', nodes)
  }

  if (!isChatPanelOpen) return null

  return (
    <>
      <div className={`chat-panel ${isCollapsed ? 'chat-panel-collapsed' : ''}`}>
        {/* Panel Header */}
        <div className="panel-header">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-violet-400" />
            {!isCollapsed && <h2 className="panel-title">Chat Assistant</h2>}
          </div>
          <button
            onClick={handleToggle}
            className="panel-toggle-btn"
            title={isCollapsed ? "Expand chat panel" : "Collapse chat panel"}
          >
            {isCollapsed ? (
              <ChevronLeft className="w-4 h-4" />
            ) : (
              <ChevronRight className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Panel Content */}
        {!isCollapsed && (
          <div className="panel-content">
            <ChatInterface
              onGraphDataChange={handleGraphDataChange}
              onDecisionNodesChange={handleDecisionNodesChange}
            />
          </div>
        )}
      </div>

      {/* Overlay for mobile */}
      {!isCollapsed && (
        <div
          className="panel-overlay md:hidden"
          onClick={() => setIsCollapsed(true)}
        />
      )}
    </>
  )
}