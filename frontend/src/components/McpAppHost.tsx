'use client'

import { useRef, useEffect, useState, useCallback } from 'react'
import { X, Maximize2, Minimize2, RefreshCw, ExternalLink } from 'lucide-react'

interface McpAppHostProps {
  /** URI of the MCP app resource (e.g., /mcp-apps/treatment-compare.html) */
  resourceUri: string
  /** Tool input data to send to the app */
  toolInput: Record<string, unknown>
  /** Tool result data (structured content for the app) */
  toolResult?: {
    content?: Array<{ type: string; text: string }>
    structuredContent?: Record<string, unknown>
  }
  /** Callback when app calls a tool */
  onToolCall?: (toolName: string, args: Record<string, unknown>) => Promise<unknown>
  /** Callback when app sends a message to chat */
  onMessage?: (message: string) => void
  /** Callback when app updates model context */
  onContextUpdate?: (context: Record<string, unknown>) => void
  /** Display mode */
  displayMode?: 'inline' | 'fullscreen' | 'pip'
  /** Height for inline mode */
  height?: string
  /** Title shown in header */
  title?: string
  /** Allow user to toggle fullscreen */
  allowFullscreen?: boolean
  /** Show refresh button */
  allowRefresh?: boolean
  /** Custom class name */
  className?: string
}

interface McpMessage {
  jsonrpc: '2.0'
  id?: string | number
  method?: string
  params?: Record<string, unknown>
  result?: unknown
  error?: { code: number; message: string }
}

export function McpAppHost({
  resourceUri,
  toolInput,
  toolResult,
  onToolCall,
  onMessage,
  onContextUpdate,
  displayMode: initialDisplayMode = 'inline',
  height = '400px',
  title,
  allowFullscreen = true,
  allowRefresh = true,
  className = '',
}: McpAppHostProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null)
  const [displayMode, setDisplayMode] = useState(initialDisplayMode)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isInitialized, setIsInitialized] = useState(false)
  const pendingRequests = useRef<Map<string | number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>>(new Map())
  const messageIdCounter = useRef(0)

  // Generate unique message ID
  const generateId = useCallback(() => {
    return `mcp-${++messageIdCounter.current}-${Date.now()}`
  }, [])

  // Send message to iframe
  const sendToIframe = useCallback((message: McpMessage) => {
    const iframe = iframeRef.current
    if (!iframe?.contentWindow) {
      console.warn('[McpAppHost] Cannot send message - iframe not ready')
      return
    }
    iframe.contentWindow.postMessage(message, '*')
  }, [])

  // Handle messages from iframe
  useEffect(() => {
    const iframe = iframeRef.current

    const handleMessage = async (event: MessageEvent) => {
      // Only accept messages from our iframe
      if (event.source !== iframe?.contentWindow) return

      const message = event.data as McpMessage
      if (!message || typeof message !== 'object') return

      console.log('[McpAppHost] Received message:', message.method || 'response', message)

      // Handle responses to our requests
      if (message.id && pendingRequests.current.has(message.id)) {
        const { resolve, reject } = pendingRequests.current.get(message.id)!
        pendingRequests.current.delete(message.id)
        if (message.error) {
          reject(new Error(message.error.message))
        } else {
          resolve(message.result)
        }
        return
      }

      // Handle requests from the app
      if (message.method) {
        try {
          let result: unknown

          switch (message.method) {
            case 'ui/initialize':
              // App is ready, send host context
              setIsInitialized(true)
              setIsLoading(false)
              result = {
                hostContext: {
                  theme: 'dark',
                  locale: 'en-US',
                  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                  displayMode,
                  availableDisplayModes: ['inline', 'fullscreen'],
                  containerSize: {
                    width: iframe?.clientWidth || 600,
                    height: iframe?.clientHeight || 400,
                  },
                  platform: 'web',
                },
                capabilities: {
                  tools: true,
                  resources: false,
                  messages: true,
                  modelContext: true,
                },
              }
              break

            case 'tools/call':
              // Forward tool call to parent
              if (onToolCall && message.params) {
                const { name, arguments: args } = message.params as { name: string; arguments: Record<string, unknown> }
                result = await onToolCall(name, args || {})
              } else {
                throw new Error('Tool calls not supported')
              }
              break

            case 'ui/message':
              // Send message to chat
              if (onMessage && message.params) {
                const { content } = message.params as { content: string }
                onMessage(content)
                result = { success: true }
              }
              break

            case 'ui/update-model-context':
              // Update model context
              if (onContextUpdate && message.params) {
                onContextUpdate(message.params as Record<string, unknown>)
                result = { success: true }
              }
              break

            case 'ui/request-display-mode':
              // Handle display mode change request
              if (message.params) {
                const { mode } = message.params as { mode: 'inline' | 'fullscreen' | 'pip' }
                if (allowFullscreen || mode === 'inline') {
                  setDisplayMode(mode)
                  result = { mode }
                } else {
                  result = { mode: displayMode }
                }
              }
              break

            case 'ui/open-link':
              // Open external link
              if (message.params) {
                const { url } = message.params as { url: string }
                window.open(url, '_blank', 'noopener,noreferrer')
                result = { success: true }
              }
              break

            default:
              throw new Error(`Unknown method: ${message.method}`)
          }

          // Send response
          if (message.id) {
            sendToIframe({
              jsonrpc: '2.0',
              id: message.id,
              result,
            })
          }
        } catch (err) {
          console.error('[McpAppHost] Error handling message:', err)
          if (message.id) {
            sendToIframe({
              jsonrpc: '2.0',
              id: message.id,
              error: {
                code: -32603,
                message: err instanceof Error ? err.message : 'Internal error',
              },
            })
          }
        }
      }
    }

    window.addEventListener('message', handleMessage)
    return () => window.removeEventListener('message', handleMessage)
  }, [displayMode, allowFullscreen, onToolCall, onMessage, onContextUpdate, sendToIframe])

  // Send tool input when initialized
  useEffect(() => {
    if (isInitialized && toolInput) {
      sendToIframe({
        jsonrpc: '2.0',
        method: 'ui/notifications/tool-input',
        params: { arguments: toolInput },
      })
    }
  }, [isInitialized, toolInput, sendToIframe])

  // Send tool result when available
  useEffect(() => {
    if (isInitialized && toolResult) {
      sendToIframe({
        jsonrpc: '2.0',
        method: 'ui/notifications/tool-result',
        params: toolResult,
      })
    }
  }, [isInitialized, toolResult, sendToIframe])

  // Handle iframe load
  const handleIframeLoad = () => {
    setIsLoading(false)
    setError(null)
  }

  // Handle iframe error
  const handleIframeError = () => {
    setIsLoading(false)
    setError('Failed to load app')
  }

  // Refresh the app
  const handleRefresh = () => {
    setIsLoading(true)
    setIsInitialized(false)
    if (iframeRef.current) {
      iframeRef.current.src = resourceUri
    }
  }

  // Toggle fullscreen
  const toggleFullscreen = () => {
    setDisplayMode(displayMode === 'fullscreen' ? 'inline' : 'fullscreen')
  }

  const containerClasses = `
    mcp-app-host relative rounded-lg border overflow-hidden
    ${displayMode === 'fullscreen' ? 'fixed inset-4 z-50 bg-gray-900' : ''}
    ${displayMode === 'pip' ? 'fixed bottom-4 right-4 w-96 h-64 z-40 shadow-2xl' : ''}
    ${className}
  `

  return (
    <div
      className={containerClasses}
      style={{
        height: displayMode === 'inline' ? height : undefined,
        borderColor: 'var(--color-border-subtle, #374151)'
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-800/80 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-gray-200">
            {title || 'Interactive App'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          {allowRefresh && (
            <button
              onClick={handleRefresh}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Refresh"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          )}
          {allowFullscreen && (
            <button
              onClick={toggleFullscreen}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title={displayMode === 'fullscreen' ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {displayMode === 'fullscreen' ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </button>
          )}
          {displayMode === 'fullscreen' && (
            <button
              onClick={() => setDisplayMode('inline')}
              className="p-1.5 text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
              title="Close"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
          <div className="flex flex-col items-center gap-3">
            <RefreshCw className="w-8 h-8 text-blue-500 animate-spin" />
            <span className="text-sm text-gray-400">Loading app...</span>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
          <div className="flex flex-col items-center gap-3 text-center p-4">
            <div className="text-red-500 text-lg">Failed to load app</div>
            <p className="text-sm text-gray-400">{error}</p>
            <button
              onClick={handleRefresh}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Iframe */}
      <iframe
        ref={iframeRef}
        src={resourceUri}
        onLoad={handleIframeLoad}
        onError={handleIframeError}
        sandbox="allow-scripts allow-forms allow-same-origin"
        className="w-full h-full bg-gray-900"
        style={{
          height: displayMode === 'inline' ? `calc(${height} - 44px)` : 'calc(100% - 44px)',
          border: 'none'
        }}
      />

      {/* Fullscreen backdrop */}
      {displayMode === 'fullscreen' && (
        <div
          className="fixed inset-0 bg-black/60 -z-10"
          onClick={() => setDisplayMode('inline')}
        />
      )}
    </div>
  )
}

export default McpAppHost
