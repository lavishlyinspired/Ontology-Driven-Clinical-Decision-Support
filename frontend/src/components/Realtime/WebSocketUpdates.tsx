"use client";

import { useState, useEffect, useRef } from "react";
import { Card, CardHeader, CardContent, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Bell, CheckCircle, AlertTriangle, Info, XCircle, Wifi, WifiOff } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";

interface WSMessage {
  id: string;
  channel: string;
  message: any;
  timestamp: string;
  priority: "low" | "medium" | "high";
  read: boolean;
}

const CHANNELS = [
  { name: "analysis_updates", label: "Analysis Updates", icon: Info },
  { name: "hitl_reviews", label: "HITL Reviews", icon: CheckCircle },
  { name: "batch_jobs", label: "Batch Jobs", icon: AlertTriangle },
  { name: "system_alerts", label: "System Alerts", icon: Bell },
];

export default function WebSocketUpdates() {
  const { user } = useAuth();
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState<WSMessage[]>([]);
  const [subscribedChannels, setSubscribedChannels] = useState<string[]>([
    "analysis_updates",
    "hitl_reviews",
  ]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!user) return;

    // Connect to WebSocket
    const wsUrl =
      process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/ws/connect";
    const ws = new WebSocket(`${wsUrl}?user_id=${user.id}`);

    ws.onopen = () => {
      console.log("WebSocket connected");
      setConnected(true);

      // Subscribe to channels
      subscribedChannels.forEach((channel) => {
        ws.send(
          JSON.stringify({
            type: "subscribe",
            channel,
          })
        );
      });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const newMessage: WSMessage = {
          id: `${Date.now()}-${Math.random()}`,
          channel: data.channel || "unknown",
          message: data.message,
          timestamp: new Date().toISOString(),
          priority: data.priority || "medium",
          read: false,
        };
        setMessages((prev) => [newMessage, ...prev].slice(0, 50)); // Keep last 50
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setConnected(false);
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [user, subscribedChannels]);

  const toggleChannel = (channel: string) => {
    if (subscribedChannels.includes(channel)) {
      setSubscribedChannels(subscribedChannels.filter((c) => c !== channel));
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "unsubscribe",
            channel,
          })
        );
      }
    } else {
      setSubscribedChannels([...subscribedChannels, channel]);
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({
            type: "subscribe",
            channel,
          })
        );
      }
    }
  };

  const markAsRead = (id: string) => {
    setMessages(
      messages.map((msg) => (msg.id === id ? { ...msg, read: true } : msg))
    );
  };

  const clearAll = () => {
    setMessages([]);
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 border-red-300 text-red-900";
      case "medium":
        return "bg-yellow-100 border-yellow-300 text-yellow-900";
      default:
        return "bg-blue-100 border-blue-300 text-blue-900";
    }
  };

  const unreadCount = messages.filter((m) => !m.read).length;

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="grid md:grid-cols-3 gap-6">
        {/* Channel Subscription Panel */}
        <Card className="md:col-span-1">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Channels</CardTitle>
              <div className="flex items-center gap-2">
                {connected ? (
                  <Wifi className="h-5 w-5 text-green-600" />
                ) : (
                  <WifiOff className="h-5 w-5 text-red-600" />
                )}
                <Badge variant={connected ? "default" : "destructive"}>
                  {connected ? "Connected" : "Disconnected"}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {CHANNELS.map((channel) => {
              const Icon = channel.icon;
              const isSubscribed = subscribedChannels.includes(channel.name);
              return (
                <div
                  key={channel.name}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <Icon className="h-5 w-5 text-gray-600" />
                    <span className="font-medium text-sm">{channel.label}</span>
                  </div>
                  <Button
                    size="sm"
                    variant={isSubscribed ? "default" : "outline"}
                    onClick={() => toggleChannel(channel.name)}
                  >
                    {isSubscribed ? "Subscribed" : "Subscribe"}
                  </Button>
                </div>
              );
            })}

            <div className="pt-4 border-t mt-4">
              <div className="text-sm text-gray-600 space-y-1">
                <p>
                  <strong>Active:</strong> {subscribedChannels.length} channels
                </p>
                <p>
                  <strong>Unread:</strong> {unreadCount} messages
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Messages Feed */}
        <Card className="md:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Bell className="h-5 w-5" />
                <CardTitle className="text-lg">Real-Time Updates</CardTitle>
                {unreadCount > 0 && (
                  <Badge variant="destructive">{unreadCount}</Badge>
                )}
              </div>
              <Button size="sm" variant="outline" onClick={clearAll}>
                Clear All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {messages.length === 0 ? (
              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription>
                  No messages yet. Subscribe to channels to receive real-time
                  updates.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`p-4 border rounded-lg ${
                      msg.read ? "bg-gray-50" : "bg-white"
                    } ${getPriorityColor(msg.priority)}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline">{msg.channel}</Badge>
                        <Badge
                          variant={
                            msg.priority === "high" ? "destructive" : "default"
                          }
                        >
                          {msg.priority}
                        </Badge>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-600">
                          {new Date(msg.timestamp).toLocaleTimeString()}
                        </span>
                        {!msg.read && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => markAsRead(msg.id)}
                          >
                            <CheckCircle className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>

                    <div className="text-sm">
                      {typeof msg.message === "string" ? (
                        <p>{msg.message}</p>
                      ) : (
                        <pre className="text-xs overflow-auto">
                          {JSON.stringify(msg.message, null, 2)}
                        </pre>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
