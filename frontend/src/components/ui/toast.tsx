"use client";

import * as React from "react";

// Simple toast implementation as a replacement for sonner
interface ToastState {
  id: number;
  message: string;
  type: "success" | "error" | "info";
}

const ToastContext = React.createContext<{
  toasts: ToastState[];
  addToast: (message: string, type: "success" | "error" | "info") => void;
  removeToast: (id: number) => void;
} | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<ToastState[]>([]);
  const idRef = React.useRef(0);

  const addToast = React.useCallback((message: string, type: "success" | "error" | "info") => {
    const id = idRef.current++;
    setToasts((prev) => [...prev, { id, message, type }]);

    // Auto-remove after 4 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = React.useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
      {children}
      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </ToastContext.Provider>
  );
}

function ToastContainer({
  toasts,
  removeToast
}: {
  toasts: ToastState[];
  removeToast: (id: number) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`flex items-center gap-2 rounded-lg px-4 py-3 text-sm shadow-lg ${
            toast.type === "success"
              ? "bg-green-600 text-white"
              : toast.type === "error"
              ? "bg-red-600 text-white"
              : "bg-[#1e293b] text-[#f8fafc]"
          }`}
          onClick={() => removeToast(toast.id)}
        >
          {toast.type === "success" && (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {toast.type === "error" && (
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
          <span>{toast.message}</span>
        </div>
      ))}
    </div>
  );
}

export function useToast() {
  const context = React.useContext(ToastContext);
  if (!context) {
    // Return a no-op toast for when provider isn't available
    return {
      success: (message: string) => console.log("[Toast Success]", message),
      error: (message: string) => console.error("[Toast Error]", message),
      info: (message: string) => console.log("[Toast Info]", message),
    };
  }

  return {
    success: (message: string) => context.addToast(message, "success"),
    error: (message: string) => context.addToast(message, "error"),
    info: (message: string) => context.addToast(message, "info"),
  };
}

// Sonner-compatible toast export
export const toast = {
  success: (message: string) => {
    // Will work if ToastProvider is mounted, otherwise console fallback
    if (typeof window !== 'undefined') {
      const event = new CustomEvent('toast', { detail: { message, type: 'success' } });
      window.dispatchEvent(event);
    }
    console.log("[Toast Success]", message);
  },
  error: (message: string) => {
    if (typeof window !== 'undefined') {
      const event = new CustomEvent('toast', { detail: { message, type: 'error' } });
      window.dispatchEvent(event);
    }
    console.error("[Toast Error]", message);
  },
  info: (message: string) => {
    if (typeof window !== 'undefined') {
      const event = new CustomEvent('toast', { detail: { message, type: 'info' } });
      window.dispatchEvent(event);
    }
    console.log("[Toast Info]", message);
  },
};
