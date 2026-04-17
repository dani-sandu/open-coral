import React, { createContext, useContext, useState, useCallback, useRef } from 'react'
import styles from './Toast.module.css'

type ToastType = 'error' | 'warning' | 'info'

interface Toast {
  id: number
  message: string
  type: ToastType
}

interface ToastContextValue {
  addToast: (message: string, type: ToastType) => void
}

const ToastContext = createContext<ToastContextValue>({ addToast: () => {} })

export function useToast(): ToastContextValue {
  return useContext(ToastContext)
}

export default function ToastProvider({ children }: { children: React.ReactNode }): React.JSX.Element {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(0)

  const removeToast = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const addToast = useCallback((message: string, type: ToastType) => {
    const id = nextId.current++
    setToasts(prev => [...prev, { id, message, type }])

    // Auto-dismiss non-errors after 8 seconds
    if (type !== 'error') {
      setTimeout(() => removeToast(id), 8000)
    }
  }, [removeToast])

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div className={styles.container}>
        {toasts.map(toast => (
          <div key={toast.id} className={`${styles.toast} ${styles[toast.type]}`}>
            <span className={styles.message}>{toast.message}</span>
            <button
              className={styles.close}
              onClick={() => removeToast(toast.id)}
            >
              x
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
