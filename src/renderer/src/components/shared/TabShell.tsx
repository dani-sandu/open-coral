import React from 'react'
import styles from './TabShell.module.css'

interface TabShellProps {
  title: string
  status?: React.ReactNode
  actions?: React.ReactNode
  children: React.ReactNode
}

export default function TabShell({ title, status, actions, children }: TabShellProps): React.JSX.Element {
  const hasStrip = status !== undefined || actions !== undefined

  return (
    <div className={styles.shell}>
      <h2 className={styles.title}>
        <span className={styles.titleIcon}>&#x2B21;</span>
        {title}
      </h2>
      {hasStrip && (
        <div className={styles.strip}>
          {status && <div className={styles.stripStatus}>{status}</div>}
          {actions && <div className={styles.stripActions}>{actions}</div>}
        </div>
      )}
      <div className={styles.content}>
        {children}
      </div>
    </div>
  )
}
