import React from 'react'
import type { CoverageReport } from '../../types'
import BlockBar from './BlockBar'
import StatusDot from '../shared/StatusDot'
import cmp from '../shared/components.module.css'
import { formatMissingRanges } from '../../utils/format'

interface Props {
  report: CoverageReport | null
  loading?: boolean
  onRefresh?: () => void
}

export default function CoverageStatus({ report, loading, onRefresh }: Props): React.JSX.Element {
  if (loading) {
    return (
      <div style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)', fontFamily: 'var(--font-mono)' }}>
        Checking coverage…
      </div>
    )
  }

  if (!report || report.totalBlocks === 0) {
    return (
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 'var(--fs-sm)', color: 'var(--dim)' }}>Load a model to check coverage.</span>
        {onRefresh && (
          <button onClick={onRefresh} className={cmp.btnSecondary}>↻</button>
        )}
      </div>
    )
  }

  // Build ranges array for BlockBar: local = green, remote = blue, missing = default
  const ranges = report.covered.map(cov => ({
    start: cov.blockStart,
    end: cov.blockEnd,
    type: (cov.peerId === 'local' ? 'local' : 'remote') as 'local' | 'remote',
  }))

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {/* Status text + refresh button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 'var(--fs-sm)', fontFamily: 'var(--font-mono)' }}>
          {report.complete
            ? (
              <span style={{ color: 'var(--green)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                <StatusDot color="green" />
                All {report.totalBlocks} blocks covered
              </span>
            )
            : (
              <span style={{ color: 'var(--dim)' }}>
                {report.totalBlocks - report.missing.length}/{report.totalBlocks} blocks covered
                {' — '}missing: {formatMissingRanges(report.missing)}
              </span>
            )
          }
        </span>
        {onRefresh && (
          <button onClick={onRefresh} className={cmp.btnSecondary}>↻</button>
        )}
      </div>

      {/* Block bar */}
      <BlockBar totalBlocks={report.totalBlocks} ranges={ranges} height={8} />

      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, fontSize: 'var(--fs-xs)', color: 'var(--dim)' }}>
        <span><span style={{ color: 'var(--green)' }}>■</span> local</span>
        <span><span style={{ color: 'var(--blue)' }}>■</span> remote</span>
        <span><span style={{ color: 'var(--red)' }}>■</span> missing</span>
      </div>
    </div>
  )
}
