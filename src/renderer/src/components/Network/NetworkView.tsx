import React, { useEffect, useState, useCallback } from 'react'
import TabShell from '../shared/TabShell'
import BlockRangeTag from '../Block/BlockRangeTag'
import StatusDot from '../shared/StatusDot'
import cmp from '../shared/components.module.css'
import styles from './NetworkView.module.css'

// ── Interfaces ─────────────────────────────────────────────────────────────────

interface BlockRange {
  start: number
  end: number
}

interface NetworkPeer {
  peerId: string
  displayName?: string
  peerType: 'local' | 'remote' | 'discovery'
  multiaddrs: string[]
  blockRanges: BlockRange[]
  isLocal: boolean
  connected: boolean
  modelInfo?: {
    repoId: string
    hfFilename: string
    blockStart: number
    blockEnd: number
    totalBlocks: number
    architecture: string
  }
}

interface NetworkConnection {
  from: string
  to: string
}

interface NetworkState {
  localPeerId: string
  localMultiaddrs: string[]
  localBlocks: BlockRange[]
  peers: NetworkPeer[]
  connections: NetworkConnection[]
  timestamp: number
}

// ── SVG colour palette ─────────────────────────────────────────────────────────
// SVG attributes (stroke/fill) do not support CSS variables, so a local map is
// used for the SVG-specific parts only. Everything else uses CSS variables.

const SVG_COLORS = {
  surface: '#181825',
  border: '#313244',
  accent: '#7c6af7',
  text: '#cdd6f4',
  dim: '#6c7086',
  connection: '#585b70',
  green: '#a6e3a1',
  yellow: '#f9e2af',
  red: '#f38ba8',
  blue: '#89b4fa',
  teal: '#94e2d5',
  peach: '#fab387',
}

const SVG_WIDTH = 800
const SVG_HEIGHT = 500

// ── Helpers ───────────────────────────────────────────────────────────────────

function shortPeerId(id: string): string {
  if (id.length <= 12) return id
  return id.slice(0, 6) + '…' + id.slice(-4)
}

function peerColor(index: number, peer: { isLocal: boolean; peerType?: string }): string {
  if (peer.isLocal) return SVG_COLORS.accent
  if (peer.peerType === 'discovery') return SVG_COLORS.yellow
  const palette = [SVG_COLORS.blue, SVG_COLORS.teal, SVG_COLORS.peach, SVG_COLORS.green, SVG_COLORS.red]
  return palette[index % palette.length]
}

function layoutPeers(peers: NetworkPeer[]): { peer: NetworkPeer; x: number; y: number; color: string }[] {
  const cx = SVG_WIDTH / 2
  const cy = SVG_HEIGHT / 2
  const radius = Math.min(SVG_WIDTH, SVG_HEIGHT) * 0.32

  const localPeer = peers.find(p => p.isLocal)
  const remotePeers = peers.filter(p => !p.isLocal)

  const laid: { peer: NetworkPeer; x: number; y: number; color: string }[] = []

  if (localPeer) {
    laid.push({ peer: localPeer, x: cx, y: cy, color: SVG_COLORS.accent })
  }

  remotePeers.forEach((peer, i) => {
    const angle = (2 * Math.PI * i) / Math.max(remotePeers.length, 1) - Math.PI / 2
    laid.push({
      peer,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      color: peerColor(i + 1, peer),
    })
  })

  return laid
}

// ── SVG sub-components ────────────────────────────────────────────────────────

function PeerNode({
  x, y, peer, color, isHovered, onHover,
}: {
  x: number; y: number; peer: NetworkPeer; color: string; isHovered: boolean; onHover: (id: string | null) => void
}): React.JSX.Element {
  const r = peer.isLocal ? 28 : 22
  return (
    <g
      onMouseEnter={() => onHover(peer.peerId)}
      onMouseLeave={() => onHover(null)}
      style={{ cursor: 'pointer' }}
    >
      {/* glow */}
      {isHovered && (
        <circle cx={x} cy={y} r={r + 8} fill={color} opacity={0.12} />
      )}
      {/* ring */}
      <circle cx={x} cy={y} r={r} fill={SVG_COLORS.surface} stroke={color} strokeWidth={peer.isLocal ? 3 : 2} />
      {/* inner dot */}
      <circle cx={x} cy={y} r={5} fill={color} opacity={0.8} />
      {/* label */}
      <text
        x={x} y={y + r + 16}
        textAnchor="middle"
        fontSize={11}
        fontFamily="monospace"
        fill={isHovered ? SVG_COLORS.text : SVG_COLORS.dim}
      >
        {peer.isLocal ? 'LOCAL' : peer.displayName ?? shortPeerId(peer.peerId)}
      </text>
      {/* block ranges */}
      {peer.blockRanges.length > 0 && (
        <text
          x={x} y={y + r + 30}
          textAnchor="middle"
          fontSize={9}
          fontFamily="monospace"
          fill={color}
          opacity={0.8}
        >
          {peer.blockRanges.map(br => `[${br.start}–${br.end}]`).join(' ')}
        </text>
      )}
    </g>
  )
}

function ConnectionLine({
  x1, y1, x2, y2, highlighted,
}: {
  x1: number; y1: number; x2: number; y2: number; highlighted: boolean
}): React.JSX.Element {
  return (
    <line
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={highlighted ? SVG_COLORS.accent : SVG_COLORS.connection}
      strokeWidth={highlighted ? 2 : 1}
      strokeDasharray={highlighted ? 'none' : '4 4'}
      opacity={highlighted ? 0.8 : 0.4}
    />
  )
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function PeerDetailPanel({ peer, color }: { peer: NetworkPeer; color: string }): React.JSX.Element {
  return (
    <div className={styles.detailPanel}>
      <div className={styles.detailPanelTitle} style={{ color }}>
        {peer.peerType === 'local' ? '⬢ Local Node' : peer.peerType === 'discovery' ? 'Coral Network' : 'Remote Peer'}
      </div>
      <div className={styles.detailRow}>
        <span style={{ color: 'var(--dim)' }}>Peer ID: </span>
        <span style={{ fontSize: 'var(--fs-xs)', wordBreak: 'break-all' }}>{peer.peerId}</span>
      </div>
      <div className={styles.detailRow}>
        <span style={{ color: 'var(--dim)' }}>Addrs: </span>
        {peer.multiaddrs.length === 0
          ? <span style={{ color: 'var(--dim)' }}>none</span>
          : peer.multiaddrs.map((a, i) => (
              <div key={i} className={styles.detailRowIndent}>{a}</div>
            ))
        }
      </div>
      {peer.blockRanges.length > 0 && (
        <div className={styles.detailRow} style={{ marginTop: 6 }}>
          <span style={{ color: 'var(--dim)' }}>Blocks: </span>
          {peer.blockRanges.map((br, i) => (
            <BlockRangeTag key={i} start={br.start} end={br.end} color={color} />
          ))}
        </div>
      )}
      {peer.modelInfo && (
        <div className={styles.detailModelSection}>
          <div className={styles.detailModelRow}>
            <span style={{ color: 'var(--dim)' }}>Model: </span>
            <span style={{ fontSize: 'var(--fs-sm)' }}>
              {peer.modelInfo.hfFilename.endsWith('.gguf')
                ? peer.modelInfo.hfFilename.slice(0, -5)
                : peer.modelInfo.hfFilename}
            </span>
          </div>
          <div className={styles.detailModelRow}>
            <span style={{ color: 'var(--dim)' }}>Repo: </span>
            <span style={{ color: 'var(--dim)', fontSize: 'var(--fs-xs)' }}>{peer.modelInfo.repoId}</span>
          </div>
          <div className={styles.detailModelRow}>
            <span style={{ color: 'var(--dim)' }}>Arch: </span>
            <span style={{ fontSize: 'var(--fs-sm)' }}>{peer.modelInfo.architecture}</span>
          </div>
          <div>
            <span style={{ color: 'var(--dim)' }}>Coverage: </span>
            <span style={{ color, fontSize: 'var(--fs-sm)' }}>
              {peer.modelInfo.blockEnd - peer.modelInfo.blockStart + 1}/{peer.modelInfo.totalBlocks} blocks
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function NetworkView(): React.JSX.Element {
  const [state, setState] = useState<NetworkState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hoveredPeer, setHoveredPeer] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const refresh = useCallback(async () => {
    try {
      const s = await window.opencoral.getNetworkState()
      setState(s)
      setError(null)
    } catch (err) {
      setError(String(err))
    }
  }, [])

  useEffect(() => {
    refresh()
    if (!autoRefresh) return
    const id = setInterval(refresh, 2000)
    return () => clearInterval(id)
  }, [refresh, autoRefresh])

  const laidOut = state ? layoutPeers(state.peers) : []
  const peerMap = new Map(laidOut.map(l => [l.peer.peerId, l]))
  const hoveredPeerData = hoveredPeer ? laidOut.find(l => l.peer.peerId === hoveredPeer) : null

  // ── TabShell status strip ──────────────────────────────────────────────────

  const statusContent = !state
    ? <StatusDot color="red" label="Node offline" />
    : (
      <>
        <StatusDot color="green" label="Online" />
        <span>
          Peer ID: <span style={{ color: 'var(--text)' }}>{shortPeerId(state.localPeerId)}</span>
        </span>
        <span>
          Peers: <span style={{ color: 'var(--text)' }}>{state.peers.filter(p => !p.isLocal).length}</span>
        </span>
        <span>
          Connections: <span style={{ color: 'var(--text)' }}>{state.connections.length}</span>
        </span>
        {state.localBlocks.length > 0 && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            Blocks:{' '}
            {state.localBlocks.map((b, i) => (
              <BlockRangeTag key={i} start={b.start} end={b.end} color="var(--accent)" />
            ))}
          </span>
        )}
      </>
    )

  const actionsContent = (
    <>
      <button className={cmp.btnSecondary} onClick={refresh}>
        &#x21BB; Refresh
      </button>
      <button
        className={`${cmp.btnSecondary}${autoRefresh ? ` ${styles.btnLive}` : ''}`}
        onClick={() => setAutoRefresh(v => !v)}
      >
        {autoRefresh ? '● Live' : '○ Paused'}
      </button>
    </>
  )

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <TabShell title="Network View" status={statusContent} actions={actionsContent}>

      {error && <div className={styles.error}>{error}</div>}

      {/* SVG graph */}
      <div className={styles.graphWrapper}>
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className={styles.svgGraph}
        >
          {/* Grid dots */}
          <defs>
            <pattern id="grid-dots" width="30" height="30" patternUnits="userSpaceOnUse">
              <circle cx="15" cy="15" r="0.8" fill={SVG_COLORS.border} />
            </pattern>
          </defs>
          <rect width={SVG_WIDTH} height={SVG_HEIGHT} fill="url(#grid-dots)" rx="8" />

          {/* Connections */}
          {state?.connections.map((conn, i) => {
            const from = peerMap.get(conn.from)
            const to = peerMap.get(conn.to)
            if (!from || !to) return null
            const highlighted = hoveredPeer === conn.from || hoveredPeer === conn.to
            return (
              <ConnectionLine
                key={i}
                x1={from.x} y1={from.y}
                x2={to.x} y2={to.y}
                highlighted={highlighted}
              />
            )
          })}

          {/* Peer nodes */}
          {laidOut.map(({ peer, x, y, color }) => (
            <PeerNode
              key={peer.peerId}
              x={x} y={y}
              peer={peer}
              color={color}
              isHovered={hoveredPeer === peer.peerId}
              onHover={setHoveredPeer}
            />
          ))}

          {/* Empty state label */}
          {(!state || state.peers.length <= 1) && (
            <text
              x={SVG_WIDTH / 2} y={SVG_HEIGHT - 30}
              textAnchor="middle"
              fontSize={12}
              fill={SVG_COLORS.dim}
              fontFamily="system-ui"
            >
              {state ? 'No remote peers connected — connect a peer to see the network' : 'Starting P2P node…'}
            </text>
          )}
        </svg>

        {/* Detail panel — absolutely positioned so SVG doesn't resize */}
        {hoveredPeerData && (
          <PeerDetailPanel peer={hoveredPeerData.peer} color={hoveredPeerData.color} />
        )}
      </div>

      {/* Peer table */}
      {state && state.peers.length > 0 && (
        <div className={styles.tableSection}>
          <h3 className={styles.tableHeading}>
            Connected Peers ({state.peers.length})
          </h3>
          <table className={styles.peerTable}>
            <thead>
              <tr>
                <th>Peer ID</th>
                <th>Type</th>
                <th>Model</th>
                <th>Blocks</th>
                <th>Address</th>
              </tr>
            </thead>
            <tbody>
              {state.peers.map((peer, i) => {
                const color = peerColor(i, peer)
                return (
                  <tr
                    key={peer.peerId}
                    className={`${styles.peerRow}${hoveredPeer === peer.peerId ? ` ${styles.peerRowHovered}` : ''}`}
                    onMouseEnter={() => setHoveredPeer(peer.peerId)}
                    onMouseLeave={() => setHoveredPeer(null)}
                  >
                    <td style={{ color }}>{peer.displayName ?? shortPeerId(peer.peerId)}</td>
                    <td>{peer.peerType}</td>
                    <td>
                      {peer.modelInfo
                        ? <span>
                            {peer.modelInfo.hfFilename.endsWith('.gguf')
                              ? peer.modelInfo.hfFilename.slice(0, -5)
                              : peer.modelInfo.hfFilename}
                          </span>
                        : <span style={{ color: 'var(--dim)' }}>—</span>
                      }
                    </td>
                    <td>
                      {peer.blockRanges.length > 0
                        ? peer.blockRanges.map((br, j) => (
                            <BlockRangeTag key={j} start={br.start} end={br.end} color={color} />
                          ))
                        : <span style={{ color: 'var(--dim)' }}>—</span>
                      }
                    </td>
                    <td style={{ color: 'var(--dim)' }}>
                      {peer.multiaddrs[0] || '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </TabShell>
  )
}
