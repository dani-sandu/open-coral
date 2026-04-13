import React, { useEffect, useState, useCallback } from 'react'

interface BlockRange {
  start: number
  end: number
}

interface NetworkPeer {
  peerId: string
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

const COLORS = {
  bg: '#1e1e2e',
  surface: '#181825',
  border: '#313244',
  text: '#cdd6f4',
  textDim: '#6c7086',
  accent: '#7c6af7',
  accentDim: '#45396e',
  green: '#a6e3a1',
  yellow: '#f9e2af',
  red: '#f38ba8',
  blue: '#89b4fa',
  teal: '#94e2d5',
  peach: '#fab387',
  connection: '#585b70',
}

const SVG_WIDTH = 800
const SVG_HEIGHT = 500

function shortPeerId(id: string): string {
  if (id.length <= 12) return id
  return id.slice(0, 6) + '…' + id.slice(-4)
}

function peerColor(index: number, isLocal: boolean): string {
  if (isLocal) return COLORS.accent
  const palette = [COLORS.blue, COLORS.teal, COLORS.peach, COLORS.yellow, COLORS.green, COLORS.red]
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
    laid.push({ peer: localPeer, x: cx, y: cy, color: COLORS.accent })
  }

  remotePeers.forEach((peer, i) => {
    const angle = (2 * Math.PI * i) / Math.max(remotePeers.length, 1) - Math.PI / 2
    laid.push({
      peer,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      color: peerColor(i + 1, false),
    })
  })

  return laid
}

function BlockRangeTag({ range, color }: { range: BlockRange; color: string }): React.JSX.Element {
  return (
    <span style={{
      display: 'inline-block',
      padding: '1px 6px',
      borderRadius: 4,
      fontSize: 10,
      fontFamily: 'monospace',
      background: color + '22',
      color: color,
      border: `1px solid ${color}44`,
      marginRight: 3,
    }}>
      blk {range.start}–{range.end}
    </span>
  )
}

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
      <circle cx={x} cy={y} r={r} fill={COLORS.surface} stroke={color} strokeWidth={peer.isLocal ? 3 : 2} />
      {/* inner dot */}
      <circle cx={x} cy={y} r={5} fill={color} opacity={0.8} />
      {/* label */}
      <text
        x={x} y={y + r + 16}
        textAnchor="middle"
        fontSize={11}
        fontFamily="monospace"
        fill={isHovered ? COLORS.text : COLORS.textDim}
      >
        {peer.isLocal ? 'LOCAL' : shortPeerId(peer.peerId)}
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
          {peer.blockRanges.map(r => `[${r.start}–${r.end}]`).join(' ')}
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
      stroke={highlighted ? COLORS.accent : COLORS.connection}
      strokeWidth={highlighted ? 2 : 1}
      strokeDasharray={highlighted ? 'none' : '4 4'}
      opacity={highlighted ? 0.8 : 0.4}
    />
  )
}

function StatusBar({ state }: { state: NetworkState | null }): React.JSX.Element {
  if (!state) {
    return (
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', padding: '8px 0' }}>
        <span style={{ color: COLORS.red, fontSize: 12 }}>● Node offline</span>
      </div>
    )
  }

  const remotePeers = state.peers.filter(p => !p.isLocal)
  return (
    <div style={{
      display: 'flex', gap: 20, alignItems: 'center', padding: '8px 0',
      fontSize: 12, color: COLORS.textDim, fontFamily: 'monospace',
    }}>
      <span style={{ color: COLORS.green }}>● Online</span>
      <span>Peer ID: <span style={{ color: COLORS.text }}>{shortPeerId(state.localPeerId)}</span></span>
      <span>Peers: <span style={{ color: COLORS.text }}>{remotePeers.length}</span></span>
      <span>Connections: <span style={{ color: COLORS.text }}>{state.connections.length}</span></span>
      {state.localBlocks.length > 0 && (
        <span>
          Blocks:{' '}
          {state.localBlocks.map((b, i) => (
            <BlockRangeTag key={i} range={b} color={COLORS.accent} />
          ))}
        </span>
      )}
    </div>
  )
}

function PeerDetailPanel({ peer, color }: { peer: NetworkPeer; color: string }): React.JSX.Element {
  return (
    <div style={{
      background: COLORS.surface,
      border: `1px solid ${COLORS.border}`,
      borderRadius: 8,
      padding: 14,
      fontSize: 12,
      fontFamily: 'monospace',
      color: COLORS.text,
      minWidth: 220,
    }}>
      <div style={{ marginBottom: 8, fontWeight: 600, color }}>
        {peer.isLocal ? '⬢ Local Node' : '⬡ Remote Peer'}
      </div>
      <div style={{ marginBottom: 4 }}>
        <span style={{ color: COLORS.textDim }}>Peer ID: </span>
        <span style={{ fontSize: 10, wordBreak: 'break-all' }}>{peer.peerId}</span>
      </div>
      <div style={{ marginBottom: 4 }}>
        <span style={{ color: COLORS.textDim }}>Addrs: </span>
        {peer.multiaddrs.length === 0
          ? <span style={{ color: COLORS.textDim }}>none</span>
          : peer.multiaddrs.map((a, i) => (
              <div key={i} style={{ fontSize: 10, paddingLeft: 8, color: COLORS.textDim }}>{a}</div>
            ))
        }
      </div>
      {peer.blockRanges.length > 0 && (
        <div style={{ marginTop: 6 }}>
          <span style={{ color: COLORS.textDim }}>Blocks: </span>
          {peer.blockRanges.map((r, i) => (
            <BlockRangeTag key={i} range={r} color={color} />
          ))}
        </div>
      )}
      {peer.modelInfo && (
        <div style={{ marginTop: 6, paddingTop: 6, borderTop: `1px solid ${COLORS.border}` }}>
          <div style={{ marginBottom: 3 }}>
            <span style={{ color: COLORS.textDim }}>Model: </span>
            <span style={{ color: COLORS.text, fontSize: 11 }}>
              {peer.modelInfo.hfFilename.endsWith('.gguf')
                ? peer.modelInfo.hfFilename.slice(0, -5)
                : peer.modelInfo.hfFilename}
            </span>
          </div>
          <div style={{ marginBottom: 3 }}>
            <span style={{ color: COLORS.textDim }}>Repo: </span>
            <span style={{ color: COLORS.textDim, fontSize: 10 }}>{peer.modelInfo.repoId}</span>
          </div>
          <div style={{ marginBottom: 3 }}>
            <span style={{ color: COLORS.textDim }}>Arch: </span>
            <span style={{ color: COLORS.text, fontSize: 11 }}>{peer.modelInfo.architecture}</span>
          </div>
          <div>
            <span style={{ color: COLORS.textDim }}>Coverage: </span>
            <span style={{ color: color, fontSize: 11 }}>
              {peer.modelInfo.blockEnd - peer.modelInfo.blockStart + 1}/{peer.modelInfo.totalBlocks} blocks
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

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

  return (
    <div style={{
      background: COLORS.bg,
      borderRadius: 12,
      border: `1px solid ${COLORS.border}`,
      padding: 20,
      fontFamily: 'system-ui',
      display: 'flex',
      flexDirection: 'column',
      flex: 1,
      minHeight: 0,
      overflow: 'hidden',
    }}>
      {/* Title bar */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 12,
      }}>
        <h2 style={{ margin: 0, fontSize: 16, color: COLORS.text, fontWeight: 600 }}>
          <span style={{ color: COLORS.accent }}>⬡</span> Network View
        </h2>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={refresh}
            style={{
              background: COLORS.surface, color: COLORS.textDim, border: `1px solid ${COLORS.border}`,
              borderRadius: 6, padding: '4px 10px', fontSize: 11, cursor: 'pointer',
            }}
          >
            ↻ Refresh
          </button>
          <button
            onClick={() => setAutoRefresh(v => !v)}
            style={{
              background: autoRefresh ? COLORS.accentDim : COLORS.surface,
              color: autoRefresh ? COLORS.accent : COLORS.textDim,
              border: `1px solid ${autoRefresh ? COLORS.accent + '44' : COLORS.border}`,
              borderRadius: 6, padding: '4px 10px', fontSize: 11, cursor: 'pointer',
            }}
          >
            {autoRefresh ? '● Live' : '○ Paused'}
          </button>
        </div>
      </div>

      <StatusBar state={state} />

      {error && (
        <div style={{ color: COLORS.red, fontSize: 12, marginBottom: 8 }}>{error}</div>
      )}

      <div style={{ position: 'relative', flex: 1, minHeight: 0, overflowY: 'auto' }}>
        {/* SVG graph */}
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          style={{
            width: '100%',
            background: COLORS.surface,
            borderRadius: 8,
            border: `1px solid ${COLORS.border}`,
          }}
        >
          {/* Grid dots */}
          <defs>
            <pattern id="grid-dots" width="30" height="30" patternUnits="userSpaceOnUse">
              <circle cx="15" cy="15" r="0.8" fill={COLORS.border} />
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
              fill={COLORS.textDim}
              fontFamily="system-ui"
            >
              {state ? 'No remote peers connected — connect a peer to see the network' : 'Starting P2P node…'}
            </text>
          )}
        </svg>

        {/* Detail panel — absolutely positioned so SVG doesn't resize */}
        {hoveredPeerData && (
          <div style={{ position: 'absolute', top: 12, right: 12, zIndex: 10 }}>
            <PeerDetailPanel peer={hoveredPeerData.peer} color={hoveredPeerData.color} />
          </div>
        )}
      </div>

      {/* Peer table */}
      {state && state.peers.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ fontSize: 13, color: COLORS.textDim, margin: '0 0 8px', fontWeight: 500 }}>
            Connected Peers ({state.peers.length})
          </h3>
          <table style={{
            width: '100%', borderCollapse: 'collapse', fontSize: 11,
            fontFamily: 'monospace', color: COLORS.text,
          }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${COLORS.border}`, color: COLORS.textDim }}>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Peer ID</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Type</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Model</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Blocks</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 500 }}>Address</th>
              </tr>
            </thead>
            <tbody>
              {state.peers.map((peer, i) => {
                const color = peer.isLocal ? COLORS.accent : peerColor(i, false)
                return (
                  <tr
                    key={peer.peerId}
                    style={{
                      borderBottom: `1px solid ${COLORS.border}`,
                      background: hoveredPeer === peer.peerId ? COLORS.surface : 'transparent',
                    }}
                    onMouseEnter={() => setHoveredPeer(peer.peerId)}
                    onMouseLeave={() => setHoveredPeer(null)}
                  >
                    <td style={{ padding: '6px 8px', color }}>{shortPeerId(peer.peerId)}</td>
                    <td style={{ padding: '6px 8px' }}>{peer.isLocal ? 'local' : 'remote'}</td>
                    <td style={{ padding: '6px 8px' }}>
                      {peer.modelInfo
                        ? <span style={{ color: COLORS.text }}>
                            {peer.modelInfo.hfFilename.endsWith('.gguf')
                              ? peer.modelInfo.hfFilename.slice(0, -5)
                              : peer.modelInfo.hfFilename}
                          </span>
                        : <span style={{ color: COLORS.textDim }}>—</span>
                      }
                    </td>
                    <td style={{ padding: '6px 8px' }}>
                      {peer.blockRanges.length > 0
                        ? peer.blockRanges.map((r, j) => (
                            <BlockRangeTag key={j} range={r} color={color} />
                          ))
                        : <span style={{ color: COLORS.textDim }}>—</span>
                      }
                    </td>
                    <td style={{ padding: '6px 8px', color: COLORS.textDim }}>
                      {peer.multiaddrs[0] || '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
