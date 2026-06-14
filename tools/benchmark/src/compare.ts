import { readFileSync } from 'fs'
import type { BenchmarkEvent, SuiteResult } from './types'

interface LatencyStats {
  n: number
  meanMs: number
  p50Ms: number
  p95Ms: number
  meanPhases: { signMs: number; sendMs: number; waitMs: number; verifyMs: number } | null
}

interface ThroughputRow {
  batchSize: number
  n: number
  meanBytesPerSec: number
  meanTokensPerSec: number
  meanWireBytes: number
}

interface HeatmapRow {
  nodeCount: number
  partitions: number
  n: number
  meanLatencyMs: number
}

interface SpecPipeRow {
  hops: number
  acceptanceRate: number
  pipelineDepth: 1 | 2
  msPerToken: number
  rollbacks: number
}

interface BaselineSummary {
  path: string
  startedAt: string
  finishedAt: string
  modelBlocks: number
  latencyMeanMs: number
  latencyJitterMs: number
  latency: LatencyStats
  throughput: ThroughputRow[]
  heatmap: HeatmapRow[]
  specPipe: SpecPipeRow[]
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))
  return sorted[idx]
}

function summarise(path: string): BaselineSummary {
  const raw = readFileSync(path, 'utf8')
  const result = JSON.parse(raw) as SuiteResult
  const events = result.events as BenchmarkEvent[]

  // Latency
  const latencyTotals: number[] = []
  const phaseSums = { signMs: 0, sendMs: 0, waitMs: 0, verifyMs: 0 }
  let phaseN = 0
  for (const e of events) {
    if (e.type !== 'latency:sample') continue
    latencyTotals.push(e.totalMs)
    if (!e.hops) continue
    for (const h of e.hops) {
      if (!h.phases) continue
      phaseSums.signMs += h.phases.signMs
      phaseSums.sendMs += h.phases.sendMs
      phaseSums.waitMs += h.phases.waitMs
      phaseSums.verifyMs += h.phases.verifyMs
      phaseN++
    }
  }
  latencyTotals.sort((a, b) => a - b)
  const latency: LatencyStats = {
    n: latencyTotals.length,
    meanMs: latencyTotals.length ? latencyTotals.reduce((a, b) => a + b, 0) / latencyTotals.length : 0,
    p50Ms: percentile(latencyTotals, 50),
    p95Ms: percentile(latencyTotals, 95),
    meanPhases: phaseN
      ? {
          signMs: phaseSums.signMs / phaseN,
          sendMs: phaseSums.sendMs / phaseN,
          waitMs: phaseSums.waitMs / phaseN,
          verifyMs: phaseSums.verifyMs / phaseN,
        }
      : null,
  }

  // Throughput
  const tpBuckets = new Map<number, { bytesPerSec: number[]; tokensPerSec: number[]; wireBytes: number[] }>()
  for (const e of events) {
    if (e.type !== 'throughput:sample') continue
    const b = tpBuckets.get(e.batchSize) ?? { bytesPerSec: [], tokensPerSec: [], wireBytes: [] }
    b.bytesPerSec.push(e.bytesPerSec)
    b.tokensPerSec.push(e.tokensPerSec)
    b.wireBytes.push(e.totalWireBytes ?? 0)
    tpBuckets.set(e.batchSize, b)
  }
  const throughput: ThroughputRow[] = [...tpBuckets.entries()]
    .sort(([a], [b]) => a - b)
    .map(([batchSize, b]) => ({
      batchSize,
      n: b.bytesPerSec.length,
      meanBytesPerSec: b.bytesPerSec.reduce((a, x) => a + x, 0) / b.bytesPerSec.length,
      meanTokensPerSec: b.tokensPerSec.reduce((a, x) => a + x, 0) / b.tokensPerSec.length,
      meanWireBytes: b.wireBytes.reduce((a, x) => a + x, 0) / b.wireBytes.length,
    }))

  // Heatmap
  const hmBuckets = new Map<string, { nodeCount: number; partitions: number; latencyMs: number[] }>()
  for (const e of events) {
    if (e.type !== 'heatmap:cell') continue
    const key = `${e.nodeCount}/${e.partitionBoundaries.length}`
    const b = hmBuckets.get(key) ?? { nodeCount: e.nodeCount, partitions: e.partitionBoundaries.length, latencyMs: [] }
    b.latencyMs.push(e.latencyMs)
    hmBuckets.set(key, b)
  }
  const heatmap: HeatmapRow[] = [...hmBuckets.values()]
    .sort((a, b) => a.nodeCount - b.nodeCount || a.partitions - b.partitions)
    .map(b => ({
      nodeCount: b.nodeCount,
      partitions: b.partitions,
      n: b.latencyMs.length,
      meanLatencyMs: b.latencyMs.reduce((a, x) => a + x, 0) / b.latencyMs.length,
    }))

  // SpecPipe
  const sp: SpecPipeRow[] = []
  for (const e of events) {
    if (e.type !== 'specpipe:sample') continue
    sp.push({
      hops: e.hops,
      acceptanceRate: e.acceptanceRate,
      pipelineDepth: e.pipelineDepth,
      msPerToken: e.msPerToken,
      rollbacks: e.rollbacks,
    })
  }

  return {
    path,
    startedAt: result.startedAt,
    finishedAt: result.finishedAt,
    modelBlocks: result.config.modelBlocks,
    latencyMeanMs: result.config.latencyMeanMs,
    latencyJitterMs: result.config.latencyJitterMs,
    latency,
    throughput,
    heatmap,
    specPipe: sp,
  }
}

function pct(a: number, b: number): string {
  if (a === 0) return '   —  '
  const d = ((b - a) / a) * 100
  const sign = d >= 0 ? '+' : ''
  return `${sign}${d.toFixed(1)}%`
}

function fmt(n: number, digits = 2): string {
  if (!Number.isFinite(n)) return '—'
  return n.toFixed(digits)
}

function printCompare(A: BaselineSummary, B: BaselineSummary): void {
  console.log('═══ Baseline comparison ═══')
  console.log(`A: ${A.path}`)
  console.log(`B: ${B.path}`)
  console.log(
    `Config: modelBlocks=${A.modelBlocks}/${B.modelBlocks}  latency mean=${A.latencyMeanMs}/${B.latencyMeanMs}ms  jitter=${A.latencyJitterMs}/${B.latencyJitterMs}ms`,
  )
  console.log()

  console.log('--- Latency (full chain) ---')
  console.log(`samples       A=${A.latency.n}    B=${B.latency.n}`)
  console.log(`mean   ms     ${fmt(A.latency.meanMs)}   →  ${fmt(B.latency.meanMs)}   (${pct(A.latency.meanMs, B.latency.meanMs)})`)
  console.log(`p50    ms     ${fmt(A.latency.p50Ms)}   →  ${fmt(B.latency.p50Ms)}   (${pct(A.latency.p50Ms, B.latency.p50Ms)})`)
  console.log(`p95    ms     ${fmt(A.latency.p95Ms)}   →  ${fmt(B.latency.p95Ms)}   (${pct(A.latency.p95Ms, B.latency.p95Ms)})`)
  if (A.latency.meanPhases && B.latency.meanPhases) {
    console.log('Phase means (per hop):')
    for (const k of ['signMs', 'sendMs', 'waitMs', 'verifyMs'] as const) {
      const a = A.latency.meanPhases[k]
      const b = B.latency.meanPhases[k]
      console.log(`  ${k.padEnd(10)}  ${fmt(a)}   →  ${fmt(b)}   (${pct(a, b)})`)
    }
  }
  console.log()

  console.log('--- Throughput by batch size ---')
  const sizes = [...new Set([...A.throughput.map(r => r.batchSize), ...B.throughput.map(r => r.batchSize)])].sort((x, y) => x - y)
  console.log('batch    wireBytes (B/req)             tokens/sec                       MB/sec')
  for (const bs of sizes) {
    const a = A.throughput.find(r => r.batchSize === bs)
    const b = B.throughput.find(r => r.batchSize === bs)
    const wA = a ? fmt(a.meanWireBytes, 0) : '—'
    const wB = b ? fmt(b.meanWireBytes, 0) : '—'
    const wPct = a && b ? pct(a.meanWireBytes, b.meanWireBytes) : '   —  '
    const tA = a ? fmt(a.meanTokensPerSec, 0) : '—'
    const tB = b ? fmt(b.meanTokensPerSec, 0) : '—'
    const tPct = a && b ? pct(a.meanTokensPerSec, b.meanTokensPerSec) : '   —  '
    const mA = a ? fmt(a.meanBytesPerSec / 1e6) : '—'
    const mB = b ? fmt(b.meanBytesPerSec / 1e6) : '—'
    const mPct = a && b ? pct(a.meanBytesPerSec, b.meanBytesPerSec) : '   —  '
    console.log(
      `${String(bs).padStart(3)}    ${wA.padStart(8)} → ${wB.padStart(8)} (${wPct.padStart(7)})    ${tA.padStart(7)} → ${tB.padStart(7)} (${tPct.padStart(7)})   ${mA.padStart(6)} → ${mB.padStart(6)} (${mPct.padStart(7)})`,
    )
  }
  console.log()

  console.log('--- Heatmap: mean latency by node count × partitions ---')
  const keys = [...new Set([
    ...A.heatmap.map(h => `${h.nodeCount}/${h.partitions}`),
    ...B.heatmap.map(h => `${h.nodeCount}/${h.partitions}`),
  ])].sort()
  console.log('nodes/parts    A meanMs    B meanMs    Δ')
  for (const key of keys) {
    const [nc, p] = key.split('/')
    const a = A.heatmap.find(h => `${h.nodeCount}/${h.partitions}` === key)
    const b = B.heatmap.find(h => `${h.nodeCount}/${h.partitions}` === key)
    const aMs = a ? fmt(a.meanLatencyMs) : '—'
    const bMs = b ? fmt(b.meanLatencyMs) : '—'
    const dPct = a && b ? pct(a.meanLatencyMs, b.meanLatencyMs) : '   —  '
    console.log(`  ${nc.padStart(2)} / ${p.padStart(2)}        ${aMs.padStart(7)}     ${bMs.padStart(7)}     ${dPct.padStart(7)}`)
  }

  if (A.specPipe.length > 0 || B.specPipe.length > 0) {
    console.log()
    console.log('--- SpecPipe: msPerToken by (hops, acceptance, depth) ---')
    const keys = [...new Set([
      ...A.specPipe.map(r => `${r.hops}/${r.acceptanceRate}/${r.pipelineDepth}`),
      ...B.specPipe.map(r => `${r.hops}/${r.acceptanceRate}/${r.pipelineDepth}`),
    ])].sort()
    console.log('hops/accept/depth     A ms/tok    B ms/tok    Δ')
    for (const key of keys) {
      const [h, ar, d] = key.split('/')
      const a = A.specPipe.find(r => `${r.hops}/${r.acceptanceRate}/${r.pipelineDepth}` === key)
      const b = B.specPipe.find(r => `${r.hops}/${r.acceptanceRate}/${r.pipelineDepth}` === key)
      const aMs = a ? fmt(a.msPerToken) : '—'
      const bMs = b ? fmt(b.msPerToken) : '—'
      const dPct = a && b ? pct(a.msPerToken, b.msPerToken) : '   —  '
      console.log(`  ${h.padStart(2)} / ${ar.padStart(4)} / ${d.padStart(1)}      ${aMs.padStart(7)}     ${bMs.padStart(7)}     ${dPct.padStart(7)}`)
    }
  }
}

function main(): void {
  const args = process.argv.slice(2)
  if (args.length !== 2) {
    console.error('usage: bun run tools/benchmark/src/compare.ts <baseline-A.json> <baseline-B.json>')
    process.exit(2)
  }
  printCompare(summarise(args[0]), summarise(args[1]))
}

main()
