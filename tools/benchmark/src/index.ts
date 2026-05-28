import { mkdirSync, writeFileSync } from 'fs'
import { join } from 'path'
import { parseArgs, runSimBenchmark } from './sim-runner'
import type { BenchmarkEvent, SuiteResult } from './types'

async function main(): Promise<void> {
  const cfg = parseArgs(process.argv.slice(2))
  const events: BenchmarkEvent[] = []
  const startedAt = new Date().toISOString()

  await runSimBenchmark(cfg, e => {
    events.push(e)
    console.log(JSON.stringify(e)) // newline-delimited JSON event stream
  })

  const result: SuiteResult = {
    events,
    startedAt,
    finishedAt: new Date().toISOString(),
    config: { modelBlocks: cfg.modelBlocks, latencyMeanMs: cfg.latencyMeanMs, latencyJitterMs: cfg.latencyJitterMs },
  }
  const outDir = join(import.meta.dir, '..', 'baselines')
  mkdirSync(outDir, { recursive: true })
  const stamp = new Date().toISOString().slice(0, 10)
  const outPath = join(outDir, `${stamp}-phase2-baseline.json`)
  writeFileSync(outPath, JSON.stringify(result, null, 2))
  console.error(`[benchmark] baseline written to ${outPath}`)
}

main().catch(err => {
  console.error('[benchmark] failed:', err)
  process.exit(1)
})
