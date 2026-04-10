import os from 'os'
import { execSync } from 'child_process'

export interface HardwareInfo {
  totalRamBytes: bigint
  freeRamBytes: bigint
  gpu: GPUInfo | null
}

export interface GPUInfo {
  name: string
  backend: 'cuda' | 'rocm' | 'metal' | 'unknown'
  totalVramBytes: bigint
  freeVramBytes: bigint
}

function detectNvidiaGPU(): GPUInfo | null {
  try {
    const out = execSync(
      'nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits',
      { encoding: 'utf8', timeout: 5000 }
    ).trim()
    const [name, totalMiB, freeMiB] = out.split(',').map(s => s.trim())
    return {
      name,
      backend: 'cuda',
      totalVramBytes: BigInt(totalMiB) * 1_048_576n,
      freeVramBytes: BigInt(freeMiB) * 1_048_576n
    }
  } catch {
    return null
  }
}

function detectAMDGPU(): GPUInfo | null {
  try {
    const out = execSync('rocm-smi --showmeminfo vram --json', { encoding: 'utf8', timeout: 5000 })
    const data = JSON.parse(out)
    const card = Object.values(data)[0] as Record<string, string>
    return {
      name: card['Card Series'] ?? 'AMD GPU',
      backend: 'rocm',
      totalVramBytes: BigInt(card['VRAM Total Memory (B)'] ?? 0),
      freeVramBytes: BigInt(card['VRAM Total Used Memory (B)'] ?? 0)
    }
  } catch {
    return null
  }
}

function detectAppleSiliconGPU(): GPUInfo | null {
  if (process.platform !== 'darwin') return null
  try {
    const out = execSync('system_profiler SPDisplaysDataType -json', { encoding: 'utf8', timeout: 5000 })
    const data = JSON.parse(out)
    const displays = data?.SPDisplaysDataType ?? []
    const gpu = displays.find((d: Record<string, string>) => d['spdisplays_vendor'] === 'Apple')
    if (!gpu) return null
    // Apple Silicon shares memory — report half of total RAM as "VRAM"
    const halfRam = BigInt(os.totalmem()) / 2n
    return {
      name: gpu['spdisplays_device-id'] ?? 'Apple Silicon GPU',
      backend: 'metal',
      totalVramBytes: halfRam,
      freeVramBytes: halfRam
    }
  } catch {
    return null
  }
}

/**
 * Detect available hardware for block hosting.
 * Tries NVIDIA first, then AMD, then Apple Silicon, then returns null GPU.
 */
export async function detectHardware(): Promise<HardwareInfo> {
  const totalRamBytes = BigInt(os.totalmem())
  const freeRamBytes = BigInt(os.freemem())
  const gpu = detectNvidiaGPU() ?? detectAMDGPU() ?? detectAppleSiliconGPU()
  return { totalRamBytes, freeRamBytes, gpu }
}

/**
 * Estimate how many Q4_K_M blocks of a given hidden size fit in available memory.
 * Rough heuristic: each block ≈ 7 × hiddenSize² × 2 bytes (Q4 ≈ 2 bytes/param).
 */
export function estimateBlockCapacity(hw: HardwareInfo, hiddenSize: number): number {
  const bytesPerBlock = BigInt(Math.round(7 * hiddenSize * hiddenSize * 2))
  const availableBytes = hw.gpu ? hw.gpu.freeVramBytes : hw.freeRamBytes / 2n
  const capacity = Number(availableBytes / bytesPerBlock)
  return Math.max(1, Math.floor(capacity))
}
