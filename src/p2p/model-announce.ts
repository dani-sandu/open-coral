import type { Libp2p } from 'libp2p'
import type { PeerId, Stream } from '@libp2p/interface'

export const MODEL_INFO_PROTOCOL = '/coral/modelinfo/1.0.0'

export interface PeerModelInfoPayload {
  repoId: string
  hfFilename: string
  blockStart: number
  blockEnd: number
  totalBlocks: number
  hiddenSize: number
  architecture: string
}

async function collectStream(stream: AsyncIterable<Uint8Array | { subarray(): Uint8Array }>): Promise<Buffer> {
  const chunks: Buffer[] = []
  for await (const chunk of stream) {
    // libp2p streams yield Uint8ArrayList (has .subarray()) or plain Uint8Array
    const bytes: Uint8Array = typeof (chunk as any).subarray === 'function'
      ? (chunk as any).subarray()
      : (chunk as Uint8Array)
    chunks.push(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength))
  }
  return Buffer.concat(chunks)
}

export async function registerModelInfoHandler(
  libp2p: Libp2p,
  getLocalInfo: () => PeerModelInfoPayload | null,
): Promise<void> {
  await libp2p.handle(MODEL_INFO_PROTOCOL, async (stream: Stream) => {
    try {
      const info = getLocalInfo()
      if (info === null) {
        await stream.close()
        return
      }
      const encoded = new TextEncoder().encode(JSON.stringify(info) + '\n')
      const ok = stream.send(encoded)
      if (!ok) {
        await new Promise<void>(resolve => {
          stream.addEventListener('drain', () => resolve(), { once: true })
        })
      }
      await stream.close()
    } catch (err) {
      stream.abort(err instanceof Error ? err : new Error(String(err)))
    }
  }, { force: true })
}

export async function queryPeerModelInfo(
  libp2p: Libp2p,
  peerId: PeerId,
): Promise<PeerModelInfoPayload | null> {
  try {
    const stream = await libp2p.dialProtocol(peerId, MODEL_INFO_PROTOCOL, {
      signal: AbortSignal.timeout(3000),
    })
    const [raw] = await Promise.all([collectStream(stream), stream.close().catch(() => {})])
    if (raw.length === 0) {
      return null
    }
    let parsed: any
    try {
      parsed = JSON.parse(raw.toString('utf8').trim())
    } catch {
      return null
    }
    const { repoId, hfFilename, blockStart, blockEnd, totalBlocks, hiddenSize, architecture } = parsed
    if (
      typeof repoId !== 'string' || repoId.length === 0 ||
      typeof hfFilename !== 'string' || hfFilename.length === 0 ||
      typeof architecture !== 'string' || architecture.length === 0 ||
      !Number.isFinite(blockStart) || blockStart < 0 ||
      !Number.isFinite(blockEnd) || blockEnd < blockStart ||
      !Number.isFinite(totalBlocks) || totalBlocks <= 0 ||
      !Number.isFinite(hiddenSize) || hiddenSize <= 0
    ) {
      return null
    }
    return { repoId, hfFilename, blockStart, blockEnd, totalBlocks, hiddenSize, architecture }
  } catch {
    return null
  }
}
