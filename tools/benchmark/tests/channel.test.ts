import { describe, it, expect } from 'bun:test'
import { Channel } from '../src/sim/channel'

async function collect(ch: Channel): Promise<number[]> {
  const out: number[] = []
  for await (const chunk of ch) out.push(...chunk)
  return out
}

describe('Channel', () => {
  it('delivers values pushed before iteration starts', async () => {
    const ch = new Channel()
    ch.push(new Uint8Array([1, 2]))
    ch.push(new Uint8Array([3]))
    ch.close()
    expect(await collect(ch)).toEqual([1, 2, 3])
  })

  it('delivers values pushed after iteration starts (reader waits)', async () => {
    const ch = new Channel()
    const p = collect(ch)
    ch.push(new Uint8Array([7]))
    ch.push(new Uint8Array([8, 9]))
    ch.close()
    expect(await p).toEqual([7, 8, 9])
  })

  it('terminates iteration on close with no further values', async () => {
    const ch = new Channel()
    const p = collect(ch)
    ch.close()
    expect(await p).toEqual([])
  })

  it('ignores pushes after close', async () => {
    const ch = new Channel()
    ch.close()
    ch.push(new Uint8Array([1]))
    expect(await collect(ch)).toEqual([])
  })
})
