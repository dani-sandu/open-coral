import { describe, it, expect } from 'bun:test'
import { Channel } from '../src/sim/channel'
import { SimStream } from '../src/sim/sim-stream'
import { sendChunked, collectChunkedStream } from '../../../src/p2p/stream-utils'

describe('SimStream', () => {
  it('survives a sendChunked → collectChunkedStream round-trip', async () => {
    const a2b = new Channel()
    const b2a = new Channel()
    const initiator = new SimStream(a2b, b2a) // writes a2b, reads b2a
    const responder = new SimStream(b2a, a2b) // writes b2a, reads a2b

    const payload = new Uint8Array(200_000) // larger than one 64KB chunk
    for (let i = 0; i < payload.length; i++) payload[i] = i % 256

    // Responder echoes the request back, then closes its write side.
    const responderDone = (async () => {
      const received = await collectChunkedStream(responder)
      await sendChunked(responder, received)
      await responder.close()
    })()

    await sendChunked(initiator, payload)
    await initiator.close() // close write side so responder's read terminates
    const echoed = await collectChunkedStream(initiator)
    await responderDone

    expect(echoed.byteLength).toBe(payload.byteLength)
    expect(Array.from(echoed.slice(0, 5))).toEqual([0, 1, 2, 3, 4])
  })

  it('send() returns true (no backpressure in-process)', () => {
    const s = new SimStream(new Channel(), new Channel())
    expect(s.send(new Uint8Array([1]))).toBe(true)
  })
})
