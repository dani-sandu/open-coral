import { describe, it, expect, beforeAll, afterAll } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import {
  INFERENCE_PROTOCOL,
  registerInferenceHandler,
  sendInferenceRequest,
  type InferenceHandler,
} from '../../src/p2p/inference-protocol'

describe('inference protocol', () => {
  let nodeA: OpenCoralNode
  let nodeB: OpenCoralNode

  beforeAll(async () => {
    nodeA = await createOpenCoralNode()
    nodeB = await createOpenCoralNode()
    await nodeA.libp2p.dial(nodeB.libp2p.getMultiaddrs()[0])
  })

  afterAll(async () => {
    await nodeA.stop()
    await nodeB.stop()
  })

  it('protocol ID is correct', () => {
    expect(INFERENCE_PROTOCOL).toBe('/opencoral/inference/1.0.0')
  })

  it('handler receives correct tensor shape', async () => {
    let receivedTokens = 0
    let receivedEmbd = 0

    const handler: InferenceHandler = async (input, nTokens, nEmbd) => {
      receivedTokens = nTokens
      receivedEmbd = nEmbd
      return input  // echo
    }
    await registerInferenceHandler(nodeB.libp2p, handler)

    const input = new Float32Array(3 * 16).fill(0.5)
    await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, 3, 16)

    expect(receivedTokens).toBe(3)
    expect(receivedEmbd).toBe(16)
  })

  it('response is a Float32Array with the correct length', async () => {
    const handler: InferenceHandler = async (input) => input  // identity

    // Unregister any previous handler and re-register
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, handler)

    const nTokens = 2
    const nEmbd = 8
    const input = new Float32Array(nTokens * nEmbd).fill(0.1)
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, nTokens, nEmbd)

    expect(output).toBeInstanceOf(Float32Array)
    expect(output.length).toBe(nTokens * nEmbd)
  })

  it('response values match echo handler output', async () => {
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, async (input) => input)

    const input = new Float32Array([1.1, 2.2, 3.3, 4.4])
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, 1, 4)

    // Float32 round-trip tolerance (IEEE 754 single precision)
    for (let i = 0; i < input.length; i++) {
      expect(output[i]).toBeCloseTo(input[i], 5)
    }
  })

  it('roundtrip with multi-token batch', async () => {
    try { await nodeB.libp2p.unhandle(INFERENCE_PROTOCOL) } catch {}
    await registerInferenceHandler(nodeB.libp2p, async (input) => input)

    const nTokens = 4
    const nEmbd = 8
    const input = new Float32Array(nTokens * nEmbd).fill(0.2)
    const output = await sendInferenceRequest(nodeA.libp2p, nodeB.libp2p.peerId, input, nTokens, nEmbd)
    expect(output.length).toBe(nTokens * nEmbd)
  })
})
