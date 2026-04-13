import { describe, it, expect, afterEach, mock } from 'bun:test'
import { createOpenCoralNode, type OpenCoralNode } from '../../src/p2p/node'
import { BlockRegistry } from '../../src/p2p/block-registry'

describe('BlockRegistry', () => {
  const nodes: OpenCoralNode[] = []
  afterEach(async () => {
    await Promise.all(nodes.map(n => n.stop()))
    nodes.length = 0
  })

  it('announces on start()', async () => {
    const node = await createOpenCoralNode()
    nodes.push(node)
    let announced = false
    const spy = mock(async () => { announced = true })

    const registry = new BlockRegistry(node.libp2p, {
      blockStart: 0,
      blockEnd: 7,
      announceBlocks: spy,
    })
    await registry.start()
    expect(announced).toBe(true)

    registry.dispose()
  })

  it('dispose() cancels re-announcement timer', async () => {
    const node = await createOpenCoralNode()
    nodes.push(node)
    let callCount = 0
    const spy = mock(async () => { callCount++ })

    const registry = new BlockRegistry(node.libp2p, {
      blockStart: 0,
      blockEnd: 7,
      reannounceIntervalMs: 50,  // fast interval for testing
      announceBlocks: spy,
    })
    await registry.start()

    registry.dispose()
    const countAtDispose = callCount         // snapshot after dispose (not before)
    await new Promise(r => setTimeout(r, 150))  // wait 3× interval
    // No more calls after dispose
    expect(callCount).toBe(countAtDispose)
  })

  it('dispose() before start() does not throw', () => {
    const stub = {} as any  // only used if start() is called, which it won't be
    const registry = new BlockRegistry(stub, {
      blockStart: 0,
      blockEnd: 7,
      announceBlocks: async () => {},
    })
    expect(() => registry.dispose()).not.toThrow()
  })

  it('start() + stop() via node.stop() integration', async () => {
    const node = await createOpenCoralNode()
    nodes.push(node)

    const registry = new BlockRegistry(node.libp2p, {
      blockStart: 0,
      blockEnd: 7,
      announceBlocks: mock(async () => {}),
    })
    await registry.start()

    // Should not throw when node stops
    await expect(node.stop()).resolves.toBeUndefined()
    registry.dispose()
  })
})
