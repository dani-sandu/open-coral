type Resolver = (r: IteratorResult<Uint8Array>) => void

/**
 * One-directional in-process byte queue exposed as an async iterable.
 * A writer calls push()/close(); a single reader consumes via `for await`.
 */
export class Channel implements AsyncIterable<Uint8Array> {
  private readonly queue: Uint8Array[] = []
  private readonly waiters: Resolver[] = []
  private closed = false

  push(data: Uint8Array): void {
    if (this.closed) return
    const waiter = this.waiters.shift()
    if (waiter) {
      waiter({ value: data, done: false })
    } else {
      this.queue.push(data)
    }
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    let waiter: Resolver | undefined
    while ((waiter = this.waiters.shift())) {
      waiter({ value: undefined as unknown as Uint8Array, done: true })
    }
  }

  async *[Symbol.asyncIterator](): AsyncIterator<Uint8Array> {
    while (true) {
      const buffered = this.queue.shift()
      if (buffered !== undefined) {
        yield buffered
        continue
      }
      if (this.closed) return
      const result = await new Promise<IteratorResult<Uint8Array>>(resolve => {
        this.waiters.push(resolve)
      })
      if (result.done) return
      yield result.value
    }
  }
}
