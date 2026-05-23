/** Token IDs for a model's <think> / </think> markers (single-token only). */
export interface ThinkTokens {
  open: number
  close: number
}

/** Default thinking-phase token allowance. Tunable. */
export const THINKING_TOKEN_BUDGET = 2048

/** True if `tokens` contains an unclosed <think> (more opens than closes). */
export function hasOpenThinkBlock(tokens: number[], thinkTokens: ThinkTokens | undefined): boolean {
  if (thinkTokens === undefined) return false
  let open = 0
  let close = 0
  for (const t of tokens) {
    if (t === thinkTokens.open) open++
    else if (t === thinkTokens.close) close++
  }
  return open > close
}

/**
 * Tracks the two-phase token budget for reasoning models: a thinking phase
 * (capped by `thinkingBudget`) and an answer phase (capped by `answerBudget`).
 * Thinking-phase tokens never count against the answer cap.
 *
 * When `thinkTokens` is undefined the model is not thinking-capable: the tracker
 * stays in the answer phase for its whole life — a plain `answerBudget` cap.
 */
export class ThinkingBudget {
  private inThinking: boolean
  private thinkingUsed = 0
  private answerUsed = 0

  constructor(
    private readonly thinkTokens: ThinkTokens | undefined,
    private readonly thinkingBudget: number,
    private readonly answerBudget: number,
    startInThinking: boolean,
  ) {
    this.inThinking = thinkTokens !== undefined && startInThinking
  }

  /** Whether the tracker is currently in the thinking phase. */
  get thinking(): boolean {
    return this.inThinking
  }

  /** Tokens still permitted in the current phase. */
  remaining(): number {
    return this.inThinking
      ? this.thinkingBudget - this.thinkingUsed
      : this.answerBudget - this.answerUsed
  }

  /**
   * True while generation should continue: either the current phase has room,
   * or the thinking phase is spent and a force-close is still pending.
   */
  shouldContinue(): boolean {
    return this.remaining() > 0 || this.needsForceClose()
  }

  /**
   * True when the thinking budget is spent but `</think>` was never emitted —
   * the caller must inject `</think>` and call `forceClose()`.
   */
  needsForceClose(): boolean {
    return this.inThinking && this.thinkingUsed >= this.thinkingBudget
  }

  /**
   * Account for one emitted token. A generated `</think>` switches to the answer
   * phase; a generated `<think>` switches back to the thinking phase.
   */
  count(tokenId: number): void {
    if (this.inThinking) this.thinkingUsed++
    else this.answerUsed++
    if (this.thinkTokens === undefined) return
    if (this.inThinking && tokenId === this.thinkTokens.close) {
      this.inThinking = false
    } else if (!this.inThinking && tokenId === this.thinkTokens.open) {
      this.inThinking = true
    }
  }

  /** Switch to the answer phase after the caller injected a forced `</think>`. */
  forceClose(): void {
    this.inThinking = false
  }
}
