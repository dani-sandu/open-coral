# OpenCoral — Decentralized LLM Desktop Client

<a href="https://github.com/dani-sandu/open-coral/releases"><img src="https://img.shields.io/github/v/release/dani-sandu/open-coral?include_prereleases&style=for-the-badge" alt="GitHub release"></a>

OpenCoral is an open-source attempt to build a **decentralized large language model network** as a modern desktop application. Inspired by [Petals](https://github.com/bigscience-workshop/petals), it splits transformer blocks across many machines and chains inference requests through them — but rebuilds the concept from scratch using a contemporary **TypeScript/Bun/Electron** stack, **GGUF** model format via [llama.cpp](https://github.com/ggerganov/llama.cpp), and **js-libp2p** for peer-to-peer networking.

Every running instance is both a **client and a node**: it loads a slice of a large model (e.g. Llama 3.1 70B), serves those blocks to other peers, and contributes compute — all without centralized GPU infrastructure.

## Key Design Choices

- **Electron + Bun** — single installer, no Python/conda dependency chain
- **llama.cpp (GGUF)** — wide hardware support (CUDA, ROCm, Metal, CPU), efficient quantization (Q4–Q8)
- **js-libp2p (Kademlia DHT)** — standards-based peer discovery, gossip, and encrypted activation-tensor relay

## Supported Models

Any GGUF model on HuggingFace that llama.cpp supports. Initial targets: **Llama 3.1 70B/405B**, **Mixtral 8x22B**, **Qwen 2.5 72B**, and **Gemma 3 27B**.

## Hardware

| Hardware | Support |
|----------|---------|
| NVIDIA GPU (CUDA) | Full — highest throughput and token earnings |
| AMD GPU (ROCm) | Full via llama.cpp ROCm backend |
| Apple Silicon (Metal) | Full via llama.cpp Metal backend |
| CPU only | Supported — fewer blocks, lower throughput weight |

Minimum: 8 GB RAM (CPU) or 8 GB VRAM (GPU).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full overview of the system structure and diagrams.

## Status

Early development — this is an experimental rebuild, not a production system.

## Roadmap

### Phase 1 
- [x] **P1-1** `sampleTopK` typed-array partial sort — eliminate 65M heap allocs/response (`sampler.ts`)
- [x] **P1-2** Batch `decodeToken` IPC — collapse 512 worker round-trips to batches of 4–8 (`inference-orchestrator.ts`)
- [x] **P1-3** `Promise.all` peer queries — 300ms serial peer latency → ~30ms parallel (`sequence-manager.ts`)
- [x] **P1-4** Remove spread copy in ngram lookup — O(n) → O(1) per token (`speculative-session.ts`)
- [x] **P1-5** Cache / skip chain plan — eliminate wasted DHT query before every inference (`block-host.ts`, `chain-plan-cache.ts`)

### Phase 2
- [ ] **P2-1** Float16 hidden-state wire format — 2× wire-size reduction per hop (`inference-protocol.ts`)
- [ ] **P2-2** Pre-dial next peer during compute — remove 1 handshake RTT per hop (`sequence-manager.ts`)
- [ ] **P2-3** Zero-copy input tensor transfer — remove memcpy on every forward call (`native-worker.ts`)
- [ ] **P2-4** FlashAttention compile flag — ~2× prefill speedup on long contexts (build config)
- [ ] **P2-5** Background routing table refresh — fully eliminate DHT from inference hot path (`sequence-manager.ts`)
- [ ] **P2-6** StreamingLLM KV eviction policy — constant KV memory regardless of conversation length (`kv-session-registry.ts`)
- [ ] **P2-7** NgramCache cleanup on session end — reduce GC pressure on long sessions (`speculative-session.ts`)
- [ ] **P2-8** DDTree-style tree verification — +20–40% speculative acceptance length, no native changes (`speculative-session.ts`)
- [ ] **P2-9** Per-phase latency profiler — IPC / network / sig-verify / forward breakdown per hop

### Phase 3
- [ ] **P3-1** Prefill/decode role separation + chunked prefill — disaggregated pipeline, pull-based KV transfer
- [ ] **P3-2** Two-level scheduler (DynaServe pattern) — global split-point assignment + local peer batching
- [ ] **P3-3** FourierCompress / SLICER activation compression — 7.6–10× wire-size reduction *(if network-bound)*
- [ ] **P3-4** DFlash + `sessionDecodeHiddenLayers` native API — 7–8× speculative speedup *(if compute-bound)*
- [ ] **P3-5** Continuous batching + PagedAttention — 2–4× multi-user throughput (native addon)
- [ ] **P3-6** SpeCache distributed KV prefetch — hide remote KV transfer latency behind compute
