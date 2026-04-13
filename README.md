# OpenCoral — Decentralized LLM Desktop Client

OpenCoral is an open-source attempt to build a **decentralized large language model network** as a modern desktop application. Inspired by [Petals](https://github.com/bigscience-workshop/petals), it splits transformer blocks across many machines and chains inference requests through them — but rebuilds the concept from scratch using a contemporary **TypeScript/Bun/Electron** stack, **GGUF** model format via [node-llama-cpp](https://github.com/withcatai/node-llama-cpp), and **js-libp2p** for peer-to-peer networking.

Every running instance is both a **client and a node**: it loads a slice of a large model (e.g. Llama 3.1 70B), serves those blocks to other peers, and earns tokens that pay for its own inference usage. The result is a Goose-style agentic desktop client where users can chat, run tools, and contribute compute — all without centralized GPU infrastructure.

## Key Design Choices

- **Electron + Bun** — single installer, no Python/conda dependency chain
- **llama.cpp (GGUF)** — wide hardware support (CUDA, ROCm, Metal, CPU), efficient quantization (Q4–Q8)
- **js-libp2p (Kademlia DHT)** — standards-based peer discovery, gossip, and encrypted activation-tensor relay
- **Token economy via signed receipts** — nodes earn and spend tokens locally without blockchain or wallet setup
- **Agentic UI** — built-in tools (file I/O, shell, web search) with KV cache kept alive across tool calls

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

```mermaid
flowchart TB
    subgraph Renderer["Renderer Process (React)"]
        UI["App Shell"]
        NV["NetworkView"]
        MP["ModelPanel"]
        MDP["ModelsPanel — browse & manage"]
        BH["BlockHostPanel"]
        CP["ChatPanel"]
        CS["CoverageStatus"]
        MS["ModelSelector"]
        UI --> NV & MP & MDP & BH & CP
        CP --> CS & MS
    end

    subgraph Preload["Preload Bridge"]
        IPC["window.opencoral.* IPC API"]
    end

    subgraph Main["Main Process (Electron + Bun)"]
        IDX["index.ts — IPC Handlers & Lifecycle"]
        ID["Identity — persistent Ed25519 key pair"]
        MM["ModelManager — GGUF parsing & sidecar metadata"]
        HF["HuggingFace — search, download, partial fetch"]
        BHM["BlockHost — load, serve & drive inference"]
        DM["DiscoveredModels — remote model & peer block-range tracking"]
        PL["PeerLatencyTracker — EWMA round-trip estimation"]

        subgraph Inference["Inference Pipeline"]
            TOK["Tokenizer — BPE / SentencePiece + chat templates"]
            SM["SequenceManager — plan chain & check coverage"]
            ABR["AsyncBlockRunner — worker-thread native forward"]
            COV["Coverage — network block availability"]
        end

        subgraph Native["Native C++ Module (NAPI)"]
            MC["ModelContext — GGUF data, sessions"]
            BF["BlockForward — embed, transformer block, logits"]
            HW["Hardware — CUDA / ROCm / Metal / CPU"]
        end
    end

    subgraph P2P["P2P Layer (libp2p)"]
        CN["OpenCoralNode — TCP + Noise (pureJsCrypto) + Yamux + Kad-DHT"]
        DHT["DHT — presence discovery + per-block content routing"]
        BREG["BlockRegistry — periodic DHT re-announcement"]
        IP["InferenceProtocol v1 + v2 — chunked framing & Ed25519-signed tensors"]
        KV["KVProtocol — remote KV-cache sessions (open / forward / close)"]
        MA["ModelAnnounce — peer metadata & block-range exchange"]
        NI["NetworkInspector — topology & stats"]
    end

    subgraph Peers["Remote Peers"]
        PA["Peer A (blocks 0–14)"]
        PB["Peer B (blocks 31–79)"]
    end

    Renderer <-->|"contextBridge"| Preload
    Preload <-->|"ipcRenderer / ipcMain"| Main

    IDX --> ID & MM & HF & BHM & DM & PL
    MM --> SM
    BHM --> ABR & BREG & IP & KV
    SM --> COV
    SM -->|"chain planning"| DM & PL
    ABR --> MC --> BF --> HW

    IDX --> CN
    CN --> DHT & IP & KV & MA & NI
    BREG --> DHT
    KV <-->|"KV-cached hidden-state tensors"| Peers
    IP <-->|"signed tensors (v2)"| Peers
```

### Distributed Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant Chat as ChatPanel
    participant BH as BlockHost (Main)
    participant Tok as Tokenizer
    participant SM as SequenceManager
    participant DM as DiscoveredModels
    participant PL as PeerLatencyTracker
    participant Local as AsyncBlockRunner
    participant KV as KVProtocol
    participant Remote as Remote Peers
    participant Native as Native C++ (llama.cpp)

    User->>Chat: Send prompt
    Chat->>BH: IPC: inference.start
    BH->>Tok: Tokenize (chat template)
    Tok-->>BH: token_ids

    BH->>SM: planChainWithCandidates()
    SM->>DM: getPeerBlockRange() per peer
    DM-->>SM: Peer → block range
    SM->>PL: bestPeer() per block
    PL-->>SM: Latency-ranked chain
    SM-->>BH: chain steps (plan only)

    BH->>KV: openRemoteSession() per remote step
    KV->>Remote: /opencoral/kv/1.0.0 open
    Remote-->>KV: session ack

    rect rgb(235, 245, 255)
        Note over BH,Remote: Prefill Phase (full prompt)
        BH->>Local: embedTokens(promptIds)
        Local->>Native: block_forward (GPU/CPU)
        Native-->>BH: hidden [N × dim]
        BH->>Local: sessionForward (local blocks)
        Local->>Native: block_forward (GPU/CPU)
        Native-->>BH: Updated hidden state
        BH->>KV: forwardRemote (remote blocks)
        KV->>Remote: Stream hidden-state tensor
        Remote-->>KV: Updated hidden state
        KV-->>BH: Result tensor
        BH->>Local: projectToLogits
        Local->>Native: logits projection
        Native-->>BH: logits [vocab_size]
        BH->>BH: sampleTopK (temp + top-k)
    end

    rect rgb(245, 255, 235)
        Note over BH,Remote: Decode Phase (KV-cached, per token)
        loop Each new token
            BH->>Local: sessionForward (single token)
            BH->>KV: forwardRemote (single token)
            KV->>Remote: Forward via KV session
            Remote-->>KV: Result
            BH->>BH: sampleTopK
        end
    end

    BH->>KV: closeRemoteSession()
    KV->>Remote: /opencoral/kv/1.0.0 close
    BH-->>Chat: IPC: inference.token stream
    Chat-->>User: Display response
```

## Status

Early development — this is an experimental rebuild, not a production system.
