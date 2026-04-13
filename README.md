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
        UI["App Shell (4 tabs)"]
        NV["NetworkView"]
        MP["ModelPanel"]
        BH["BlockHostPanel"]
        CP["ChatPanel"]
        UI --> NV & MP & BH & CP
    end

    subgraph Preload["Preload Bridge"]
        IPC["window.opencoral.* IPC API"]
    end

    subgraph Main["Main Process (Electron + Bun)"]
        IDX["index.ts — IPC Handlers & Lifecycle"]
        MM["ModelManager — GGUF parsing & sidecar metadata"]
        HF["HuggingFace — search, download, partial fetch"]
        BHM["BlockHost — load & serve local blocks"]

        subgraph Inference["Inference Pipeline"]
            TOK["Tokenizer — BPE / SentencePiece + chat templates"]
            SM["SequenceManager — plan chain, prefill & decode"]
            BR["BlockRunner — native forward pass"]
            COV["Coverage — network block availability"]
        end

        subgraph Native["Native C++ Module (NAPI)"]
            MC["ModelContext — GGUF data, sessions"]
            BF["BlockForward — embed, transformer block, logits"]
            HW["Hardware — CUDA / ROCm / Metal / CPU"]
        end
    end

    subgraph P2P["P2P Layer (libp2p)"]
        CN["OpenCoralNode — TCP + Noise + Yamux + Kad-DHT"]
        DHT["DHT — per-block provider records"]
        IP["InferenceProtocol — binary tensor streaming"]
        MA["ModelAnnounce — peer metadata exchange"]
        DM["DiscoveredModels — remote model tracking"]
        NI["NetworkInspector — topology & stats"]
    end

    subgraph Peers["Remote Peers"]
        PA["Peer A (blocks 0–14)"]
        PB["Peer B (blocks 31–79)"]
    end

    Renderer <-->|"contextBridge"| Preload
    Preload <-->|"ipcRenderer / ipcMain"| Main

    IDX --> MM & HF & BHM
    BHM --> BR
    SM --> BR
    SM --> COV
    SM -->|"remote blocks"| IP
    BR --> MC --> BF --> HW

    IDX --> CN
    CN --> DHT & IP & MA & DM & NI
    IP <-->|"hidden-state tensors"| Peers
```

### Distributed Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant Chat as ChatPanel
    participant Main as Main Process
    participant Tok as Tokenizer
    participant SM as SequenceManager
    participant DHT as DHT Lookup
    participant Local as Local BlockRunner
    participant Remote as Remote Peers
    participant Native as Native C++ (llama.cpp)

    User->>Chat: Send prompt
    Chat->>Main: IPC: inference.start
    Main->>Tok: Tokenize (chat template)
    Tok-->>SM: token_ids

    SM->>DHT: Find providers for each block
    DHT-->>SM: Block → Peer mapping

    rect rgb(235, 245, 255)
        Note over SM,Remote: Prefill Phase (full prompt)
        SM->>Local: Embed tokens → hidden state
        Local->>Native: block_forward (GPU/CPU)
        Native-->>Local: hidden [N × dim]
        SM->>Remote: Stream hidden state (blocks 0-14)
        Remote-->>SM: Updated hidden state
        SM->>Local: Forward pass (blocks 15-30)
        Local->>Native: block_forward (GPU/CPU)
        Native-->>SM: Updated hidden state
        SM->>Remote: Stream hidden state (blocks 31-79)
        Remote-->>SM: Final hidden state
        SM->>Local: Project to logits
        Local->>Native: logits projection
        Native-->>SM: logits [vocab_size]
        SM->>SM: Sample token (temp + top-k)
    end

    rect rgb(245, 255, 235)
        Note over SM,Remote: Decode Phase (KV-cached, per token)
        loop Each new token
            SM->>Local: Forward single token (KV cache)
            SM->>Remote: Forward single token (KV cache)
            SM->>SM: Sample next token
        end
    end

    SM-->>Main: Generated text + timing
    Main-->>Chat: IPC: inference.token stream
    Chat-->>User: Display response
```

## Status

Early development — this is an experimental rebuild, not a production system.
