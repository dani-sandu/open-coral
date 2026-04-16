FROM oven/bun:alpine AS base

WORKDIR /app/bootstrap-node

COPY bootstrap-node/package.json bootstrap-node/bun.lock ./
RUN bun install --frozen-lockfile --production

COPY bootstrap-node/index.ts bootstrap-node/tsconfig.json ./

# Copy source files imported from the parent src/ tree
COPY src/p2p/key-registry.ts src/p2p/stream-utils.ts src/p2p/ed25519-helpers.ts /app/src/p2p/
COPY src/main/identity.ts /app/src/main/

EXPOSE 4001

VOLUME ["/data"]

CMD ["bun", "run", "index.ts"]
