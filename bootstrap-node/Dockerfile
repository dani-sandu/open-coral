FROM oven/bun:alpine AS base

WORKDIR /app

COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

COPY index.ts tsconfig.json ./

EXPOSE 4001

VOLUME ["/data"]

CMD ["bun", "run", "index.ts"]
