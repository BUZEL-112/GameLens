# ── Build Stage ────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies first (better cache)
COPY package.json package-lock.json* ./
RUN npm ci --prefer-offline

# Copy source and data
COPY . .

# Ensure images mount point exists
RUN mkdir -p /app/public/images

# Build Next.js
RUN npm run build

# ── Run Stage ──────────────────────────────────────────────────────────────────
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Copy built output
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public
COPY --from=builder /app/data ./data
COPY --from=builder /app/next.config.mjs ./next.config.mjs

# Ensure images directory exists for volume mount
RUN mkdir -p /app/public/images

EXPOSE 3000

CMD ["npx", "next", "start"]
