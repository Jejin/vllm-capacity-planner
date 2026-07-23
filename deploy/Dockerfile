# vLLM Capacity Planner — container image (build the SPA, serve API + SPA from one process).
FROM node:24-slim
WORKDIR /app
COPY package.json ./
COPY domain/package.json domain/
COPY server/package.json server/
COPY web/package.json web/
RUN npm install
COPY . .
RUN npm run build -w web
ENV NODE_ENV=production PORT=8080 DATA_FILE=/data/catalog.json WEB_DIST=/app/web/dist
EXPOSE 8080
# /healthz + /readyz for probes
CMD ["node", "node_modules/tsx/dist/cli.mjs", "server/src/index.ts"]
