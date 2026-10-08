# syntax=docker/dockerfile:1
# llmon — single static-ish binary, tiny runtime image.
#   docker build -t llmon .
#   docker run -p 11435:11435 -v llmon-models:/root/.llmon/models llmon
FROM rust:1.82-slim AS build
WORKDIR /src
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release --locked && strip target/release/llmon

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/llmon /usr/local/bin/llmon
# Performance-first defaults (fix.md §6); override at `docker run -e`.
ENV LLMON_HOST=0.0.0.0 LLMON_PORT=11435
ENV LLMON_CTX=2048 LLMON_BATCH=512 LLMON_UBATCH=512
ENV LLMON_FLASH_ATTN=auto LLMON_CACHE_TYPE_K=f16 LLMON_CACHE_TYPE_V=f16
ENV LLMON_SPEC_TYPE=ngram-mod LLMON_KEEP_ALIVE=600
EXPOSE 11435
VOLUME ["/root/.llmon/models"]
ENTRYPOINT ["llmon", "serve"]
