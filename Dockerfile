FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
ENV RUSTFLAGS="-C target-cpu=cascadelake -C llvm-args=-x86-branches-within-32B-boundaries"
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=build /src/target/release/solve /solve
# Fresh pages for gigabytes of records are cheaper as huge pages.
ENV MIMALLOC_ALLOW_LARGE_OS_PAGES=1
ENTRYPOINT ["/solve"]
