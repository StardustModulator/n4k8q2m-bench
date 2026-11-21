FROM rust:1-bookworm AS build
WORKDIR /src
RUN rustup component add llvm-tools-preview
COPY Cargo.toml Cargo.lock ./
COPY src ./src
# Profile-guided build: instrument, run on self-generated input shaped like
# the task's files, then rebuild with the profile.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C profile-generate=/tmp/pgo"
RUN cargo build --release --bin solve --bin traingen \
 && ./target/release/traingen /tmp/train 1500000 \
 && mkdir -p /tmp/out8 /tmp/out1 \
 && DATA_DIR=/tmp/train OUTPUT_DIR=/tmp/out8 NCPU=8 ./target/release/solve \
 && DATA_DIR=/tmp/train OUTPUT_DIR=/tmp/out1 NCPU=1 ./target/release/solve \
 && "$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata" merge -o /tmp/pgo.profdata /tmp/pgo \
 && rm -rf /tmp/train /tmp/out8 /tmp/out1
ENV RUSTFLAGS="-C target-cpu=x86-64-v3 -C profile-use=/tmp/pgo.profdata"
RUN cargo build --release --bin solve

FROM debian:bookworm-slim
COPY --from=build /src/target/release/solve /solve
# Fresh pages for gigabytes of records are cheaper as huge pages.
ENV MIMALLOC_ALLOW_LARGE_OS_PAGES=1
ENTRYPOINT ["/solve"]
