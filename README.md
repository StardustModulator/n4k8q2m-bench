Private x86 timing harness for the Shopping Sessions solver. Dispatch `bench.yml`
with the solver commits to compare; it generates the practice-sized dataset with
the kit's generator, checks each commit on the dev set, and times the Docker
image like the scorer does (container start to exit).
