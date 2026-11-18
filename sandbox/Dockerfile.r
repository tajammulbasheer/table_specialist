# Ours: the repo's sandbox/Dockerfile.r (r-base by digest, non-root) plus dplyr, which the repo's NL-to-R prompt
# (Generation/NL2DSL/NL2R/nl2r_generation_v6.py) puts in every script: "library(dplyr)". Debian's packaged build.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends r-cran-dplyr && rm -rf /var/lib/apt/lists/* \
    && Rscript -e 'library(dplyr); cat(as.character(packageVersion("dplyr")), "\n")'
USER 65534:65534
