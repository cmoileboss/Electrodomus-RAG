FROM pgvector/pgvector:pg18

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        postgresql-server-dev-18 \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build pg_textsearch from source (pre-built binaries require GLIBC >= 2.38)
RUN git clone --branch v1.4.0 --depth 1 https://github.com/timescale/pg_textsearch.git /tmp/pg_textsearch \
    && cd /tmp/pg_textsearch \
    && make \
    && make install \
    && rm -rf /tmp/pg_textsearch

COPY init.sql /docker-entrypoint-initdb.d/init.sql