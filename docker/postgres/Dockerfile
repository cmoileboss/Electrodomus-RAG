FROM postgres:17

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN git clone --branch v0.8.0 --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Install pg_textsearch
RUN git clone --branch v1.4.0 --depth 1 https://github.com/timescale/pg_textsearch.git /tmp/pg_textsearch \
    && cd /tmp/pg_textsearch \
    && make \
    && make install \
    && rm -rf /tmp/pg_textsearch

COPY init.sql /docker-entrypoint-initdb.d/init.sql
