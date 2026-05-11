#!/bin/sh
# =============================================================================
# Container entrypoint
# =============================================================================
# 1. Seed the persistent volume (mounted at /srv on Railway) from the
#    bootstrap copies baked into the image at /app/bootstrap. Only fires
#    when the target dir is empty — once the cron has populated the
#    volume, the bootstrap copies are inert.
# 2. Exec whatever was passed in (the cron service overrides CMD; the web
#    service uses the default uvicorn invocation).
# =============================================================================
set -e

seed_if_empty() {
  SRC="$1"
  DST="$2"
  if [ -d "$SRC" ]; then
    mkdir -p "$DST"
    # `ls -A` returns non-empty when DST has any entry (incl. dotfiles).
    if [ -z "$(ls -A "$DST" 2>/dev/null)" ]; then
      echo "[start.sh] seeding $DST from $SRC"
      cp -r "$SRC"/. "$DST"/
    else
      echo "[start.sh] $DST already populated — skipping bootstrap seed"
    fi
  fi
}

# OUTPUT_DIR / DATA_DIR are set in the Dockerfile (default /srv/outputs and
# /srv/data). The bootstrap dirs are baked at /app/bootstrap/{outputs,data}.
seed_if_empty /app/bootstrap/outputs "${OUTPUT_DIR:-/srv/outputs}"
seed_if_empty /app/bootstrap/data    "${DATA_DIR:-/srv/data}"

# If a startCommand was provided (cron service), exec it. Otherwise run the
# web service's default uvicorn invocation.
if [ $# -gt 0 ]; then
  exec "$@"
fi

cd /app/src
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
