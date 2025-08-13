#!/bin/sh
set -eu

# If called with "serve", start uvicorn on 8080 (SageMaker's expectation).
if [ "${1:-}" = "serve" ]; then
  exec python -m uvicorn inference.predict:app --host 0.0.0.0 --port 8080
fi

# Otherwise run whatever was passed.
exec "$@"