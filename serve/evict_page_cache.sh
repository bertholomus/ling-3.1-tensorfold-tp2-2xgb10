#!/usr/bin/env bash
# Evict a rank folder's page cache every 5 s while the server loads and runs, so the streamed weight read does not
# fill the page cache the GPU allocates from (a GB10's CPU and GPU share one memory). posix_fadvise(DONTNEED) needs no
# root on files you can read. Stops when STOPFILE appears.
# usage: evict_page_cache.sh 'RANK_DIR/*.safetensors' STOPFILE
set -u
GLOB=${1:?glob}; STOP=${2:?stopfile}
while [ ! -f "$STOP" ]; do
  for f in $GLOB; do
    [ -f "$f" ] || continue
    python3 - "$f" <<'PY' 2>/dev/null
import ctypes, os, sys
f = os.open(sys.argv[1], os.O_RDONLY)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.posix_fadvise(f, 0, 0, 4)   # POSIX_FADV_DONTNEED
os.close(f)
PY
  done
  sleep 5
done
