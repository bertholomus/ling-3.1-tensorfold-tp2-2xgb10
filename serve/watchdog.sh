#!/usr/bin/env bash
# Stop the server before its node runs out of memory (a GB10's CPU and GPU share one memory): sample the node's
# MemAvailable every 2 s and SIGKILL the server's PID the moment it falls under FLOOR_GIB (default 1.5). With CONTAINER,
# PID is the server's PID inside that container and the kill runs there (docker exec). Stops when the server exits or
# STOPFILE appears.
# usage: watchdog.sh PID STOPFILE LOG [FLOOR_GIB] [CONTAINER]
set -u
PID=${1:?pid}; STOP=${2:?stopfile}; LOG=${3:?log}; FLOOR=${4:-1.5}; CONT=${5:-}
run() { if [ -n "$CONT" ]; then docker exec "$CONT" "$@"; else "$@"; fi; }
echo "$(date -u +%FT%TZ) watchdog armed for pid $PID${CONT:+ in $CONT}, floor $FLOOR GiB" >> "$LOG"
while [ ! -f "$STOP" ]; do
  if ! run kill -0 "$PID" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) pid $PID gone" >> "$LOG"
    break
  fi
  MA=$(awk '/MemAvailable/{printf "%.2f", $2/1048576}' /proc/meminfo)
  if awk -v a="$MA" -v f="$FLOOR" 'BEGIN { exit !(a < f) }'; then
    if run kill -9 "$PID" 2>/dev/null; then
      echo "$(date -u +%FT%TZ) KILLED pid $PID: MemAvailable $MA GiB < $FLOOR" >> "$LOG"
    else
      echo "$(date -u +%FT%TZ) pid $PID already gone (MemAvailable $MA GiB)" >> "$LOG"
    fi
    break
  fi
  sleep 2
done
echo "$(date -u +%FT%TZ) watchdog done" >> "$LOG"
