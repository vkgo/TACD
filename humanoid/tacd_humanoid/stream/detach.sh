#!/usr/bin/env bash
# Start a long-lived service detached from the caller, with stdin on a FIFO (operator keys) and its
# own log.  Use as:  setsid nohup detach.sh PIDFILE LOGFILE FIFO -- command args... &
# The PID written to PIDFILE is the PID of the final command (this script execs into it).
set -euo pipefail
pidfile=$1 log=$2 fifo=$3
[ "$4" = "--" ] || { echo "usage: detach.sh PIDFILE LOGFILE FIFO -- cmd..." >&2; exit 2; }
shift 4
[ -p "$fifo" ] || mkfifo "$fifo"
echo $$ > "$pidfile"
exec 0<>"$fifo" >"$log" 2>&1
exec "$@"
