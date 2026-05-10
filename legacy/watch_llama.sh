#!/bin/bash
# watch_llama.sh — Legacy llama.cpp slot progress watcher
# Reads journalctl logs and displays per-slot progress bars with ETA.
# Designed to be called repeatedly (e.g. via llama_watch.sh or `watch -n1`).

SERVICE_NAME="llama.service"
STATE_FILE="/tmp/slot_eta_state.txt"

# --- Colors ---
C_PURP="\033[38;2;155;89;182m"
C_GREEN="\033[38;2;46;204;113m"
C_RESET="\033[0m"
C_BOLD="\033[1m"

clear
printf "%b=== TIME: %s ===%b\n" "$C_BOLD" "$(date '+%H:%M:%S')" "$C_RESET"

# --- Memory ---
RAM_USED=$(free -h | awk '/^Mem:/ {print $3 "/" $2}')
VRAM_FILE=$(find /sys/class/drm/card*/device/mem_info_vram_used -type f -print -quit 2>/dev/null)
if [ -n "$VRAM_FILE" ]; then
    VRAM_USED=$(awk '{ printf "%.2f", $1 / 1073741824 }' "$VRAM_FILE")
    VRAM_TOT_FILE="${VRAM_FILE/used/total}"
    if [ -f "$VRAM_TOT_FILE" ]; then
        VRAM_TOT=$(awk '{ printf "%.2f", $1 / 1073741824 }' "$VRAM_TOT_FILE")
        printf "RAM: %s  |  VRAM: %s / %s GiB\n" "$RAM_USED" "$VRAM_USED" "$VRAM_TOT"
    else
        printf "RAM: %s  |  VRAM: %s GiB\n" "$RAM_USED" "$VRAM_USED"
    fi
else
    printf "RAM: %s  |  VRAM: N/A\n" "$RAM_USED"
fi

# --- Recent Logs ---
printf "\n%b=== SERVER LOG (last 10) ===%b\n" "$C_BOLD" "$C_RESET"
journalctl -u "$SERVICE_NAME" -n 100 --no-pager 2>/dev/null \
    | tail -n 10 \
    | sed -E 's/.*llama-server\[[0-9]+\]: //'

# --- Slot Progress ---
printf "\n%b=== SLOT PROGRESS ===%b\n" "$C_BOLD" "$C_RESET"

NOW_MS=$(date +%s%3N)
LOG=$(journalctl -u "$SERVICE_NAME" -n 100 --no-pager 2>/dev/null)
CURRENT=$(echo "$LOG" | awk '
/slot update_slots.*progress =/ {
    match($0, /id +([0-9]+)/, sid)
    match($0, /task ([0-9]+)/, tid)
    match($0, /progress = ([0-9.]+)/, prog)
    if (sid[1] != "" && tid[1] != "" && prog[1] != "") {
        slot_prog[sid[1]] = prog[1] + 0
        slot_task[sid[1]] = tid[1]
    }
}
END {
    for (s in slot_prog) print s, slot_task[s], slot_prog[s]
}')

if [ -z "$CURRENT" ]; then
    printf "No active slots found.\n"
else
    declare -A ST_TASK ST_LAST_TS ST_LAST_PROG ST_EMA_RATE
    if [ -f "$STATE_FILE" ]; then
        while read -r slot task last_ts last_prog ema_rate; do
            ST_TASK[$slot]=$task
            ST_LAST_TS[$slot]=$last_ts
            ST_LAST_PROG[$slot]=$last_prog
            ST_EMA_RATE[$slot]=$ema_rate
        done < "$STATE_FILE"
    fi

    NEW_STATE_LINES=()

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        slot=$(awk '{print $1}' <<< "$line")
        task=$(awk '{print $2}' <<< "$line")
        prog=$(awk '{print $3}' <<< "$line")

        prev_task="${ST_TASK[$slot]}"
        prev_ts="${ST_LAST_TS[$slot]}"
        prev_prog="${ST_LAST_PROG[$slot]}"
        ema_rate="${ST_EMA_RATE[$slot]:-0}"

        if [ "$prev_task" != "$task" ] || [ -z "$prev_ts" ] || awk "BEGIN {exit !($prog < $prev_prog)}"; then
            prev_ts=$NOW_MS
            prev_prog=$prog
            ema_rate=0
            remain_s=""
        else
            delta_p=$(awk "BEGIN {print $prog - $prev_prog}")
            delta_t=$(( NOW_MS - prev_ts ))
            if awk "BEGIN {exit !($delta_p > 0.001 && $delta_t > 500)}"; then
                inst_rate=$(awk "BEGIN {print $delta_p / ($delta_t / 1000.0)}")
                if awk "BEGIN {exit !($ema_rate == 0)}"; then
                    ema_rate=$inst_rate
                else
                    ema_rate=$(awk "BEGIN {print ($ema_rate * 0.6) + ($inst_rate * 0.4)}")
                fi
                prev_ts=$NOW_MS
                prev_prog=$prog
            fi
            if awk "BEGIN {exit !($ema_rate > 0)}"; then
                remain_s=$(awk "BEGIN {print (1.0 - $prog) / $ema_rate}")
            else
                remain_s=""
            fi
        fi

        NEW_STATE_LINES+=("$slot $task $prev_ts $prev_prog $ema_rate")

        awk -v slot="$slot" -v p="$prog" -v rem="$remain_s" 'BEGIN {
            w = 40
            filled = int(p * w)
            if (filled < 0) filled = 0
            if (filled > w) filled = w
            bar = ""
            for (i = 0; i < filled; i++) bar = bar "\xe2\x96\x88"
            for (i = filled; i < w; i++) bar = bar "\xe2\x96\x91"
            if (p >= 0.999) {
                eta = "DONE"
            } else if (rem != "") {
                mins = int(rem / 60)
                secs = int(rem % 60)
                eta = sprintf("ETA %dm %02ds", mins, secs)
            } else {
                eta = "ETA calculating..."
            }
            printf "Slot %-2s [%s] %5.1f%%  %s\n", slot, bar, p * 100, eta
        }'
    done <<< "$CURRENT"

    printf '%s\n' "${NEW_STATE_LINES[@]}" > "$STATE_FILE"
fi

printf "\n%bPress 'q' to exit%b\n" "$C_GREEN" "$C_RESET"
