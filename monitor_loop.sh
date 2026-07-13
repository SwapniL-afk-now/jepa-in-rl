#!/bin/bash
LOG=/workspace/exploration/monitor_status.log
MJSON=/workspace/exploration/drgrpo_pass8_math_pool.json
CJSON=/workspace/exploration/drgrpo_pass8_code_pool.json
MLOG=/workspace/exploration/scratch_math.log
CLOG=/workspace/exploration/scratch_code.log
for i in $(seq 1 60); do
  TS=$(date '+%H:%M:%S')
  pgrep -f eval_math_benchmarks >/dev/null && MA=alive || MA=dead
  pgrep -f eval_code_benchmarks >/dev/null && CA=alive || CA=dead
  [ -f "$MJSON" ] && MJ=yes || MJ=no
  [ -f "$CJSON" ] && CJ=yes || CJ=no
  GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | paste -sd'|' -)
  MP=$(grep -iE 'seed|MACRO|benchmark|it/s|generation' "$MLOG" 2>/dev/null | tail -1 | cut -c1-120)
  CP=$(grep -iE 'seed|MACRO|benchmark|it/s|generation' "$CLOG" 2>/dev/null | tail -1 | cut -c1-120)
  {
    echo "[$TS] i=$i math=$MA(json:$MJ) code=$CA(json:$CJ) gpu=[$GPU]"
    echo "    M> $MP"
    echo "    C> $CP"
  } >> "$LOG"
  if [ -f "$MJSON" ] && [ -f "$CJSON" ] && [ "$MA" = dead ] && [ "$CA" = dead ]; then
    echo "[$TS] BOTH DONE at i=$i" >> "$LOG"; break
  fi
  sleep 30
done
echo "[$(date '+%H:%M:%S')] LOOP_EXIT" >> "$LOG"
