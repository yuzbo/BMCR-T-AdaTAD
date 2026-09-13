#!/bin/bash
# Resume existing B/MQ partial targets after observed SSH transport disconnects.
set -euo pipefail
cd /c/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/paper_20260913
asset="$1"
case "$asset" in adatad_anet_b.pth|internvideo1_mq.pth) ;; *) exit 2;; esac
for attempt in {1..12}; do
  if printf '%s\n' "reput research/paper/assets/$asset /data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913/research/paper/assets/$asset" |
    sftp -b - -B 65536 -R 128 -F 'C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config' -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=5 bcr-4090; then
    exit 0
  fi
  printf 'Transport interrupted; resume attempt %s for %s\n' "$attempt" "$asset"
  sleep 10
done
exit 1
