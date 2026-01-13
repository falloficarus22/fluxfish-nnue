#!/bin/bash
cd /root/fluxfish-nnue
if [ -f "/root/icarus_env/bin/python3" ]; then
    exec /root/icarus_env/bin/python3 uci.py
else
    exec python3 uci.py
fi
