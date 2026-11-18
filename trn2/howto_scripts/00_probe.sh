#!/bin/bash
mkdir -p $HOME/howto
hostname; whoami; nproc; free -g | head -2
which python python3; python3 --version
ls /dev/neuron* | head -3; ls /dev/neuron* | wc -l
neuron-ls 2>&1 | head -30
pip list 2>/dev/null | grep -i -E "torch|neuron|nki|vllm|nkilib" 
env | grep -i -E "neuron|vllm|xla" | sort
ls /tmp 2>&1 | head; df -h /tmp $HOME /tmp 2>&1
neuron-top -h 2>&1 | head -2; ps aux | grep -i -E "python|neuron" | grep -v grep | head -20
which neuron-profile neuron-explorer 2>&1
