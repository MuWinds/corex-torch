#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
rm -f $HOME/build/scan24c.log
setsid nohup ninja -k 50 > $HOME/build/scan24c.log 2>&1 < /dev/null &
echo PID=$! > $HOME/build/scan24c.pid
sleep 2
cat $HOME/build/scan24c.pid