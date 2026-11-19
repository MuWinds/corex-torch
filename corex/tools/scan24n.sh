#!/bin/bash
export PATH=$HOME/venvs/torchbuild310/bin:$PATH
cd $HOME/build/torch24/build || exit 1
ninja -k 50 > $HOME/build/scan24n.log 2>&1
echo ninja_rc=$?
echo failed=$(grep -c FAILED: $HOME/build/scan24n.log)
grep FAILED: $HOME/build/scan24n.log | sed s/.*torch24..// | sed s/.o..// | sort -u | head -6