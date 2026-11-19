#!/bin/bash
cd $HOME/build
rm -f torch24-src.tgz
tar czf torch24-src.tgz --exclude=torch24/build --exclude=torch24/dist --exclude=torch24/.git --exclude='*.pyc' torch24 2>/dev/null
echo tar_rc=$?
ls -la torch24-src.tgz