#/bin/bash
if [ ! -n "$1" ];then
    echo "You must set project name"
    exit -1
fi
project="# "$1
echo $project >> README.md
git init
