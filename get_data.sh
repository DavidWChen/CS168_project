#!/usr/bin/env bash

if [ -d mnist_data ]; then
    echo "mnist_data directory already present, exiting"
    exit 1
fi

mkdir mnist_data
wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=mnist_data --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd mnist_data
gunzip *
popd