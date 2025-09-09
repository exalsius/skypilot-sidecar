#!/bin/sh
set -e

home="$(cd -- "$(dirname -- "$0")"; pwd)"

uv run "$home/src/main.py"