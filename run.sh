#!/bin/bash
set -e
set -o pipefail

function warning()
{
    yellow="33m"
    echo -e "\033[${yellow}[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: $* \033[0m" >&1
}

function error() {
    local red="31m"
    echo -e "\033[${red}[$(date +'%Y-%m-%dT%H:%M:%S%z')] ERROR: $* \033[0m" >&2
    exit 1
}

function info() {
    green="32m"
    echo -e "\033[${green}[$(date +'%Y-%m-%dT%H:%M:%S%z')] INFO: $* \033[0m" >&1
}

if which git > /dev/null 2>&1; then
    NVBENCH_HOME=$( git rev-parse --show-toplevel )
else
    NVBENCH_HOME="${PWD}"
    if basename "${NVBENCH_HOME}" != "nvbench"; then
        error "export NVBENCH_HOME=<path-to-nvbench>"
    fi
fi

usage () {
  echo "USAGE: run.sh <benchmark>"
  echo
  echo "OPTIONS:"
  echo "      -n, --model                    morel name"
  echo "      -h, --help                     show usage"
}

function run() {
    local model="dlrm"

    while (("$#")); do
        case "$1" in
            -m | --model)
                shift
                model="$1"
                shift
                ;;
            -h | --help)
                usage
                exit 1
                ;;
            *)
                usage
                error "-- Invalidate options ${1}, use -h or --help"
                ;;
        esac
    done

    pip install -r "${NVBENCH_HOME}/requirements.txt"
    pip install -r "${NVBENCH_HOME}/models/${model}/requirements.txt"
    bash "${NVBENCH_HOME}models/${model}/run.sh"
}

run "$@"