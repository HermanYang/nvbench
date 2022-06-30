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

usage () {
  echo "USAGE: run.sh <benchmark>"
  echo
  echo "OPTIONS:"
  echo "      -b, --batch_size                      batch size                          [default: 1]"
  echo "      -w, --worker_number                   cards to use, one card per worker   [default: 1]"
  echo "      -l, --index_number_per_lookup         [default: 1]"
  echo "      -i, --interation                      [default: 10]"
  echo "      -h, --help                            show usage"
}

model_path="$(dirname "$0")"

function run(){
    export OMP_NUM_THREADS=4
    local batch_size=1
    local worker_number=1
    local table_size=47000000
    local index_number_per_lookup=1
    local iteration=10

    while (("$#")); do
        case "$1" in
            -b | --batch_size)
                shift
                batch_size="$1"
                shift
                ;;
            -w | --worker_number)
                shift
                worker_number="$1"
                shift
                ;;
            -l | --index_number_per_lookup)
                shift
                index_number_per_lookup="$1"
                shift
                ;;
            -i | --iteration)
                shift
                iteration="$1"
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

    if [[ "${worker_number}" == 1 ]]; then 
        python "${model_path}/"run.py --mlp-bottom 128-64-32 \
            --embedding "${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}" \
            --mlp-top 128-32-1 \
            --category-feature-size 32 \
            --max-embedding-index ${table_size} \
            --iterations "${iteration}" \
            --batch-size "${batch_size}" \
            --index-number-per-lookup "${index_number_per_lookup}"  \
            --mode latency
    else
        torchrun --nproc_per_node="${worker_number}" \
            "${model_path}/"run.py --mlp-bottom 128-64-32 \
            --embedding "${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}-${table_size}" \
            --mlp-top 128-32-1 \
            --category-feature-size 32 \
            --max-embedding-index ${table_size} \
            --iterations "${iteration}" \
            --batch-size "${batch_size}"\
            --index-number-per-lookup "${index_number_per_lookup}" \
            --mode latency
    fi
}

run "$@"