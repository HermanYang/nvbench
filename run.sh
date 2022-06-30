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
  echo "      -n, --model                    morel name     [default: dlrm]"
  echo "      -o, --output                   output name    [default: <model name>]"
  echo "      -h, --help                     show usage"
}

function run() {
    local model="dlrm"
    local output="${model}"

    while (("$#")); do
        case "$1" in
            -m | --model)
                shift
                model="$1"
                shift
                ;;
            -o | --output)
                shift
                output="$1"
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

    NVBENCH_HOME="$(dirname "$0")"

    info "install requirements..."
    pip install -r "${NVBENCH_HOME}/requirements.txt" > /dev/null
    pip install -r "${NVBENCH_HOME}/models/${model}/requirements.txt" > /dev/null

    # outdir="/tmp/${output}"
    outdir="${output}"
    rm -rf "${outdir}" && mkdir "${outdir}"

    worker_number=4
    batch_size_list=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096)

    info "run benchmark ${model}..."

    info "System Level profiling..."
    for batch_size in "${batch_size_list[@]}"
    do
        command="bash ${NVBENCH_HOME}/models/${model}/run.sh --worker_number ${worker_number} --batch_size ${batch_size}"
        nsys profile --trace=cuda,nvtx,cublas --output "${outdir}/${model}_${worker_number}_${batch_size}" --gpu-metrics-device=all --force-overwrite true ${command}
        nsys export --type sqlite --output "${outdir}/${model}_${worker_number}_${batch_size}.sqlite" "${outdir}/${model}_${worker_number}_${batch_size}.nsys-rep"
    done

    # TODO
    info "Device Level Profiling..."

    info "Power Draw Sampling..."
    for batch_size in "${batch_size_list[@]}"
    do
        command="bash ${NVBENCH_HOME}/models/${model}/run.sh --worker_number ${worker_number} --batch_size ${batch_size}"
        bash "${NVBENCH_HOME}/cuprof/power_sampler" "${command}" "${outdir}/${model}_${worker_number}_${batch_size}_power_draw"
    done

    info "Analyzing..."
    python "${NVBENCH_HOME}/analyzer" --input "${output}" --output "${output}"

    info "Creating output file"
    tar czf "${output}.tar.gz" "${outdir}"

    info "Done"
}

run "$@"