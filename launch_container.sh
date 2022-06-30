#!/bin/bash
set -e
set -o pipefail

function warning()
{
    yellow="33m"
    echo -e "\033[${yellow}[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: $* \033[0m" 2>&1
}

function error() {
    local red="31m"
    echo -e "\033[${red}[$(date +'%Y-%m-%dT%H:%M:%S%z')] ERROR: $* \033[0m" 1>&2
    exit 1
}

function info() {
    green="32m"
    echo -e "\033[${green}[$(date +'%Y-%m-%dT%H:%M:%S%z')] INFO: $* \033[0m" 2>&1
}

function usage() {
    echo "Usage: bash launch_container.sh [OPTIONS...]"
    echo 
    echo "      -h, --help        show helps"
    echo "      -f, --force       force rebuild image"
    echo "      -r, --root        run with root user"
    exit 1
}

function build_image() {
    docker build --tag nvbench:dev -f 
}

function launch_container() {
    if which git > /dev/null 2>&1; then
        NVBENCH_HOME=$( git rev-parse --show-toplevel )
    else
        NVBENCH_HOME="${PWD}"
        if basename "${NVBENCH_HOME}" != "nvbench"; then
            error "export NVBENCH_HOME=<path-to-nvbench>"
        fi
    fi

    # default values
    local force_rebuild_image=false
    local container_name="nvbench_${USER}"
    local login_user
    login_user="$(whoami)"
    local image=nvbench:dev

    while (($#)); 
    do
        case "$1" in
        -r | --root)
            login_user="root"
            shift 1
            ;;
        -f | --force)
            force_rebuild_image=true
            shift 1
            ;;
        *)
          usage
          ;;
        esac
      done

    # check docker
    if ! which docker &>/dev/null; then
        error "command 'docker' not found"
    fi

    # check image, pull if not exist
    if [[ ! "$(docker image ls -aq "${image}")" ]] || [[ "${force_rebuild_image}" == true ]]; then
        info "build ${image}"
        info "$(docker build --build-arg NVBENCH_HOME="${NVBENCH_HOME}" --tag "${image}" "${NVBENCH_HOME}/docker/dev")"
    fi

    # check container, clean it if exist
    if [[ $(docker ps -aq -f name="${container_name}") ]]; then
        # cleanup
        info "$(docker container rm --force "${container_name}") stop and removed"
    fi

    # create and start container
    if ! docker create --gpus all --rm --interactive --tty --name "${container_name}" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v "${NVBENCH_HOME}":/nvbench --hostname="${container_name}" "${image}" &>/dev/null; then
        error "create container with image ${image} failed"
    fi
    info "$(docker start "${container_name}") start"

    # install sudo if command not exit
    if ! docker exec --user root "${container_name}" which sudo &>/dev/null; then
      docker exec --user root "${container_name}" bash -c "apt-get update"
      info "install sudo..."
      info "$(docker exec --user root "${container_name}" bash -c "apt-get install -y --no-install-recommends sudo")"
    fi

    # create user
    user_name="$(whoami)"
    user_id="$(id --user "${user_name}")"
    group_name="$(id --group --name "${user_name}")"
    group_id="$(id --group "${user_name}")"
    docker exec --user root "${container_name}" bash -c "echo root:root | chpasswd"
    docker exec --user root "${container_name}" groupadd -f --g "${group_id}" "${group_name}" &> /dev/null
    docker exec --user root "${container_name}" useradd  -G sudo -g "${group_id}" --uid "${user_id}" -m "${user_name}" &>/dev/null
    docker exec --user root "${container_name}" bash -c "echo ${user_name}:${user_name} | chpasswd" &> /dev/null
    docker exec --user root "${container_name}" bash -c "echo -e \"\n%sudo ALL=(ALL:ALL) NOPASSWD:ALL\n\" >> /etc/sudoers" &> /dev/null

    # set up default shell
    docker exec --user root "${container_name}" bash -c "chsh -s /bin/bash ${user_name}"

    # enter container
    docker exec --user "${login_user}" --interactive --tty --workdir /nvbench "${container_name}" /bin/bash

    # stop and remove container when exit
    info "$(docker container rm --force "${container_name}") stop and removed"
}

launch_container "$@"