FROM nvcr.io/nvidia/pytorch:22.05-py3

# Install userful applications
RUN apt-get update &&                           \
    apt-get install -y --no-install-recommends  \ 
    sudo                                        \
    wget                                        \
    curl                                        \
    ssh                                         \
    tree                                        \
    htop                                        \
    locate                                      \
    bash-completion                             \
    git                                         \
    vim                                         \
    zip                                         \
    unzip                                       \
    shellcheck                                  

RUN apt-get update && apt-get upgrade -y --no-install-recommends