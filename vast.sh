#!/bin/bash

sudo add-apt-repository ppa:fish-shell/release-4 -y

apt-get update && \
    apt-get install -y --no-install-recommends

apt-get install -y --no-install-recommends \
    silversearcher-ag fish vim gdb cmake wget nvtop htop

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# activate uv in current shell
source $HOME/.local/bin/env

echo /usr/local/bin/fish | sudo tee -a /etc/shells


TMUX_CONF=~/.tmux.conf
if ! grep -q "set-option -g default-shell /usr/bin/fish" "$TMUX_CONF"; then
    echo 'set-option -g default-shell /usr/bin/fish' >> "$TMUX_CONF"
fi

if ! grep -q "set-option -g default-command /usr/bin/fish" "$TMUX_CONF"; then
    echo 'set-option -g default-command /usr/bin/fish' >> "$TMUX_CONF"
fi

# Reload tmux config if inside tmux
if [ -n "$TMUX" ]; then
    tmux source-file ~/.tmux.conf
fi

# If the container doesn't contain the cuda toolkit run this
# needed for nvcc, ncu, compute-sanitizer

# if ! command -v nvcc &> /dev/null; then
#     echo "CUDA Toolkit not found, installing..."
#     wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run && \
#     sudo sh cuda_12.2.2_535.104.05_linux.run --silent --toolkit
#     # wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run && \
#     #     sudo sh cuda_12.4.1_550.54.15_linux.run
# else
#     echo "CUDA Toolkit is already installed."
# fi


# This isn't quite working yet, needs to work when none of the config files even exist
# git clone https://github.com/suhithr/dotfiles.git && cd dotfiles && \
#     ./minimal_install.sh

git clone https://github.com/suhithr/speedymatmul.git

# repo setup
cd ./speedymatmul && mkdir ./build && cd ./build

echo "tmux new -s vast"

exec fish

# Run this in fish
# fish_add_path /usr/local/cuda/bin
# set --path -Uxp LD_LIBRARY_PATH /usr/local/cuda/lib64/