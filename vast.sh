#!/bin/bash

sudo add-apt-repository ppa:fish-shell/release-4 -y

apt-get update && \
    apt-get install -y --no-install-recommends

apt-get install -y --no-install-recommends \
    silversearcher-ag fish vim gdb cmake
curl -LsSf https://astral.sh/uv/install.sh | sh

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


# This isn't quite working yet, needs to work when none of the config files even exist
# git clone https://github.com/suhithr/dotfiles.git && cd dotfiles && \
#     ./minimal_install.sh

git clone https://github.com/suhithr/speedymatmul.git

# repo setup
cd ./speedymatmul && mkdir ./build && cd ./build

exec fish

