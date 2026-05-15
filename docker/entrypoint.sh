#!/usr/bin/env bash
# Container entrypoint: bring up sshd + frpc, then exec the user command.
#
# Controlled by env vars (all optional):
#   FRP_DISABLED            = 1 to skip frpc / sshd entirely
#   FRP_SERVER_ADDR         default point.dd.works
#   FRP_SERVER_PORT         default 7000
#   FRP_PROXY_PREFIX        default "$HOSTNAME" — must be unique across
#                           containers connected to the same frps
#   FRP_SSH_REMOTE_PORT     default 2334
#   FRP_COMFYUI_REMOTE_PORT default 18188
#   SSH_AUTHORIZED_KEYS     newline-separated pubkeys appended to
#                           /root/.ssh/authorized_keys
#   SSH_ROOT_PASSWORD       if set, enables password login for root
#                           (key auth is preferred — only use this for
#                           debugging when you can't preload a key)

set -e

if [ "${FRP_DISABLED:-0}" != "1" ]; then
    # sshd host keys (generated once at container startup).
    [ -f /etc/ssh/ssh_host_rsa_key ] || ssh-keygen -A

    mkdir -p /root/.ssh
    chmod 700 /root/.ssh

    if [ -n "${SSH_AUTHORIZED_KEYS:-}" ]; then
        printf '%s\n' "${SSH_AUTHORIZED_KEYS}" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
    fi

    if [ -n "${SSH_ROOT_PASSWORD:-}" ]; then
        echo "root:${SSH_ROOT_PASSWORD}" | chpasswd
        sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
        sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
    fi

    /usr/sbin/sshd

    : "${FRP_SERVER_ADDR:=point.dd.works}"
    : "${FRP_SERVER_PORT:=7000}"
    : "${FRP_PROXY_PREFIX:=$(hostname)}"
    : "${FRP_SSH_REMOTE_PORT:=2334}"
    : "${FRP_COMFYUI_REMOTE_PORT:=18188}"
    export FRP_SERVER_ADDR FRP_SERVER_PORT FRP_PROXY_PREFIX \
           FRP_SSH_REMOTE_PORT FRP_COMFYUI_REMOTE_PORT

    mkdir -p /etc/frp
    envsubst < /etc/frp/frpc.toml.tmpl > /etc/frp/frpc.toml
    echo "[entrypoint] frpc config:" >&2
    sed 's/^/  /' /etc/frp/frpc.toml >&2

    frpc -c /etc/frp/frpc.toml &
fi

exec "$@"
