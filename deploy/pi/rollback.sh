#!/bin/bash
# Root rollback; --hub-only leaves existing network configuration alone.
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
test "$(id -u)" = 0
test "$#" = 0 || { test "$#" = 1 && test "$1" = --hub-only; }
if [ -e /etc/systemd/system/naila-hub.service ]; then
    systemctl disable --now naila-hub.service
    rm -f /etc/systemd/system/naila-hub.service
    systemctl daemon-reload
fi
# Preserve /etc/naila/hub.env for subsequent installs.
if [ "${1:-}" = --hub-only ]; then exit 0; fi
rm -f /etc/NetworkManager/dispatcher.d/90-naila-dual-uplink
rm -f /etc/sysctl.d/90-naila-dual-uplink.conf
ip -4 rule del priority 10100 table 101 2>/dev/null || true
ip -4 rule del priority 10200 table 102 2>/dev/null || true
ip -4 route flush table 101 2>/dev/null || true
ip -4 route flush table 102 2>/dev/null || true
sysctl -q -w net.ipv4.conf.eth0.arp_ignore=0 net.ipv4.conf.eth0.arp_announce=0
sysctl -q -w net.ipv4.conf.wlan0.arp_ignore=0 net.ipv4.conf.wlan0.arp_announce=0
# Main routes/NM profiles, rp_filter and Wi-Fi power saving were not changed.
ip -4 rule show
ip -4 route show table main
