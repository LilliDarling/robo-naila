#!/bin/bash
# Run as root on a NetworkManager Pi with both DHCP connections already active.
set -euo pipefail
PATH=/usr/sbin:/usr/bin:/sbin:/bin
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
backup=/root/naila-dual-uplink-backup
test "$(id -u)" = 0
for cmd in nmcli ip sysctl python3; do command -v "$cmd" >/dev/null; done
test ! -e "$backup"
test ! -e /etc/sysctl.d/90-naila-dual-uplink.conf
test ! -e /etc/NetworkManager/dispatcher.d/90-naila-dual-uplink
if ip -4 rule show | grep -Eq '^(10100|10200):'; then exit 1; fi
for table in 101 102; do
    test -z "$(ip -4 route show table "$table" 2>/dev/null || true)"
done
for nic in eth0 wlan0; do
    test "$(sysctl -n "net.ipv4.conf.$nic.arp_ignore")" = 0
    test "$(sysctl -n "net.ipv4.conf.$nic.arp_announce")" = 0
    test -n "$(nmcli -g IP4.ADDRESS device show "$nic")"
    test -n "$(nmcli -g IP4.GATEWAY device show "$nic")"
done
install -d -m 0700 "$backup"
ip -4 rule show > "$backup/rules.before"
ip -4 route show table all > "$backup/routes.before"
install -m 0700 "$source_dir/rollback.sh" "$backup/rollback.sh"
trap '"$backup/rollback.sh"' ERR
install -m 0644 "$source_dir/etc/sysctl.d/90-naila-dual-uplink.conf" /etc/sysctl.d/
install -m 0755 "$source_dir/etc/NetworkManager/dispatcher.d/90-naila-dual-uplink" /etc/NetworkManager/dispatcher.d/
for nic in eth0 wlan0; do
    addresses=$(nmcli -g IP4.ADDRESS device show "$nic")
    env IP4_NUM_ADDRESSES="$(printf '%s\n' "$addresses" | wc -l | tr -d ' ')" \
        IP4_ADDRESS_0="$addresses" IP4_GATEWAY="$(nmcli -g IP4.GATEWAY device show "$nic")" \
        /etc/NetworkManager/dispatcher.d/90-naila-dual-uplink "$nic" up
done
sysctl -p /etc/sysctl.d/90-naila-dual-uplink.conf
printf 'Installed dual-uplink configuration; rollback: sudo %s/rollback.sh\n' "$backup"
