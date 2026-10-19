#!/usr/bin/perl
# Loopback TCP relay for trace export: LISTEN -> TARGET.
# Runs as a root LaunchDaemon because macOS Local Network privacy blocks LAN connections from
# daemons with a UserName (errno 65, "No route to host"). The AI server stays unprivileged and
# exports to the loopback listener. Usage: otlp-relay.pl 127.0.0.1:4318 192.168.1.239:4318
use strict;
use warnings;
use IO::Socket::INET;
use IO::Select;
use Socket qw(SOL_SOCKET SO_ERROR);

my ($listen, $target) = @ARGV;
die "usage: $0 LISTEN_HOST:PORT TARGET_HOST:PORT\n" unless $listen && $target;
die "refusing non-loopback listen address $listen\n" unless $listen =~ /^127\.0\.0\.1:\d+$/;

my $server = IO::Socket::INET->new(LocalAddr => $listen, Listen => 16, ReuseAddr => 1)
    or die "listen $listen: $!\n";
$SIG{CHLD} = 'IGNORE';
$| = 1;
print "otlp_relay_started listen=$listen target=$target\n";

while (1) {
    my $client = $server->accept or next;
    my $pid = fork;
    if (!defined $pid) { close $client; next; }
    if ($pid) { close $client; next; }

    close $server;
    # With Timeout, IO::Socket::INET 1.x can return an unconnected socket on a refused connect,
    # so check the connection itself and report SO_ERROR.
    my $upstream = IO::Socket::INET->new(PeerAddr => $target, Timeout => 3);
    unless ($upstream && $upstream->connected) {
        my $errno = $upstream ? unpack('i', getsockopt($upstream, SOL_SOCKET, SO_ERROR)) : $! + 0;
        $! = $errno;
        print "otlp_relay_connect_failed target=$target errno=$errno error=$!\n";
        exit 1;
    }
    my %peer = ($client => $upstream, $upstream => $client);
    my $select = IO::Select->new($client, $upstream);
    # Idle connections close after 60s; the OTLP exporter reconnects per batch as needed.
    while (my @ready = $select->can_read(60)) {
        for my $from (@ready) {
            my $read = sysread($from, my $buffer, 65536);
            exit 0 unless $read;
            my $offset = 0;
            while ($offset < $read) {
                my $written = syswrite($peer{$from}, $buffer, $read - $offset, $offset);
                exit 1 unless defined $written;
                $offset += $written;
            }
        }
    }
    exit 0;
}
