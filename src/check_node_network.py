#!/usr/bin/env python3
import os
import socket
import subprocess
import psutil


def probe_interface(iface_ip, master_ip, master_port, timeout=1.0):
    """
    Attempt to connect to MASTER_ADDR:MASTER_PORT from the given local IP address
    to see if routing is valid. Returns True if successful, False otherwise.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        # Bind to the local interface IP before connecting
        sock.bind((iface_ip, 0))
        sock.connect((master_ip, master_port))
        sock.close()
        return True
    except Exception:
        return False


def get_all_ip_addresses(exclude_ifaces=("lo", "docker0")):
    """
    Return a list of (iface_name, ip_address) for non-excluded interfaces.
    """
    results = []
    for iface, addrs in psutil.net_if_addrs().items():
        if iface in exclude_ifaces:
            continue
        for addr in addrs:
            if addr.family == socket.AF_INET and addr.address != '127.0.0.1':
                results.append((iface, addr.address))
    return results


def find_working_interface(master_ip, master_port):
    """
    Find the first local interface that can route to MASTER_ADDR:MASTER_PORT.
    Return the interface name or None.
    """
    all_ips = get_all_ip_addresses()
    for iface_name, ip_addr in all_ips:
        if probe_interface(ip_addr, master_ip, master_port):
            return iface_name
    return None


def get_network_info():
    """Get detailed network interface information."""
    info = []
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                info.append({
                    'iface': iface,
                    'ip': addr.address,
                    'netmask': addr.netmask,
                    'broadcast': getattr(addr, 'broadcast', None)
                })
    return info


def main():
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    
    # Network diagnostics
    print("\n=== Network Interface Report ===")
    for info in get_network_info():
        print(f"\nInterface: {info['iface']}")
        print(f"IP: {info['ip']}")
        print(f"Netmask: {info['netmask']}")
        print(f"Broadcast: {info['broadcast']}")
        
        # Test connectivity
        if info['ip']:
            result = probe_interface(info['ip'], master_addr, master_port)
            print(f"Master connectivity test: {'SUCCESS' if result else 'FAILED'}")

    # Detect best interface
    best_iface = find_working_interface(master_addr, master_port)
    
    print("\n=== NCCL Configuration Recommendations ===")
    if best_iface:
        print(f"Recommended NCCL_SOCKET_IFNAME: {best_iface}")
        print("Export command: export NCCL_SOCKET_IFNAME=" + best_iface)
    else:
        print("Warning: No working interface detected for NCCL communication")
        print("Consider manually setting: export NCCL_SOCKET_IFNAME=bond0")
    
    print("\nAdditional recommended NCCL settings:")
    print("export NCCL_DEBUG=INFO")
    print("export NCCL_IB_DISABLE=0")
    print("export NCCL_NET_GDR_LEVEL=2")
    
    # If InfiniBand is detected
    if any(iface.startswith(('ib', 'mlx')) for iface, _ in get_all_ip_addresses()):
        print("\nInfiniBand detected - Additional settings:")
        print("export NCCL_IB_GID_INDEX=3")
        print("export NCCL_IB_TC=106")
        print("export NCCL_IB_SL=3")
        print("export NCCL_IB_TIMEOUT=22")


if __name__ == "__main__":
    main()
