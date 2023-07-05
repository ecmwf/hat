import os
import platform
import socket


def get_host():
    """Local host on Mac and network host on HPC
    (note the network address on HPC is not constant)"""

    # get the hostname
    hostname = socket.gethostname()

    # get the IP address(es) associated with the hostname
    ip_addresses = socket.getaddrinfo(
        hostname, None, socket.AF_INET, socket.SOCK_STREAM
    )

    # return first valid address
    for ip_address in ip_addresses:
        network_host = ip_address[4][0]
        return network_host


def mac_or_hpc():
    """Is this running on a Mac or the HPC or other?"""

    if platform.system() == "Darwin":
        return "mac"
    elif platform.system() == "Linux" and os.environ.get("ECPLATFORM"):
        return "hpc"
    else:
        return "other"


def host_and_port(host="127.0.0.1", port=8000):
    """return network host and port for tiler app to use"""

    computer = mac_or_hpc()

    if computer == "hpc":
        host = get_host()
        port = 8700

    return (host, port)
