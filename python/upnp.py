#!/usr/bin/env python3
import socket

import upnpclient


def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]


if __name__ == '__main__':
    devices = upnpclient.discover()
    print(devices)

    d = devices[0]
    print(d.WANIPConn1.GetStatusInfo())

    d.WANIPConn1.AddPortMapping(
        NewRemoteHost='',
        NewExternalPort=60000,
        NewProtocol='TCP',
        NewInternalPort=60000,
        NewInternalClient=get_local_ip(),
        NewEnabled='1',
        NewPortMappingDescription='file_server',
        NewLeaseDuration=0)
