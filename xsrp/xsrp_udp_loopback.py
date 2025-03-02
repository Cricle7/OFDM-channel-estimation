import socket
import numpy as np

class UDPConfig:
    def __init__(self):
        self.sample_length = 30720

udp_cfg = UDPConfig()

def udp_tx(txdataIQ, pcip, xsrpip, port):
    SAMPLE_LENGTH = 30720  # Sample length
    TxdataI = np.zeros(SAMPLE_LENGTH)
    TxdataQ = np.zeros(SAMPLE_LENGTH)
    if txdataIQ.shape[0] >= SAMPLE_LENGTH:
        TxdataI = np.real(txdataIQ[0:SAMPLE_LENGTH])
        TxdataQ = np.imag(txdataIQ[0:SAMPLE_LENGTH])
    else:
        TxdataI[1:txdataIQ.shape[0]] = np.real(txdataIQ)
        TxdataQ[1:txdataIQ.shape[0]] = np.imag(txdataIQ)

    # Sending data processing
    len = SAMPLE_LENGTH * 2
    dataIQ = np.zeros(len)
    for i in range(SAMPLE_LENGTH):
        dataIQ[i * 2] = TxdataI[i]
        dataIQ[i * 2 + 1] = TxdataQ[i]
    dataIQ = dataIQ * (2047 / max(dataIQ))  # Amplify peak value
    dataIQ = np.fix(dataIQ)

    # Avoid overflow
    for i in range(len):
        if dataIQ[i] > 2047:
            dataIQ[i] = 2047
        elif dataIQ[i] < 0:
            dataIQ[i] = 4096 + dataIQ[i]
    for i in range(SAMPLE_LENGTH):
        dataIQ[i * 2] = dataIQ[i * 2] * 16
        dataIQ[i * 2 + 1] = np.fix(dataIQ[i * 2 + 1] / 256) + np.mod(dataIQ[i * 2 + 1], 256) * 256
    dataIQ = dataIQ.astype("uint16")

    # Define configuration constants
    test_set_sync_clock = bytes.fromhex("000099bb69000000000000000000000000000000")
    test_Set_router = bytes.fromhex("000099bb68000000000600000000000000000000")
    test_set_delay_system = bytes.fromhex("000099bb67000000000000000000000000000000")
    test_tx_command = bytes.fromhex("000099bb65020003000178000000000000000000")
    test_Send_IQ = bytes.fromhex("000099bb64000000000000000000000000000000")

    udp_addr = (pcip, port)
    dest_addr = (xsrpip, port)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    udp_socket.sendto(test_set_sync_clock, dest_addr)
    udp_socket.sendto(test_Set_router, dest_addr)
    udp_socket.sendto(test_set_delay_system, dest_addr)
    udp_socket.sendto(test_tx_command, dest_addr)
    udp_socket.sendto(test_Send_IQ, dest_addr)

    SEND_PACKET_LENGTH = 512
    for pn in range(int(np.fix(SAMPLE_LENGTH * 2 / SEND_PACKET_LENGTH))):
        send_data_packet = int(dataIQ[pn * SEND_PACKET_LENGTH]).to_bytes(length=2, byteorder='big', signed=False)
        for i in range(SEND_PACKET_LENGTH - 1):
            send_data_packet += int(dataIQ[pn * SEND_PACKET_LENGTH + i + 1]).to_bytes(length=2, byteorder='big', signed=False)
        udp_socket.sendto(send_data_packet, dest_addr)

    udp_socket.close()

def udp_recv(pcip, xsrpip, port):
    udp_addr = (pcip, port)
    dest_addr = (xsrpip, port)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    udp_socket.settimeout(5)
    while True:
        # sending commend begin
        hex_data = "000099bb66010001008000F00000000000000000"
        send_data = bytes.fromhex(hex_data)
        udp_socket.sendto(send_data, dest_addr)
        # sending end

        # receiving data begin
        recv_data_all = None
        for package_id in range(128):
            try:
                recv_data, source = udp_socket.recvfrom(960)  # 960表示本次接收的最大字节数
                if recv_data_all:
                    recv_data_all = recv_data_all + recv_data
                else:
                    recv_data_all = recv_data
            except socket.timeout:
                print("Timeout occurred, no data received.")
                break
        if len(recv_data_all) == udp_cfg.sample_length * 4:
            break