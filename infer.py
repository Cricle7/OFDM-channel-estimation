import os
import numpy as np
import scipy.io
from sklearn import preprocessing
import time
import matplotlib.pyplot as plt
import itertools
import socket
from scipy.signal import hilbert
import models
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # try to use GPU
print(DEVICE)

class UDP_config:
    def __init__(self):
        self.sample_length=30720

udp_cfg=UDP_config()

def Source_data_gen(frame_num, bk_num, bk_size, data_type, file_name):
    if data_type==0:
        raise NotImplementedError
    if data_type==1:
        PN9=np.ones(512)
        for i in range(9,510):
            PN9[i]=(PN9[i-9]+PN9[i-5])%2
        output=np.zeros((bk_num,bk_size,frame_num))
        for i in range(frame_num):
            for j in range(bk_num):
                for k in range(bk_size):
                    output[j][k][i]=PN9[(i*bk_num*j*bk_size+k)%511]
        return output

def modfun(inputdata,prbnum,modutype):
    temp=1/(2**0.5)
    QPSK_table=temp*np.array([1+1j,1-1j,-1+1j,-1-1j])
    temp=1/(10**0.5)
    QAM16_table=temp*np.array([(1+1j), (1+3j), (3+1j), (3+3j), (1-1j), (1-3j), (3-1j), (3-3j),(-1+1j),(-1+3j),(-3+1j),(-3+3j),(-1-1j),(-1-3j),(-3-1j),(-3-3j)])
    QAM64_table = temp *np.array( [(3 + 3j), (3 + 1j), (1 + 3j), (1 + 1j), (3 + 5j), (3 + 7j), (1 + 5j), (1 + 7j),
                          (5 + 3j), (5 + 1j), (7 + 3j), (7 + 1j), (5 + 5j), (5 + 7j), (7 + 5j), (7 + 7j),
                          (3 - 3j), (3 - 1j), (1 - 3j), (1 - 1j), (3 - 5j), (3 - 7j), (1 - 5j), (1 - 7j),
                          (5 - 3j), (5 - 1j), (7 - 3j), (7 - 1j), (5 - 5j), (5 - 7j), (7 - 5j), (7 - 7j),
                          (-3 + 3j), (-3 + 1j), (-1 + 3j), (-1 + 1j), (-3 + 5j), (-3 + 7j), (-1 + 5j), (-1 + 7j),
                          (-5 + 3j), (-5 + 1j), (-7 + 3j), (-7 + 1j), (-5 + 5j), (-5 + 7j), (-7 + 5j), (-7 + 7j),
                          (-3 - 3j), (-3 - 1j), (-1 - 3j), (-1 - 1j), (-3 - 5j), (-3 - 7j), (-1 - 5j), (-1 - 7j),
                          (-5 - 3j), (-5 - 1j), (-7 - 3j), (-7 - 1j), (-5 - 5j), (-5 - 7j), (-7 - 5j), (-7 - 7j)])
    symbol_len = prbnum;
    out=np.zeros(symbol_len,dtype='complex')
    inputdata=inputdata[0]
    if (modutype == 1): # QPSK
        for kkk in range(symbol_len):
            temp = inputdata[2 * kkk] * 2 + inputdata[2 * kkk+1];
            out[kkk] = QPSK_table[int(temp[0])];
    elif(modutype == 2): # 16QAM
        for kkk in range(symbol_len):
            temp = inputdata[4 * kkk] * 8 + inputdata[4 * kkk+1] * 4 + inputdata[4 * kkk+2] * 2 + inputdata[4 * kkk+3];
            out[kkk] = QAM16_table[int(temp[0])];
    else: # 64QAM
        for kkk in range(symbol_len):
            temp = inputdata[6 * kkk] * 32 + inputdata[6 * kkk +1] * 16 + inputdata[6 * kkk +2] * 8+ inputdata[6 * kkk +3] * 4 + inputdata[6 * kkk +4] * 2 + inputdata[6 * kkk+5];
            out[kkk] = QAM64_table[int(temp[0])];
    return out

def RFLoopback(txdataIQ,pcip,xsrpip):
    # # 对数据长度进行补零或溢出删除
    SAMPLE_LENGTH = 30720; # 样点数30.72Msps * 1ms
    TxdataI = np.zeros(SAMPLE_LENGTH);
    TxdataQ = np.zeros(SAMPLE_LENGTH);
    if txdataIQ.shape[0] >= SAMPLE_LENGTH:
        TxdataI = np.real(txdataIQ[0:SAMPLE_LENGTH]);
        TxdataQ = np.imag(txdataIQ[0:SAMPLE_LENGTH]);
    else:
        TxdataI[1:txdataIQ.shape[0]]=np.real(txdataIQ);
        TxdataQ[1:txdataIQ.shape[0]]=np.imag(txdataIQ);

   #################################################################################################################
    ## 发送数据处理
    #放大峰值，浮点取整
    len = SAMPLE_LENGTH * 2;
    dataIQ = np.zeros(len);
    for i in range(SAMPLE_LENGTH):
        dataIQ[i*2]=TxdataI[i]
        dataIQ[i*2+1]=TxdataQ[i]
    dataIQ = dataIQ* (2047 / max(dataIQ)); # 放大峰值至2000, 接近理论峰值2047
    dataIQ = np.fix(dataIQ); # 浮点数强制取整

    # 防止溢出，并对负数进行补码操作
    for i in range(len):
        if dataIQ[i]>2047:
            dataIQ[i]=2047
        elif dataIQ[i]<0:
            dataIQ[i]=4096+dataIQ[i]
    for i in range(SAMPLE_LENGTH):
        dataIQ[i*2]=dataIQ[i*2]*16
        dataIQ[i*2+1]=np.fix(dataIQ[i*2+1]/256)+np.mod(dataIQ[i*2+1],256)*256
    dataIQ=dataIQ.astype("uint16")
    ##########################################################定义配置参数常量
    test_set_sync_clock = bytes.fromhex("000099bb69000000000000000000000000000000") ; # OK
    test_Set_router = bytes.fromhex("000099bb68000000000600000000000000000000")# 网口环回 % OK
    test_set_delay_system = bytes.fromhex("000099bb67000000000000000000000000000000")
    test_tx_command = bytes.fromhex("000099bb65020003000178000000000000000000")# 0A10个时隙，03FF时隙开关000001111111111，0000分频，7800数据个数 / 时隙30720 % OK
    test_Send_IQ = bytes.fromhex("000099bb64000000000000000000000000000000")
    test_Get_IQ = bytes.fromhex("000099bb66010001008000F00000000000000000")#01采集时隙号，0000分频，0060包的数量96，00F0包的大小240（实际接收字节数为配置值乘以4） % OK
    udp_addr = (pcip, 12345)
    dest_addr = (xsrpip, 13345)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)
    udp_socket.sendto(test_set_sync_clock, dest_addr)
    udp_socket.sendto(test_Set_router, dest_addr)
    udp_socket.sendto(test_set_delay_system, dest_addr)
    udp_socket.sendto(test_tx_command, dest_addr)
    udp_socket.sendto(test_Send_IQ, dest_addr)
    SEND_PACKET_LENGTH = 512;
    for pn in range(int(np.fix(SAMPLE_LENGTH*2/SEND_PACKET_LENGTH))):
        send_data_packet=int(dataIQ[pn*SEND_PACKET_LENGTH]).to_bytes( length=2,byteorder='big',signed=False)
        for i in range(SEND_PACKET_LENGTH-1):
            send_data_packet=send_data_packet+int(dataIQ[pn*SEND_PACKET_LENGTH+i+1]).to_bytes( length=2,byteorder='big',signed=False)
        #send_data_packet=dataIQ[pn*SEND_PACKET_LENGTH:(pn+1)*SEND_PACKET_LENGTH]
        udp_socket.sendto(send_data_packet,dest_addr)
    udp_socket.close()
def modulation_x(mod_type,fs,len,fm,fc,recurrent_num):
    # mod_type: 选择六种模拟调制方式中的一种
    # fs: 采样频率
    # len: 采样点数
    # fm: 基带信号频率
    # fc: 载波频率
    dt = 1 / fs;
    t0=np.arange(0,len*dt,dt);
    A = 1 ;# 调制信号m(t)的幅度
    A0 = 2; # 直流偏量A0
    mt = A * np.sin(2 * np.pi * fm * t0); # 调制信号（基带信号）
    A1 = mt + A0; # 加直流分量后的调制信号
    carry = np.cos(2 * np.pi * fc * t0); # 载波
    snr = 30;
    if mod_type==1:#AM
        for rn in range(recurrent_num):
            y1 = np.cos(2 * np.pi * fc * t0); # 载波1
            y2 = -np.sin(2 * np.pi * fc * t0); # 载波2
            S_AM1 = A1*y1;
            S_AM2 = A1*y2;
            S_AM_IQ=S_AM1+1j*S_AM2
            modSignal=S_AM_IQ
            # xsrp begin
            FRAME_LEN = 30720; # 帧样点数据长度，固定30720
            pcip = '192.168.1.180';
            xsrpip = '192.168.1.166';
            tx_data_fill = np.zeros(FRAME_LEN - len);
            tx_dataIQ = np.concatenate((modSignal, tx_data_fill),axis=0);
            RFLoopback(tx_dataIQ,pcip,xsrpip);
            #xsrp end
    elif mod_type==2:#FM
        for rn in range(recurrent_num):
            Kf = fm;
            mti = np.sin(2 * np.pi * fm * t0) / (2 * np.pi * fm); # mt = cos(2 * pi * f * t0)的积分函数
            S_FM = np.sqrt(2) * np.cos(2 * np.pi * fc * t0 + 2 * np.pi * Kf * mti);
            S_FM_IQ = hilbert(S_FM);
            modSignal = S_FM_IQ;
            # xsrp begin
            FRAME_LEN = 30720;  # 帧样点数据长度，固定30720
            pcip = '192.168.1.180';
            xsrpip = '192.168.1.166';
            tx_data_fill = np.zeros(FRAME_LEN - len);
            tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
            RFLoopback(tx_dataIQ, pcip, xsrpip);
            #xsrp end
    elif mod_type==3:#QPSK
        info_data = Source_data_gen(1, 1, len * 2, 1, 0);
        IQ_s=modfun(info_data,len,1);
        y1 = np.cos(2 * np.pi * fc * t0); # 载波1
        y2 = -np.sin(2 * np.pi * fc * t0); # 载波2
        qpskI = np.real(IQ_s)* y1; # I路调制信号
        qpskQ = np.imag(IQ_s)* y2; # Q路调制信号
        modSignal = qpskI+1j*qpskQ # 调制信号
        # xsrp begin
        FRAME_LEN = 30720;  # 帧样点数据长度，固定30720
        pcip = '192.168.1.180';
        xsrpip = '192.168.1.166';
        tx_data_fill = np.zeros(FRAME_LEN - len);
        tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
        RFLoopback(tx_dataIQ, pcip, xsrpip);
        # xsrp end
    elif mod_type==4:#QAM16
        info_data = Source_data_gen(1, 1, len * 4, 1, 0);
        IQ_s=modfun(info_data,len,2);
        y1 = np.cos(2 * np.pi * fc * t0); # 载波1
        y2 = -np.sin(2 * np.pi * fc * t0); # 载波2
        qam16I = np.real(IQ_s)* y1; # I路调制信号
        qam16Q = np.imag(IQ_s)* y2; # Q路调制信号
        modSignal = qam16I+1j*qam16Q # 调制信号
        # xsrp begin
        FRAME_LEN = 30720;  # 帧样点数据长度，固定30720
        pcip = '192.168.1.180';
        xsrpip = '192.168.1.166';
        tx_data_fill = np.zeros(FRAME_LEN - len);
        tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
        RFLoopback(tx_dataIQ, pcip, xsrpip);
        # xsrp end
    elif mod_type==5:#QAM64
        info_data = Source_data_gen(1, 1, len * 8, 1, 0);
        IQ_s=modfun(info_data,len,3);
        y1 = np.cos(2 * np.pi * fc * t0); # 载波1
        y2 = -np.sin(2 * np.pi * fc * t0); # 载波2
        qam64I = np.real(IQ_s)* y1; # I路调制信号
        qam64Q = np.imag(IQ_s)* y2; # Q路调制信号
        modSignal = qam64I+1j*qam64Q # 调制信号
        # xsrp begin
        FRAME_LEN = 30720;  # 帧样点数据长度，固定30720
        pcip = '192.168.1.180';
        xsrpip = '192.168.1.166';
        tx_data_fill = np.zeros(FRAME_LEN - len);
        tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
        RFLoopback(tx_dataIQ, pcip, xsrpip);
        # xsrp end
    elif mod_type==6:
        info_data=Source_data_gen(1,1,len,1,0)
        IQ_s=info_data*2-1
        IQ_s=np.reshape(IQ_s,len)
        y1 = np.cos(2 * np.pi * fc * t0);  # 载波1
        y2 = -np.sin(2 * np.pi * fc * t0);  # 载波2
        bpskI = IQ_s * y1;  # I路调制信号
        bpskQ = IQ_s * y2;  # Q路调制信号
        modSignal = bpskI + 1j * bpskQ  # 调制信号
        # xsrp begin
        FRAME_LEN = 30720;  # 帧样点数据长度，固定30720
        pcip = '192.168.1.180';
        xsrpip = '192.168.1.166';
        tx_data_fill = np.zeros(FRAME_LEN - len);
        tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
        RFLoopback(tx_dataIQ, pcip, xsrpip);
        # xsrp end

def send_to_x():
    # AM: 1,
    #FM: 2,
    #QPSK: 3,
    #QAM16: 4,
    #QAM64: 5,
    #BPSK: 6
    modetype = 6;
    fs = 30720;
    len = 30000;
    fm = 200;
    fc = 3000;
    recurrent_num = 10;
    modulation_x(modetype,fs,len,fm,fc,recurrent_num)

def main():
    print("program start")
    send_to_x()
    print("sending data finished")
    model_path = 'checkpoints/5_best_model1_x.pth'
    model_main = models.DnnNet0()
    model_main = model_main.to(DEVICE)
    model_main.load(model_path)
    model=model_main
    print("loading model finished")
    udp_addr = ('192.168.1.180', 12345)
    dest_addr = ('192.168.1.166', 13345)
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
    # receiving end
    print("receiving finished,begin to process data")
    recv_data_all_int = np.zeros(len(recv_data_all))
    for i in range(len(recv_data_all)):
        recv_data_all_int[i] = recv_data_all[i]
    recv_data_all_float = np.array(recv_data_all_int, dtype=np.float32)
    udp_data_ri = np.zeros(int(len(recv_data_all) / 4))
    udp_data_rq = np.zeros(int(len(recv_data_all) / 4))
    for m in range(int(len(recv_data_all) / 4)):
        udp_data_ri[m] = recv_data_all_float[m * 4] * 256 + recv_data_all_float[m * 4 + 1]
        udp_data_rq[m] = recv_data_all_float[m * 4 + 2] * 256 + recv_data_all_float[m * 4 + 3]
        # 负数的处理
        if udp_data_ri[m] >= 2049:
            udp_data_ri[m] = udp_data_ri[m] - 4096
        if udp_data_rq[m] >= 2049:
            udp_data_rq[m] = udp_data_rq[m] - 4096
    udp_data_ri = udp_data_ri / 2047;
    udp_data_rq = udp_data_rq / 2047;
    print("abs(i,q):( {},{} )".format(np.mean(np.abs(udp_data_ri)),np.mean(np.abs(udp_data_rq))))
    plt.figure(1)
    plt.title("real")
    ri_x=np.arange(0,udp_data_ri.shape[0],1)
    plt.plot(ri_x,udp_data_ri)
    plt.show()
    plt.figure(2)
    plt.title("imag")
    rq_x=np.arange(0,udp_data_rq.shape[0],1)
    plt.plot(rq_x,udp_data_rq)
    plt.show()
    #udp_data_ri=np.reshape(udp_data_ri,(1,udp_cfg.sample_length))
    #udp_data_rq=np.reshape(udp_data_rq,(1,udp_cfg.sample_length))
    sample_mod=udp_data_ri+1j*udp_data_rq
    #sample_mod = scipy.io.loadmat('./data/BPSK.mat')['moddata'][0]
    each_sample_num = 20
    input_data=[]
#    print("preprocessing input")
    #for i in range(each_sample_num):
        #current_frame = sample_mod[i * 128:(i + 1) * 128]
        #current_frame = np.concatenate((np.real(current_frame), np.imag(current_frame)), axis=0)
        #input_data.append(current_frame)
    #input_data = np.vstack(input_data)
    #input_data = input_data.reshape((-1, 256))
    ##scaler = sklearn.preprocessing.MinMaxScaler()
    ##input_data = scaler.fit_transform(input_data)
    #for i in range(input_data.shape[0]):
        #input_data[i]=input_data[i]/np.max(input_data[i])
    #input_data = input_data.reshape((-1, 1, 2, 128))
    #mod_type=['qam16','qam64','qpsk','fm','am','bpsk']
    #pred_count=np.zeros(6)
    #print("infering model")
    #for i in range(each_sample_num):
        #current_input=torch.from_numpy(input_data[i].astype("float32")).to(DEVICE)
        #with torch.no_grad():
            #output = model(current_input)[0]
        #predict = np.argmax(output.to("cpu").numpy())
        #pred_count[predict]=pred_count[predict]+1
    #for i in range(6):
        #print("{}: {}".format(mod_type[i],pred_count[i]/each_sample_num))

if __name__=="__main__":
    main()