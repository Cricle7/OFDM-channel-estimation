import numpy as np
from xsrp_udp_loopback import *
from scipy.signal import hilbert
# 随机二进制序列（PRBS，Pseudo-Random Binary Sequence）
def Source_data_gen(frame_num, bk_num, bk_size, data_type, file_name=None):
    if data_type == 0:
        raise NotImplementedError("Data type 0 is not implemented.")
    if data_type == 1:
        # 生成 PN9 伪随机序列
        PN9 = np.ones(511)  # 9阶移位寄存器，长度 511
        for i in range(9, 511):
            PN9[i] = (PN9[i - 9] + PN9[i - 5]) % 2  # 线性反馈移位寄存器公式
        # 生成输出数组
        output = np.zeros((bk_num, bk_size, frame_num))

        # 填充数据
        for i in range(frame_num):
            for j in range(bk_num):
                for k in range(bk_size):
                    output[j, k, i] = PN9[(i * bk_num * bk_size + k) % 511]  # 优化索引计算

        return output
def modfun(inputdata, prbnum, modutype):
    temp = 1 / (2 ** 0.5)
    QPSK_table = temp * np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    temp = 1 / (10 ** 0.5)
    QAM16_table = temp * np.array([(1 + 1j), (1 + 3j), (3 + 1j), (3 + 3j), (1 - 1j), (1 - 3j), (3 - 1j), (3 - 3j),
                                   (-1 + 1j), (-1 + 3j), (-3 + 1j), (-3 + 3j), (-1 - 1j), (-1 - 3j), (-3 - 1j), (-3 - 3j)])
    QAM64_table = temp * np.array([(3 + 3j), (3 + 1j), (1 + 3j), (1 + 1j), (3 + 5j), (3 + 7j), (1 + 5j), (1 + 7j),
                                   (5 + 3j), (5 + 1j), (7 + 3j), (7 + 1j), (5 + 5j), (5 + 7j), (7 + 5j), (7 + 7j),
                                   (3 - 3j), (3 - 1j), (1 - 3j), (1 - 1j), (3 - 5j), (3 - 7j), (1 - 5j), (1 - 7j),
                                   (5 - 3j), (5 - 1j), (7 - 3j), (7 - 1j), (5 - 5j), (5 - 7j), (7 - 5j), (7 - 7j),
                                   (-3 + 3j), (-3 + 1j), (-1 + 3j), (-1 + 1j), (-3 + 5j), (-3 + 7j), (-1 + 5j), (-1 + 7j),
                                   (-5 + 3j), (-5 + 1j), (-7 + 3j), (-7 + 1j), (-5 + 5j), (-5 + 7j), (-7 + 5j), (-7 + 7j),
                                   (-3 - 3j), (-3 - 1j), (-1 - 3j), (-1 - 1j), (-3 - 5j), (-3 - 7j), (-1 - 5j), (-1 - 7j),
                                   (-5 - 3j), (-5 - 1j), (-7 - 3j), (-7 - 1j), (-5 - 5j), (-5 - 7j), (-7 - 5j), (-7 - 7j)])

    symbol_len = prbnum
    out = np.zeros(symbol_len, dtype='complex')
    inputdata = inputdata[0]
    
    if modutype == 1:  # QPSK
        for kkk in range(symbol_len):
            temp = inputdata[2 * kkk] * 2 + inputdata[2 * kkk + 1]
            out[kkk] = QPSK_table[int(temp[0])]
    elif modutype == 2:  # 16QAM
        for kkk in range(symbol_len):
            temp = inputdata[4 * kkk] * 8 + inputdata[4 * kkk + 1] * 4 + inputdata[4 * kkk + 2] * 2 + inputdata[4 * kkk + 3]
            out[kkk] = QAM16_table[int(temp[0])]
    else:  # 64QAM
        for kkk in range(symbol_len):
            temp = inputdata[6 * kkk] * 32 + inputdata[6 * kkk + 1] * 16 + inputdata[6 * kkk + 2] * 8 + inputdata[6 * kkk + 3] * 4 + inputdata[6 * kkk + 4] * 2 + inputdata[6 * kkk + 5]
            out[kkk] = QAM64_table[int(temp[0])]
    return out

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

class Modulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.modetype = cfg['modetype']
        self.fs = cfg['fs']
        self.length = cfg['length']
        self.fm = cfg['fm']
        self.fc = cfg['fc']
        self.recurrent_num = cfg['recurrent_num']
        self.pcip = cfg['pcip']
        self.xsrpip = cfg['xsrpip']
        self.port = cfg['port']
        self.frame_len = cfg['frame_len']
        self.dt = 1 / self.fs
        self.t0 = np.arange(0, self.length * self.dt, self.dt)

    def _do_qpsk_rf(self):
        info_data = Source_data_gen(1, 1, len * 2, 1, 0);
        IQ_s=modfun(info_data,len,1);
        y1 = np.cos(2 * np.pi * self.fc * self.t0); # 载波1
        y2 = -np.sin(2 * np.pi * self.fc * self.t0); # 载波2
        qpskI = np.real(IQ_s)* y1; # I路调制信号
        qpskQ = np.imag(IQ_s)* y2; # Q路调制信号
        modSignal = qpskI+1j*qpskQ # 调制信号
        tx_data_fill = np.zeros(self.frame_len - len);
        tx_dataIQ = np.concatenate((modSignal, tx_data_fill), axis=0);
        udp_tx(tx_dataIQ, self.pcip, self.xsrpip, self.port);
    def _do_bpsk_rf(self):
        bpsk_data = np.random.randint(0, 2, self.length)
        bpsk_data = bpsk_data * 2 - 1

        carrier_i = np.cos(2 * np.pi * self.fc * self.t0)
        carrier_q = -np.sin(2 * np.pi * self.fc * self.t0)
        bpsk_i = bpsk_data * carrier_i
        bpsk_q = bpsk_data * carrier_q
        modSignal = bpsk_i + 1j * bpsk_q
        udp_tx(modSignal, self.pcip, self.xsrpip, self.port)
    def _do_ofdm(self, length):

        # Step 1: 调用 _do_qpsk 生成基带 QPSK 符号
        qpsk_symbols = self._do_qpsk(length)

        # Step 2: 进行 IFFT 处理（得到时域 OFDM 信号）
        N = len(qpsk_symbols)  # QPSK 符号的数量 = 子载波数目
        time_signal = np.fft.ifft(qpsk_symbols) * np.sqrt(N)  # IFFT（可以加上缩放因子）

        # Step 3: 添加循环前缀（CP）
        cp_len = 16  # 假设循环前缀长度为16（可以调整）
        cyclic_prefix = time_signal[-cp_len:]  # 从时域信号的尾部截取 CP
        ofdm_signal = np.concatenate([cyclic_prefix, time_signal])  # 将 CP 加入到时域信号的前面

        # Step 4: 上变频到射频（如果需要）
        num_samples = len(ofdm_signal)
        t = np.arange(num_samples) / self.fs  # 时间轴
        carrier_i = np.cos(2 * np.pi * self.fc * t)  # I 路载波
        carrier_q = -np.sin(2 * np.pi * self.fc * t)  # Q 路载波

        # 上变频：将 I/Q 调制信号与载波相乘
        ofdm_i = np.real(ofdm_signal) * carrier_i
        ofdm_q = np.real(ofdm_signal) * carrier_q

        # 将 I/Q 分量加和得到最终的发射信号
        passband_signal = ofdm_i + 1j * ofdm_q

        # Step 5: 发送信号
        RFLoopback(passband_signal, self.pcip, self.xsrpip)