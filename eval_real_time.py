import tkinter

import torch
import matplotlib.pyplot as plt
import numpy as np
import models
import socket
import sklearn.preprocessing
import scipy.io

class UDP_config:
    def __init__(self):
        self.sample_length=30720

udp_cfg=UDP_config()

def main():
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
    print("abs: {},{}".format(np.mean(np.abs(udp_data_ri)),np.mean(np.abs(udp_data_rq))))
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
    #sample_mod = scipy.io.loadmat('./data/QAM16.mat')['moddata'][0]
    #print("abs: {},{}".format(np.mean(np.abs(np.real(sample_mod))), np.mean(np.abs(np.imag(sample_mod)))))
    each_sample_num = 200
    input_data=[]
    for i in range(each_sample_num):
        current_frame = sample_mod[i * 128:(i + 1) * 128]
        current_frame = np.concatenate((np.real(current_frame), np.imag(current_frame)), axis=0)
        input_data.append(current_frame)
    input_data = np.vstack(input_data)
    input_data = input_data.reshape((-1, 256))
    #scaler = sklearn.preprocessing.MinMaxScaler()
    #input_data = scaler.fit_transform(input_data)
    for i in range(input_data.shape[0]):
        input_data[i]=input_data[i]/np.max(input_data[i])
    input_data = input_data.reshape((-1, 1, 2, 128))

    MODEL_PATH1 = 'checkpoints/5_best_model1_x.pth'
    MODEL_PATH2 = 'checkpoints/4_best_model2_x.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    model_main = models.DnnNet0()
    model_main = model_main.to(DEVICE)
    model_main.load(MODEL_PATH1)
    model_conf = models.DnnNet0()
    model_conf = model_conf.to(DEVICE)
    #model_conf.load(MODEL_PATH2)
    mod_type=['qam16','qam64','qpsk','fm','am','bpsk']
    pred_count1=np.zeros(6)
    for i in range(each_sample_num):
        current_input=torch.tensor(input_data[i],dtype=torch.float32).to(DEVICE)
        output=model_main(current_input)
        predict = torch.argmax(output, 1)
        pred_count1[predict.cpu()]=pred_count1[predict.cpu()]+1
    print("model1 result:")
    for i in range(6):
        print("{}: {}".format(mod_type[i],pred_count1[i]/each_sample_num))
    # pred_count2 = np.zeros(6)
    # for i in range(each_sample_num):
    #     current_input = torch.tensor(input_data[i], dtype=torch.float32).to(DEVICE)
    #     output = model_conf(current_input)
    #     predict = torch.argmax(output, 1)
    #     pred_count2[predict.cpu()] = pred_count2[predict.cpu()] + 1
    # print("model2 result:")
    # for i in range(6):
    #     print("{}: {}".format(mod_type[i], pred_count2[i] / each_sample_num))

if __name__=="__main__":
    main()