import time
import numpy as np
import yaml
from modulation import modulation_x
Modulator
def send_to_x(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        all_cfg = yaml.safe_load(f)
    mod_cfg = all_cfg.get('modulation', {})
    modulator = Modulator(mod_cfg)
    modulator.run()  # 执行调制、发送逻辑
def main():
    print("program start")
    send_to_x()
    print("sending data finished")
    
    # Perform your UDP communication and other operations here

if __name__ == "__main__":
    main()