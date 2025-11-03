import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("GPU是否可用：", torch.cuda.is_available())  # 输出True则表示GPU可用
print("GPU设备名：", torch.cuda.get_device_name(0))  # 显示GPU型号（如NVIDIA GeForce RTX 3060）
print(torch.cuda.is_available())

print(device)
print("模型设备：", next(self.dqn.parameters()).device)  # 应输出"cuda:0"
print("数据设备：", sa_tensor.device)  # 应输出"cuda:0"