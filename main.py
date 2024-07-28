import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

# device = torch.device('cpu')
device = torch.device('cuda')


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# 定义模型的参数
input_size = 56
hidden_size = 64
num_classes = 6

# 创建模型
model = LSTM(input_size, hidden_size, num_classes)
model = model.to(device)

for pack_num in [24]:
    test_path = f'dataset.npz'
    test_data = np.load(test_path)
    X_test_resampled = test_data['X']
    y_test_resampled = test_data['y']
    mask = y_test_resampled < 6
    X_test_resampled = X_test_resampled[mask]
    y_test_resampled = y_test_resampled[mask]
    X_test_resampled = X_test_resampled.reshape(-1, pack_num - 1, input_size)

    test_dataset = NumpyDataset(X_test_resampled, y_test_resampled)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    temacc = 0.5
    t = -1
    # %%
    model.load_state_dict(torch.load("model.pth"))
    # %%
    model.eval()
    total_loss = 0
    total_correct = 0

    total_infer_time = 0
    total_lat_time = 0
    num_batches = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_dataloader):
            torch.cuda.synchronize()
            latstaretime = time.perf_counter()

            data = data.to(torch.float32)
            mean = data.mean()
            std = data.std()
            data = (data - mean) / std
            labels = labels.flatten()
            labels = labels.to(torch.long)
            data = data.to(device)
            labels = labels.to(device)

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            output_probabilities = model(data)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            infertime = end_time - start_time
            total_infer_time += infertime
            loss = criterion(output_probabilities, labels)
            _, predicted_labels = torch.max(output_probabilities, 1)

            torch.cuda.synchronize()
            latendtime = time.perf_counter()
            lattime = latendtime - latstaretime
            total_lat_time += lattime

            num_batches += 1

            # print(f"infertime Time: {infertime:.6f} seconds")
            # print(f"lattime Time: {lattime:.6f} seconds")
            # print(f"-------------------------------------------------")

    avg_infer_time = total_infer_time / num_batches
    avg_lat_time = total_lat_time / num_batches

    print(f"Average Inference Time: {avg_infer_time:.6f} seconds")
    print(f"Average Latency Time: {avg_lat_time:.6f} seconds")
