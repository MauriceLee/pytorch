import torch
import torch.utils.data as Data  # 進行批次(小批)訓練的途徑

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5  # 一小批有5個，5個5個來訓練
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # this is y data (torch tensor)

# 用torch來定義個數據庫，利用Data的型式，把x,y放到數據庫裡
# 使用數據x來訓練(data_tensor)，使用數據y來算誤差(target_tensor)
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training 每次訓練會不會打亂
    # subprocesses for loading data (每次提取batch時，都使用雙線程) 會更有效率
    num_workers=2,
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        # for each training step, enumerate=>每一次提取的時候都給他一個索引(step)
        for step, (batch_x, batch_y) in enumerate(loader):
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()
