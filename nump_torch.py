import torch
import numpy as np

# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy', '\n', np_data,
#     '\ntorch', '\n', torch_data,
#     '\ntensor2array', '\n', tensor2array
# )

# abs
# data = [-1, -2, 1, 2]
# tensor = torch.FloatTensor(data)  # 32bit
# print(
#     tensor,
#     np.abs(data),
#     torch.abs(tensor),
#     np.sin(data),
#     torch.sin(tensor),
#     np.mean(data),
#     torch.mean(tensor),
# )

data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
data = np.array((data))

print(
    '\nnumpy', '\n', np.matmul(data, data),
    '\nnumpy', '\n', data.dot(data),
    '\ntorch', '\n', torch.mm(tensor, tensor),
    # '\ntorch', '\n', tensor.dot(tensor)
)
