import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])  # 不會做反向傳遞
# 把雞蛋(tensor)都到籃子(variable)裡
variable = Variable(tensor, requires_grad=True)  # 誤差反向傳遞時會不會計算節點的梯度

print('\ntensor', '\n', tensor)
print('\nvariable', '\n', variable)

t_out = torch.mean(tensor*tensor)  # x^2
v_out = torch.mean(variable*variable)

print('\ntensor', '\n', t_out)
print('\nvariable', '\n', v_out)

v_out.backward()
# 誤差的反向傳遞
# v_out = 1/4 * sum(var*var)
# d(v_out)/d(var) = 1/4 * 2 * variable = variable / 2
print(variable.grad)

print(variable)  # tensor有containing
print(variable.data)  # tensor無containing
print(variable.data.numpy())  # tensor轉成numpy
