

请在如下概率二值CNN网络的基础上修改，变更模型中的数据引入和模型定义部分，构建概率二值MLP模型：

```
torch.manual_seed(1)
torch.cuda.manual_seed(1)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

# CNN 网络定义
# 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = PBinarizeConv2d(1, 32, kernel_size=3)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = PBinarizeConv2d(32, 64, kernel_size=3)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = PBinarizeLinear(1600, 512)
        self.fc2 = PBinarizeLinear(512, 10)


    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
  
    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = x.view(x.size(0), -1)
      x = self.fc1(x)
      x = self.fc2(x)
      return x
  

model = Net()
print(model)
torch.cuda.device('cuda')
model.cuda()


# 训练和测试
def train(epoch):
    model.train()
    losses = []
    trainloader = tqdm(train_loader)
    
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

#         if epoch%40==0:
#             optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        loss.backward()
    
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-0.9,0.9))
    
        losses.append(loss.item())
        trainloader.set_postfix(loss=np.mean(losses), epoch=epoch)



def test():
    model.eval()
    test_loss = 0
    correct = 0
    testloader = tqdm(test_loader)
    for data, target in testloader:
        data, target = data.cuda(), target.cuda()
        with torch.no_grad():
          data = Variable(data)
        target = Variable(target)
        output = model(data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        loss = test_loss / len(test_loader.dataset)
        acc = (100. *correct / len(test_loader.dataset)).numpy()

        testloader.set_postfix(loss= loss,acc=str(acc)+'%')
    
    test_loss /= len(test_loader.dataset)
    total_acc = correct.item()/len(test_loader.dataset)
    loss_history.append(test_loss)
    acc_history.append(total_acc)
    
    
    
%%time
loss_history = []
acc_history = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(20):
    train(epoch)
    test()
```



数据引入方面参考如下代码，你需要将pd.DataFrame转化为pytorch中的DataLoader，请注意，由于输入数据特征维度的不同，你可能需要自定义输入层参数，以使得模型维度和输入数据匹配：

```
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


# Iris
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# 处理缺失值
missing_threshold = 0.3
missing_ratio = X.isnull().mean()
X = X.drop(columns=missing_ratio[missing_ratio > missing_threshold].index)  # 删除缺失值比例超过30%的特征

# 处理数据类型转换
for col in X.select_dtypes(include=['object']).columns:
    if X[col].nunique() < 20:  # 类别变量进行编码
        X[col] = LabelEncoder().fit_transform(X[col])
    else:  # 删除冗余字符串特征 (通常为注释)
        X.drop(columns=[col], inplace=True)

# 对剩余缺失值采用中位数填充
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 归一化特征
scaler_minmax = MinMaxScaler()
scaler_zscore = StandardScaler()

# 对具有高斯分布的特征进行z-score标准化，否则进行min-max归一化
X_scaled = X_imputed.copy()
for col in X_scaled.columns:
    if np.abs(X_scaled[col].skew()) < 1:  # 判断是否接近正态分布
        X_scaled[col] = scaler_zscore.fit_transform(X_scaled[[col]])
    else:
        X_scaled[col] = scaler_minmax.fit_transform(X_scaled[[col]])

# 处理类别不均衡
try:
    smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
except:
    (X_resampled, y_resampled) = (X_scaled, y)

# 分层拆分数据集（保持类别比例）
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# 查看处理后数据集大小
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
```

