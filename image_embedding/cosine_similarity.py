import torch
import torch.nn as nn
import numpy as np

def compute_similarity_torch(vector1, vector2, method='cosine'):
    if method == 'euclidean':
        return torch.norm(vector1 - vector2, p=2, dim=(1, 2, 3)).item()
    elif method == 'cosine':
        vector1 = torch.from_numpy(vector1).float()  # 使用 float32 提高精度
        vector2 = torch.from_numpy(vector2).float()  # 使用 float32 提高精度
        vector1 = vector1.reshape(-1)
        vector2 = vector2.reshape(-1)
        
        # 检查向量中是否存在 nan
        if torch.isnan(vector1).any() or torch.isnan(vector2).any():
            print("Warning: Input vectors contain NaN values.")
            return None
        
        # 归一化向量
        vector1 = nn.functional.normalize(vector1, dim=0)
        vector2 = nn.functional.normalize(vector2, dim=0)
        
        cosine_sim = nn.functional.cosine_similarity(vector1, vector2, dim=0)
        return cosine_sim.item()
        

def compute_similarity(vector1, vector2, method='cosine'):  # ndarray
    if method == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif method == 'cosine':
        vector1 = vector1.flatten().astype(np.float32)  # 展平并转换为 float32
        vector2 = vector2.flatten().astype(np.float32)  # 展平并转换为 float32
        
        # 检查向量中是否存在 nan
        if np.isnan(vector1).any() or np.isnan(vector2).any():
            print("Warning: Input vectors contain NaN values.")
            return None
        
        # 归一化向量
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            print("Warning: One of the input vectors has a norm of zero.")
            return None
        
        vector1 = vector1 / norm1
        vector2 = vector2 / norm2
        
        cosine_sim = np.dot(vector1, vector2)
        return cosine_sim     

image1_path = "./dog2.dat"        # dog1.png embedding
image2_path = "./dog2.dat"        # dog2.png embedding
imagev = "./koala1.dat"     # koala1.png embedding

data1 = np.fromfile(image1_path, dtype=np.float32).reshape(256, 4096)  # 使用 float32
data2 = np.fromfile(image2_path, dtype=np.float32).reshape(256, 4096)  # 使用 float32
datav = np.fromfile(imagev, dtype=np.float32).reshape(256, 4096)       # 使用 float32
# import pdb; pdb.set_trace()
# 计算狗与狗之间的相似性
similarity_dog_dog = compute_similarity(data1, data2, method='cosine')

# 计算狗与其他之间的相似性
similarity_dog_other = compute_similarity(data1, datav, method='cosine')

print(f"Similarity between two dogs (Cosine): {similarity_dog_dog}")
print(f"Similarity between dog and other (Cosine): {similarity_dog_other}")