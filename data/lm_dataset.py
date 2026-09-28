import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, tokens, context_size):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        #Tensor — это по сути массив чисел,
        #похожий на NumPy array,
        #но PyTorch умеет использовать его для нейросетей и вычислений на GPU.
        self.context_size = context_size

    def __len__(self):
        #Сколько обучающих примеров есть в Dataset?
        # Если
        # 100 токенов
        # context_size = 10
        return (len(self.tokens) - 1) // self.context_size

    def __getitem__(self, index):
        #Дай мне обучающий пример номер index
        start = index * self.context_size
        chunk = self.tokens[start:start+ self.context_size +1]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
