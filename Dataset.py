import torch
from torch.utils.data import Dataset
import pandas as pd
import csv


class pvaeDataset(Dataset): #Dataloader를 쓰려면 Dataset instance일 필요가 있으므로 Dataset을 상속해야함.
    def __init__(self, data_path, max_len, prob_num):
        super(pvaeDataset).__init__()
        self.data=[]
        self.data_path = data_path
        self.max_len = max_len
        self.prob_num = prob_num
        self.preprocess()

    def preprocess(self):
        step =3
        with open(self.data_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                row = [value.strip() for value in row if value.strip()]
                if i % step ==0:
                    temp=[row]
                else:
                    temp.append(row)

                if i % step ==2:
                    if int(float(temp[0][0]))>= self.max_len:
                        student_log = [ int(float(temp[1][j]))+self.prob_num *int((float(temp[2][j]))) for j in range(self.max_len)] # To avoid collision at calculating loss
                        self.data.append(student_log)


    def __getitem__(self, index):
        student_idx = self.data[index]

        return torch.tensor(student_idx)

    def __len__(self):
        return len(self.data)