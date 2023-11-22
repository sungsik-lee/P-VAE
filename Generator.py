import csv
import torch

def target_load(data_path, prob_num, max_len):
    data=[]
    step =3
    with open(data_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i, row in enumerate(csv_reader):
            if i % step ==0:
                 temp=[row]
            else:
                temp.append(row)

            if i % step ==2:
                if int(float(temp[0][0]))< max_len:
                    student_log = [int(float(temp[1][j]))+prob_num *int(float(temp[2][j])) for j in range(int(float(temp[0][0])))]
                    student_log= torch.tensor(student_log)
                    data.append([int(i-2), student_log ])
    return data


def making_mask(target_Data, data_dim):
    total_mask = []
    for i in range(len(target_Data)):
        zero_tensor = torch.zeros(data_dim, dtype=int)
        zero_tensor[ : len(target_Data[i][1])] = 1
        total_mask.append(zero_tensor)
    return total_mask


def predict_list(Aug_data, data_dim, prob_num):
    Total_data=[]

    for i ,_ in enumerate(Aug_data):
        Student_log = []
        Student_log.append([str(data_dim)])

        prob_idx=[]
        for j , log in enumerate(Aug_data[i][1][0]):
            if int(log)  >= prob_num:
                prob_idx.append(log - prob_num)
            else:
                prob_idx.append(log)
        prob_idx = [x.item() for x in prob_idx]

        Student_log.append(prob_idx)

        prob_ans=[]
        for j , log in enumerate(Aug_data[i][1][0]):
            if int(log) >= prob_num:
                prob_ans.append(1)
            else:
                prob_ans.append(0)
        Student_log.append(prob_ans)
        Student_log.append(Aug_data[i][0])
        Total_data.append(Student_log)
    return Total_data    

def csv_maker(data_path, new_data_path, Total_data, max_len):
    
    #Export full generated data by csv

    with open(data_path, "r") as csvfile, open(new_data_path, "w", newline="") as tempcsvfile:
        csv_reader = csv.reader(csvfile)
        csv_writer = csv.writer(tempcsvfile)

        for i, row in enumerate(csv_reader):
            row = [value.strip() for value in row if value.strip()]
            step =3
            if i % step ==0:
                if int(row[0])< max_len:
                    for j , data in enumerate(Total_data):
                        if  int(data[3]) == i:
                            csv_writer.writerow(data[0])
                            csv_writer.writerow(data[1])
                            csv_writer.writerow(data[2])
                else:
                    csv_writer.writerow(row)
            else:
                if len(row) >= max_len:
                    csv_writer.writerow(row)
