import torch

def generate_random_mask(batch_size, data_dim):
    batch_mask =[]
    for i in range(batch_size):
        len_one = torch.randint(1, data_dim ,(1,))
        mask = torch.cat((torch.ones( data_dim - len_one,  ), torch.zeros(len_one , )), dim=0)

        batch_mask.append(mask)
    batch_mask_tensor = torch.stack(batch_mask, dim =0)
    batch_mask_tensor = batch_mask_tensor.unsqueeze(-1)
    batch_mask_tensor = batch_mask_tensor.to(torch.bool)
    return batch_mask_tensor