from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler

from Loss import *
from Dataset import *
from Masking import *
from Cfg import get_cfg
from P_VAE import *
from Generator import *

if __name__ == "__main__":
    
    cfg = get_cfg()
    max_len = cfg.max_len
    
    if max_len == 20:
        K_dim_e = 50
        h1_dim_e = 400
        h2_dim_e = 500
        lat_dim = 5
        data_dim = 20 
        
        h1_dim_d = 300
        h2_dim_d = 1500
        h3_dim_d = 2500
        
    elif max_len == 50:
        K_dim_e = 50
        h1_dim_e = 300
        h2_dim_e = 400
        lat_dim = 5
        data_dim = 50 

        h1_dim_d = 200
        h2_dim_d = 1000
        h3_dim_d = 2000
        
    elif max_len == 200:
        K_dim_e = 100
        h1_dim_e = 700
        h2_dim_e = 900
        lat_dim = 25
        data_dim = 200 

        h1_dim_d = 500
        h2_dim_d = 3000
        h3_dim_d = 5000        
    
    elif max_len == 300:
        K_dim_e = 100
        h1_dim_e = 550
        h2_dim_e = 650
        lat_dim = 25
        data_dim = 300 

        h1_dim_d = 300
        h2_dim_d = 2000
        h3_dim_d = 4000    
        
    if cfg.data == "ASSISTmentsChall":
        prob_num = 102
        emb_dim = 50
    elif cfg.data == "ASSISTments0910":
        prob_num = 124
        emb_dim = 50
    elif cfg.data == "ASSISTments2015":
        prob_num = 100
        emb_dim = 50      
    elif cfg.data == "STATIC":
        prob_num = 81
        emb_dim = 50
    

    lr = 1e-4
    epoch = 50
    batch_size = 32
    
    step_size = 5
    gamma = 0.7

    '''
    Upload Datasets
    '''
    
    original_data_path = f"Data/{cfg.data}_Processed.csv"
    
    dataset = pvaeDataset(original_data_path, max_len, prob_num)

    train_DL = DataLoader(dataset, batch_size = batch_size, shuffle=True, drop_last=True)

    #define model
    model_encoder = PartialVAE_Encoder(K_dim_e, h1_dim_e, h2_dim_e, lat_dim, data_dim, emb_dim, prob_num)
    model_decoder_1 =Decoder(lat_dim, h1_dim_d, h2_dim_d, h3_dim_d, data_dim, prob_num)

    parameters = list(model_encoder.parameters()) + list(model_decoder_1.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr) 
    

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_stack = []
    NoT = len (train_DL.dataset)

    model_encoder.train()
    model_decoder_1.train()

    for i in range(epoch):
        rloss =0

        for x_batch in train_DL:
            
            mask = generate_random_mask(batch_size, data_dim)


            z, mu, logvar =model_encoder(x_batch, mask)
             
                
            y=model_decoder_1(z)

            obs_ELBO = obs_compute_ELBOs(x_batch, mu, logvar)
            obs_recons = obs_compute_recons(x_batch, mask, y, data_dim, prob_num )

            loss = obs_ELBO +obs_recons
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_b = loss.item()* x_batch.shape[0]
            rloss += loss_b
        scheduler.step()
        #print loss
        loss_real = rloss/NoT
        loss_stack.append(loss_real)
        
        print(f'Epoch: {i+1}, train loss : {round(loss_real, 3)}')
        print('-'*20)

model_decoder_2 =Decoder(lat_dim, h1_dim_d, h2_dim_d, h3_dim_d, data_dim, prob_num)

optimizer2 = torch.optim.Adam(model_decoder_2.parameters(), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

loss_stack = []
NoT = len (train_DL.dataset)
model_decoder_2.train()

for i in range(epoch):
    rloss =0
    for x_batch in train_DL:

        mask = generate_random_mask(batch_size, data_dim)

        z, mu, logvar = model_encoder(x_batch, mask)   
            
        y=model_decoder_2(z)

        loss = unobs_compute_ELBOs(x_batch, mask, y, data_dim, prob_num)

        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        loss_b = loss.item()* x_batch.shape[0]
        rloss += loss_b
    scheduler.step()
    #print loss
    loss_real = rloss/NoT
    loss_stack.append(loss_real)
    print(f'Epoch: {i+1}, train loss : {round(loss_real, 3)}')
    print('-'*20)

#Save model

save_model_path_encoder =f'/Users/sungsiklee/Desktop/Coding/P-VAE/Results/p-vae_encoder{cfg.data}_{max_len}.pt'
save_model_path_decoder =f'/Users/sungsiklee/Desktop/Coding/P-VAE/Results/p-vae_decoder{cfg.data}_{max_len}.pt'

torch.save(model_encoder.state_dict(), save_model_path_encoder)
torch.save(model_decoder_2.state_dict(), save_model_path_decoder)    


target_data= target_load(original_data_path, prob_num, max_len)
mask = making_mask(target_data, data_dim)

Aug_data =[]
for i in range(len(target_data)):

    empty_aug_target = torch.cat((target_data[i][1], torch.zeros(data_dim - target_data[i][1].size(0), dtype=int)), dim=0)
    empty_aug_target = empty_aug_target.reshape(1, data_dim)

    ith_mask = mask[i]
    ith_mask = ith_mask.reshape(1, data_dim, 1)
    f_mask = 1 - ith_mask

    z, mu, logvar = model_encoder( empty_aug_target , ith_mask)
    y= model_decoder_2(z)


    y= y.view(1, data_dim , 2*prob_num )
    max_values, max_indices = torch.max(y, dim=2)
    empty_pred = max_indices*f_mask.squeeze()
    aug_data = empty_pred + empty_aug_target

    Aug_data.append([target_data[i][0] ,aug_data])

#Processed형태로 맞추기    

Total_data = predict_list(Aug_data, data_dim, prob_num)
new_data_path = f'Data/{cfg.data}_{max_len}ver.csv'

Final_csv = csv_maker(original_data_path, new_data_path, Total_data, max_len)
    

