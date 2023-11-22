import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, K_dim_e, h1_dim_e, h2_dim_e, lat_dim ):
        super().__init__()

        self.K_dim = K_dim_e
        self.h1_dim = h1_dim_e
        self.h2_dim = h2_dim_e
        self.lat_dim = lat_dim

        self.fc = nn.Sequential(
            nn.Linear(K_dim_e, h1_dim_e),
            nn.ReLU(),
            nn.Linear(h1_dim_e, h2_dim_e),
            nn.ReLU()
        )

        self.mu = nn.Linear(h2_dim_e, lat_dim)
        self.logvar = nn.Linear(h2_dim_e, lat_dim)


    def forward(self, x):
        x = self.fc(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

class PartialVAE_Encoder(nn.Module):
    def __init__(self, K_dim_e, h1_dim_e, h2_dim_e, lat_dim, data_dim, emb_dim, prob_num):
        super(PartialVAE_Encoder, self).__init__()

        self.K_dim =K_dim_e
        self.h1_dim =h1_dim_e
        self.h2_dim =h2_dim_e
        self.lat_dim = lat_dim
        self.data_dim = data_dim
        self.emb_dim = emb_dim
        self.prob_num = prob_num

        self.encoder = Encoder(K_dim_e, h1_dim_e, h2_dim_e, lat_dim)

        self.pnnn = nn.Sequential(
                    nn.Linear(emb_dim, K_dim_e),
                    nn.ReLU()
        )
        self.embedding_layer = nn.Embedding(2*self.prob_num, self.emb_dim)


    def embedding_net(self, x, mask):
        masked_batch=[]
        for i in range(len(x)):
            gin2=[]
            for j in range(len(x[i])):
                embed_x= self.embedding_layer(x[i][j])
                gin2.append(embed_x)
            emb = torch.stack(gin2, dim =0)

            pnnn_dataset =[]

            for k, value in enumerate(mask[i].squeeze()):
                if value ==True:
                    pnnn_data = self.pnnn(emb[k])
                    pnnn_dataset.append(pnnn_data)
            masked_pnnn = torch.stack(pnnn_dataset, dim =0)
            pnc = masked_pnnn.sum(dim=0)

            masked_batch.append(pnc)
        masked_batch_tensor = torch.stack(masked_batch, dim =0)
        return masked_batch_tensor

    def representation_trick(self, lat_dim, mu, logvar):
        z = [ ]
        for j in range(len(mu)):
                epsilon = torch.randn(lat_dim)
                z_data= mu[j] + epsilon * torch.exp(0.5*logvar[j])
                z.append(z_data)
        z_batch =torch.stack(z, dim =0) 
        
        return z_batch

    def forward(self, x, mask):
        masked_batch_tensor = self.embedding_net(x, mask)
        mu, logvar = self.encoder(masked_batch_tensor)
        z = self.representation_trick(self.lat_dim, mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, lat_dim, h1_dim_d, h2_dim_d, h3_dim_d, data_dim, prob_num):
        super().__init__()

        self.data_dim = data_dim
        self.h1_dim = h1_dim_d
        self.h2_dim = h2_dim_d
        self.h3_dim = h3_dim_d
        self.lat_dim = lat_dim
        self.prob_num = prob_num

        self.fc = nn.Sequential(
            nn.Linear(lat_dim, h1_dim_d),
            nn.ReLU(),
            nn.Linear(h1_dim_d, h2_dim_d),
            nn.ReLU(),
            nn.Linear(h2_dim_d, h3_dim_d),
            nn.ReLU(),
            nn.Linear(h3_dim_d, 2*self.data_dim*self.prob_num)
        )

    def forward(self, z):
        y= self.fc(z)
        return y