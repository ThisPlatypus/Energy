import torch
import torch.nn as nn

# define the generator of the gan
class Generator(nn.Module):
    def __init__(self,  model_typle, z_d=5):
        super(Generator, self).__init__()
    
        self.model_typle = model_typle
        self.out_scale1 = 12 # The scale is just for setting the number of neurons easier
        self.out_scale2 = 6
        
        self.models = nn.ModuleDict({
            '30m': nn.Sequential(
                
                # first layer
                nn.Linear(z_d, self.out_scale1*5),
                nn.BatchNorm1d(self.out_scale1*5),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*5, self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale1*10, 48),
                nn.Tanh()
            ),
            
            '60m': nn.Sequential(
                         
                # first layer
                nn.Linear(z_d, self.out_scale2*5),
                nn.BatchNorm1d(self.out_scale2*5),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale2*5, self.out_scale2*10),
                nn.BatchNorm1d(self.outa_scale2*10),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale2*10, 24),
                nn.Tanh()  
            )
        })
        
    def forward(self, x):
        out = self.models[self.model_typle](x)
        return out

# define the discriminator of the gan
class Discriminator(nn.Module):
    def __init__(self, model_typle):
        super(Discriminator, self).__init__()
        
        self.model_typle = model_typle
        
        self.out_scale1 = 48 #for 30m
        # 48/48 afher 10 model collapse
        # 24/48 afher 10 model collapse
        self.out_scale2 = 24
        
        self.models = nn.ModuleDict({
            '30m': nn.Sequential(
                
                # first layer
                nn.Linear(48, self.out_scale1*10),
                nn.BatchNorm1d(self.out_scale1*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale1*10, self.out_scale1*5),
                nn.BatchNorm1d(self.out_scale1*5),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale1*5, 1),
                nn.Sigmoid()
            ),
            
            '60m': nn.Sequential(
                
                # first layer
                nn.Linear(24, self.out_scale2*10),
                nn.BatchNorm1d(self.out_scale2*10),
                nn.LeakyReLU(),
                
                # second layer
                nn.Linear(self.out_scale2*10, self.out_scale2*5),
                nn.BatchNorm1d(self.out_scale2*5),
                nn.LeakyReLU(),
                
                # last layer
                nn.Linear(self.out_scale2*5, 1),
                nn.Sigmoid()
                
            )
            })
        
    def forward(self, x):
        score = self.models[self.model_typle](x)
        return score