import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channel, first_out_channel):
        super(UNet, self).__init__()
        foc = first_out_channel
        self.b1 = self.createBlock(in_channel, foc)
        torch.nn.init.kaiming_uniform_(self.b1[0].weight)
        torch.nn.init.kaiming_uniform_(self.b1[3].weight)
        self.maxPooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b2 = self.createBlock(foc, foc*2)
        torch.nn.init.kaiming_uniform_(self.b2[0].weight)
        torch.nn.init.kaiming_uniform_(self.b2[3].weight)
        self.maxPooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b3 = self.createBlock(foc*2, foc*4)
        torch.nn.init.kaiming_uniform_(self.b3[0].weight)
        torch.nn.init.kaiming_uniform_(self.b3[3].weight)
        self.maxPooling3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b4 = self.createBlock(foc*4, foc*8)
        torch.nn.init.kaiming_uniform_(self.b4[0].weight)
        torch.nn.init.kaiming_uniform_(self.b4[3].weight)
        self.convTrans1 = nn.ConvTranspose2d(foc*8, foc*4, kernel_size=2, stride=2,bias=False)
        torch.nn.init.kaiming_uniform_(self.convTrans1.weight)
        self.b5 = self.createBlock(foc*8, foc*4)
        torch.nn.init.kaiming_uniform_(self.b5[0].weight)
        torch.nn.init.kaiming_uniform_(self.b5[3].weight)
        self.convTrans2 = nn.ConvTranspose2d(foc*4, foc*2, kernel_size=2, stride=2,bias=False)
        torch.nn.init.kaiming_uniform_(self.convTrans2.weight)
        self.b6 = self.createBlock(foc*4, foc*2)
        torch.nn.init.kaiming_uniform_(self.b6[0].weight)
        torch.nn.init.kaiming_uniform_(self.b6[3].weight)
        self.convTrans3 = nn.ConvTranspose2d(foc*2, foc, kernel_size=2, stride=2,bias=False)
        torch.nn.init.kaiming_uniform_(self.convTrans3.weight)
        self.b7 = self.createBlock(foc*2, foc)
        torch.nn.init.kaiming_uniform_(self.b7[0].weight)
        torch.nn.init.kaiming_uniform_(self.b7[3].weight)
        self.b8 = nn.Conv2d(in_channels=foc, out_channels=1, kernel_size=1,bias=False)
        torch.nn.init.kaiming_uniform_(self.b8.weight)

    def forward(self,im):
        x1 = self.b1(im) # 1,32,(H,W) 
        x = self.maxPooling1(x1) # 1,32,(1/2)*(H,W) 
        x2 = self.b2(x) # 1,64,(1/2)*(H,W)
        x = self.maxPooling2(x2) # 1,64,(1/4)*(H,W) 
        x3 = self.b3(x) # 1,128,(1/4)*(H,W)
        x = self.maxPooling3(x3) # 1,128,(1/8)*(H,W) 
        x = self.b4(x) # 1,256,(1/8)*(H,W)
        x = self.convTrans1(x) # 1,128,(1/4)*(H,W)
        x = torch.cat((x, x3), dim=1) # 1,256,(1/4)*(H,W)
        x = self.b5(x) # 1,128,(1/4)*(H,W)
        x = self.convTrans2(x) # 1,64,(1/2)*(H,W)
        x = torch.cat((x, x2), dim=1) # 1,128,(1/2)*(H,W)
        x = self.b6(x) # 1,64,(1/2)*(H,W)
        x = self.convTrans3(x) # 1,32,(H,W)
        x = torch.cat((x, x1), dim=1) # 1,64,(H,W)
        x = self.b7(x) # 1,32,(H,W)
        x = self.b8(x)  # 1,1,(H,W)
        x = torch.sigmoid(x)
        return x      

    @staticmethod
    def createBlock(in_chnl, out_chnl):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=3, padding=1,bias=False),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_chnl, out_channels=out_chnl, kernel_size=3, padding=1, bias=False),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True)
            )
