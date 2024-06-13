import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self,in_channels=1, out_channels=2, features=[64,128,256,512], ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)

        # Downsampling in UNET:
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Upsampling in UNET:
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        #print(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # reversing the skip connection before we go to expand part(decoder)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                print(idx)
                print(x.shape , skip_connection.shape)
                skip_connection = self.crop(skip_connection, x.shape[2], x.shape[3])
                print(x.shape , skip_connection.shape)
   

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
    def crop(self, tensor, target_height, target_width):
        _,_, height, width = tensor.size() # (batch_size, channels, height, width)
        delta_height= height - target_height
        delta_width = width - target_width
        crop_top = delta_height // 2
        crop_left = delta_width // 2

        return tensor[:, :, crop_top:crop_top+target_height, crop_left:crop_left + target_width]



def test():
    x = torch.randn((3,1,572,572))
    model = UNET(in_channels=1, out_channels=2)
    pred = model(x)
    print(x.shape)
    print(pred.shape)
    #assert preds.shape == x.shape

if __name__ == "__main__":
    test()

    # Outputs: 
    # 
    # x.shape: torch.Size([3, 1, 572, 572]) , having 1 channel and desired input
    # predicted.shape(pred.shape): torch.Size([3, 2, 388, 388])
    # Size output at end is smaller due to "Valid Padding == 0" being used