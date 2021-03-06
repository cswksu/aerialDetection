import torch
import torch.nn as nn
import torch.nn.functional as F

class UpConv(nn.Module):
    def __init__(self, in_features, out_features, should_batch_norm):
        super().__init__()
        self.should_batch_norm = should_batch_norm
        self.upscaler = nn.ConvTranspose2d(in_features, out_features, 2, 2)
        self.conv1 = nn.Conv2d(2*out_features, out_features, 3, 1, 1, bias=(not should_batch_norm))
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1, bias=(not should_batch_norm))
        self.act0 = nn.ReLU()
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        if self.should_batch_norm:
            self.norm1 = nn.BatchNorm2d(out_features)
            self.norm2 = nn.BatchNorm2d(out_features)

    
    def forward(self, encoder_input, decoder_input):
        upsampled = self.upscaler(decoder_input)
        upsampled = self.act0(upsampled)
        stacked_layer = torch.cat((upsampled, encoder_input), 1)
        out = self.conv1(stacked_layer)
        if self.should_batch_norm:
            out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        if self.should_batch_norm:
            out = self.norm2(out)
        out = self.act2(out)
        return out

    

class DownConv(nn.Module):
    def __init__(self, in_features, out_features, should_pool, should_batch_norm):
        super().__init__()
        self.should_batch_norm = should_batch_norm
        self.conv1 = nn.Conv2d(in_features, out_features, 3, 1, 1, padding_mode='reflect', bias=(not self.should_batch_norm))
        self.conv2 = nn.Conv2d(out_features, out_features, 3, 1, 1, padding_mode='reflect', bias=(not self.should_batch_norm))
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.should_pool = should_pool
        
        if should_pool:
            self.pool = nn.MaxPool2d(2, 2)
        if self.should_batch_norm:
            self.norm1 = nn.BatchNorm2d(out_features)
            self.norm2 = nn.BatchNorm2d(out_features)
    
    def forward(self, x):
        y = self.conv1(x)
        if self.should_batch_norm:
            y = self.norm1(y)
        y = self.act1(y)
        y = self.conv2(y)
        if self.should_batch_norm:
            y = self.norm2(y)
        y = self.act2(y)
        full_size = y
        if self.should_pool:
            y = self.pool(y)
        return y, full_size


class ResUNet(nn.Module):
    def __init__(self, num_blocks = 3, first_block_features=64, batch_norm=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.first_block_features = first_block_features

        self.upsample_blocks = []
        self.downsample_blocks = []

        last_layer_features = 3
        targ_layer_features = self.first_block_features
        for layer_num in range(self.num_blocks-1):
            thisDownBlock = DownConv(last_layer_features, targ_layer_features, True, batch_norm)
            self.downsample_blocks.append(thisDownBlock)
            last_layer_features = targ_layer_features
            targ_layer_features *= 2
        thisDownBlock = DownConv(last_layer_features, targ_layer_features, False, batch_norm)
        self.downsample_blocks.append(thisDownBlock)
        last_layer_features = targ_layer_features
        targ_layer_features = last_layer_features // 2

        for layer_num in range(self.num_blocks-1):
            thisUpBlock = UpConv(last_layer_features, targ_layer_features, batch_norm)
            self.upsample_blocks.append(thisUpBlock)
            last_layer_features = targ_layer_features
            targ_layer_features = targ_layer_features // 2
        self.conv = nn.Conv2d(last_layer_features, 2, 1, 1)
        self.dropout = nn.Dropout2d(0.25)
        #self.softmax = nn.Softmax()
        #self.sigmoid = nn.Sigmoid()
        self.upsample_blocks = nn.ModuleList(self.upsample_blocks)
        self.downsample_blocks = nn.ModuleList(self.downsample_blocks)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if not batch_norm:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        skip_connections = []
        for module in self.downsample_blocks:
            x, before_pool = module(x)
            skip_connections.append(before_pool)
        for idx, module in enumerate(self.upsample_blocks):
            concat_block = skip_connections[-idx-2]
            x = module(concat_block, x)
        x = self.dropout(x)
        x = self.conv(x)
        #x = self.softmax(x)
        return x
