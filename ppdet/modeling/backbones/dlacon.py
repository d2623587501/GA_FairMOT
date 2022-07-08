# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec
import numpy as np

DLA_cfg = {34: ([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512])}

class Identity(nn.Layer):
    def __init_(self):
        super().__init__()

    def forward(self, x):
        return x

class GlobalAttention(nn.Layer):
    def __init__(self, in_H, in_W, in_channels):  # in_H 608; in_W 1088
        super().__init__()
        self.in_H=in_H
        self.in_W=in_W

        self.conv_h1 = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=[in_H, 1], stride=1, padding=0)
        self.conv_w1 = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=[1, in_W], stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(in_channels)
        self.relu1 = nn.ReLU()

        self.conv_h2 = nn.Conv2D(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv_w2 = nn.Conv2D(in_channels=in_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2D(1)
        self.relu2 = nn.ReLU()
        
        self.fla = paddle.nn.Flatten(start_axis=1, stop_axis=3)
        self.lin_h1 = nn.Linear(in_W,in_W*2)
        self.lin_w1 = nn.Linear(in_H,in_H*2)
        self.lin_h2 = nn.Linear(in_W*2,in_W)
        self.lin_w2 = nn.Linear(in_H*2,in_H)

        self.conv_h3 = nn.Conv2D(in_channels=1, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w3 = nn.Conv2D(in_channels=1, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2D(in_channels)
        self.relu3 = nn.ReLU()

        self.conv_con = nn.Conv2D(in_channels=in_channels*2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.bn_con = nn.BatchNorm2D(in_channels)

        self.sigmoid = nn.Sigmoid()
        self.relu4 = nn.ReLU()
        

    def forward(self, x):
        feats = x

        x_h = self.conv_h1(x)
        x_w = self.conv_w1(x)
        x_h = self.bn1(x_h)
        x_w = self.bn1(x_w)
        # x_h = self.relu1(x_h)
        # x_w = self.relu1(x_w)
        # x_h [6, 16, 1, 1088] ; x_w [6, 16, 608, 1]

        x_h = self.conv_h2(x_h)
        x_w = self.conv_w2(x_w)
        x_h = self.bn2(x_h)
        x_w = self.bn2(x_w)
        # x_h = self.relu2(x_h)
        # x_w = self.relu2(x_w)
        # x_h [6, 1, 1, 1088] ; x_w [6, 1, 608, 1]

        x_h = self.fla(x_h)
        x_w = self.fla(x_w)
        # x_h [6, 1088] ; x_w [6, 608]

        x_h = self.lin_h1(x_h)
        x_w = self.lin_w1(x_w)
        x_h = self.lin_h2(x_h)
        x_w = self.lin_w2(x_w)
        # x_h [6, 1088] ; x_w [6, 608]

        x_h = paddle.reshape(x_h,[x.shape[0],1,1,-1])
        x_w = paddle.reshape(x_w,[x.shape[0],1,-1,1])
        # x_h [6, 1, 1, 1088] ; x_w [6, 1, 608, 1]

        x_h = self.conv_h3(x_h)
        x_w = self.conv_w3(x_w)
        x_h = self.bn3(x_h)
        x_w = self.bn3(x_w)
        # x_h = self.relu3(x_h)
        # x_w = self.relu3(x_w)
        # # x_h [6, 16, 1, 1088] ; x_w [6, 16, 608, 1]

        x_att = paddle.mm(x_w,x_h)
        x_att = self.sigmoid(x_att)
        # x_att [6, 16, 608, 1088]

        # W = paddle.static.create_parameter(shape=[self.in_H, self.in_W], dtype='float32')
        # x_att = paddle.multiply(W, x_att)

        # feats = paddle.add(feats, x_att)

        feats = paddle.concat([feats, x_att], axis=1)
        # feats [6, 32, 608, 1088]
        feats = self.conv_con(feats)
        feats = self.bn_con(feats)
        
        return feats

class BasicBlock(nn.Layer):
    def __init__(self, ch_in, ch_out, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=3,
            stride=stride,
            bias_on=False,
            norm_decay=None)
        self.conv2 = ConvNormLayer(
            ch_out,
            ch_out,
            filter_size=3,
            stride=1,
            bias_on=False,
            norm_decay=None)

    def forward(self, inputs, residual=None):
        if residual is None:
            residual = inputs

        out = self.conv1(inputs)
        out = F.relu(out)

        out = self.conv2(out)

        out = paddle.add(x=out, y=residual)
        out = F.relu(out)

        return out


class Root(nn.Layer):
    def __init__(self, ch_in, ch_out, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = ConvNormLayer(
            ch_in,
            ch_out,
            filter_size=1,
            stride=1,
            bias_on=False,
            norm_decay=None)
        self.residual = residual

    def forward(self, inputs):
        children = inputs
        out = self.conv(paddle.concat(inputs, axis=1))
        if self.residual:
            out = paddle.add(x=out, y=children[0])
        out = F.relu(out)

        return out


class Tree(nn.Layer):
    def __init__(self,
                 level,
                 block,
                 ch_in,
                 ch_out,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * ch_out
        if level_root:
            root_dim += ch_in
        if level == 1:
            self.tree1 = block(ch_in, ch_out, stride)
            self.tree2 = block(ch_out, ch_out, 1)
        else:
            self.tree1 = Tree(
                level - 1,
                block,
                ch_in,
                ch_out,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)
            self.tree2 = Tree(
                level - 1,
                block,
                ch_out,
                ch_out,
                1,
                root_dim=root_dim + ch_out,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual)

        if level == 1:
            self.root = Root(root_dim, ch_out, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.level = level
        if stride > 1:
            self.downsample = nn.MaxPool2D(stride, stride=stride)
        if ch_in != ch_out:
            self.project = ConvNormLayer(
                ch_in,
                ch_out,
                filter_size=1,
                stride=1,
                bias_on=False,
                norm_decay=None)

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root([x2, x1] + children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@register
@serializable
class DLA(nn.Layer):
    """
    DLA, see https://arxiv.org/pdf/1707.06484.pdf

    Args:
        depth (int): DLA depth, should be 34.
        residual_root (bool): whether use a reidual layer in the root block

    """

    def __init__(self, depth=34,ga_block=False,residual_root=False,in_H=608, in_W=1088):
        super(DLA, self).__init__()
        levels, channels = DLA_cfg[depth]
        if depth == 34:
            block = BasicBlock
        self.channels = channels
        self.base_layer = nn.Sequential(
            ConvNormLayer(
                3,
                channels[0],
                filter_size=7,
                stride=1,
                bias_on=False,
                norm_decay=None),
            nn.ReLU())
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root)

        if ga_block:
            self.ga = GlobalAttention(in_H, in_W, in_channels=16)
        else:
            self.ga = Identity()

    def _make_conv_level(self, ch_in, ch_out, conv_num, stride=1):
        modules = []
        for i in range(conv_num):
            modules.extend([
                ConvNormLayer(
                    ch_in,
                    ch_out,
                    filter_size=3,
                    stride=stride if i == 0 else 1,
                    bias_on=False,
                    norm_decay=None), nn.ReLU()
            ])
            ch_in = ch_out
        return nn.Sequential(*modules)

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.channels[i]) for i in range(6)]

    def forward(self, inputs):
        outs = []
        im = inputs['image']
        feats = self.base_layer(im)
        
        feats = self.ga(feats)
        
        for i in range(6):
            feats = getattr(self, 'level{}'.format(i))(feats)
            outs.append(feats)

        return outs
