import numbers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class deform_network(nn.Module):

    def __init__(self, points_channel, poses_channel):
        super(deform_network, self).__init__()
        self.embed = torch.nn.Linear(points_channel, 36)
        self.linear1 = torch.nn.Linear(36 + poses_channel, 128)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(128, 128)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(128 + poses_channel, 128)
        self.relu3 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(p=0.5)
        self.linear4 = torch.nn.Linear(128, 128)
        self.relu4 = torch.nn.ReLU()
        self.dropout4 = torch.nn.Dropout(p=0.5)
        self.linear5 = torch.nn.Linear(128, 10)
        


    def forward(self, points, scales, rotations, poses):
        points_embedding = self.embed(points)
        #print("points shape: ", points.shape)
        #print("poses shape: ", poses.shape)
        #for i, pose in enumerate(poses):
            #pose = pose.squeeze(0).repeat(points.shape[1], 1)
            #print("pose shape: ", pose.shape)
            #print("poses[i] shape: ", poses[i].shape)
            #poses[i] = pose.unsqueeze(0)
        #print("poses shape: ", poses.shape)
        #print("poses shape 1: ", points.shape[1])
        x1 = self.linear1(torch.cat([poses.unsqueeze(1).repeat(1, points.shape[1], 1), points_embedding], dim=-1))
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        x2 = self.linear2(x1)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        x3 = self.linear3(torch.cat([poses.unsqueeze(1).repeat(1, points.shape[1], 1), x2], dim=-1))
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        x4 = self.linear4(x3)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)
        offset = self.linear5(x4)

        points = points + offset[..., :3]
        scales = scales + offset[..., 3:6]
        rotations = rotations + offset[..., 6:]
        
        #print("points shape: ", points.shape)
        #print("scales shape: ", scales.shape)
        #print("rotations shape: ", rotations.shape)


        return points, scales, rotations, offset

