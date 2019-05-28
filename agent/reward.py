# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:56:06 2019

@author: wangjingyi
"""

from enum import Enum


class ProcessType(Enum):
    #胶原蛋白
    #高流量
    COLLAGE_HIGHFLOW = 1
    #高转化
    COLLAGE_HIGHCONVERSION = 2
    #高ROI
    COLLAGE_HIGHROI = 3
    #牡蛎
    #高流量
    OYSTER_HIGHFLOW = 4
    #高转化
    OYSTER_HIGHCONVERSION = 5
    #高ROI
    OYSTER_HIGHROI = 6
    


def get_reward_value(observation,parameter_size,index,reward_type):
    reward = 0.
    if reward_type == ProcessType.COLLAGE_HIGHCONVERSION.value:
        popularity = observation[index*parameter_size]
        conversion = observation[index*parameter_size + 1]
        transform_1 = observation[index*parameter_size + 2]
        transform_2 = observation[index*parameter_size + 3]
        transform_3 = observation[index*parameter_size + 4]
        reward = popularity*conversion + popularity*transform_1*5 + popularity*transform_2 + popularity*transform_3
    else:
        popularity = observation[index*parameter_size]
        conversion = observation[index*parameter_size + 1]
        transform_1 = observation[index*parameter_size + 2]
        transform_2 = observation[index*parameter_size + 3]
        transform_3 = observation[index*parameter_size + 4]
        reward = popularity*conversion + popularity*transform_1*5 + popularity*transform_2 + popularity*transform_3
    return reward
