# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:56:06 2019

@author: wangjingyi
"""
#高流量
COLLAGE_HIGHFLOW = 1
#高转化
COLLAGE_HIGHCONVERSION = 2
#高ROI
COLLAGE_HIGHROI = 3
#高转化+高ROI
COLLAGE_HIGHROI_HIGHCONVERSION = 4
    


def get_reward_value(observation,parameter_size,index,reward_type):
    reward = 0.
    if reward_type == COLLAGE_HIGHCONVERSION:
        popularity = observation[index*parameter_size]
        conversion = observation[index*parameter_size + 1]
        transform_1 = observation[index*parameter_size + 2]
        transform_2 = observation[index*parameter_size + 3]
        transform_3 = observation[index*parameter_size + 4]
        reward = popularity*conversion + popularity*transform_1*5 + popularity*transform_2 + popularity*transform_3
        
    elif reward_type == COLLAGE_HIGHFLOW:
        popularity = observation[index*parameter_size]
        click_num = observation[index*parameter_size + 1]
        click_rate = observation[index*parameter_size + 2]
        if click_rate == 0:
            reward = popularity
        else:
            reward = popularity + click_num / click_rate * 10
    elif reward_type == COLLAGE_HIGHROI:
        buy_num = observation[index*parameter_size]
        money_num = observation[index*parameter_size + 1]
        roi_rate = observation[index*parameter_size + 2]
        reward = buy_num + money_num*2  + roi_rate * 5
    
    elif reward_type == COLLAGE_HIGHROI_HIGHCONVERSION:
        popularity = observation[index*parameter_size]
        conversion = observation[index*parameter_size + 1]
        transform_1 = observation[index*parameter_size + 2]
        transform_2 = observation[index*parameter_size + 3]
        transform_3 = observation[index*parameter_size + 4]
        converation_value = popularity*conversion + popularity*transform_1*5 + popularity*transform_2 + popularity*transform_3
        
        buy_num = observation[index*parameter_size + 5]
        money_num = observation[index*parameter_size + 6]
        roi_rate = observation[index*parameter_size + 7]   
        roi_value = buy_num + money_num*2  + roi_rate * 5
        
        reward = converation_value + roi_value * 2.0
        
    return reward
