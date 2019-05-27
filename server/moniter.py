# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:15:21 2019

@author: wangjingyi
"""


import sys 
sys.path.append("..") 

import logger
from enum import Enum

import datetime
import time

global NOISE_TYPE_WORD
NOISE_TYPE_WORD = '--noise_type=adaptive-param_0.2,normal_0.1'

global DDPG_RUN_STR
DDPG_RUN_STR = '--alg=ddpg --env=wordgame --play '

import subprocess


class ProcessType(Enum):
    #胶原蛋白
    COLLAGE_HIGHFLOW = 1
    COLLAGE_HIGHCONVERSION = 2
    COLLAGE_HIGHROI = 3
    #牡蛎
    OYSTER_HIGHFLOW = 4
    OYSTER_HIGHCONVERSION = 5
    OYSTER_HIGHROI = 6
    
    

class MoniterProcess(object):
    def __init__(self,process_id,reward_type,num_timesteps,log_file,model_file):
        super(MoniterProcess,self).__init__()
        self.process_id = process_id
        self.reward_type = reward_type
        self.num_timesteps = num_timesteps
        self.log_file = log_file
        self.model_file = model_file
        self.run_process = 0.
        self.key_words_list = []
        self.rank_reward_list = []
        self.iscompleted = False
        self.isstarted = False
        self.os_id = 0
        
    def get_run_process_id(self):
        return self.process_id
    
    def get_current_process_info(self):
        return self.rank_reward_list,self.key_words_list,self.iscompleted,self.isstarted,self.run_process,self.os_id
    
    def start_run_process(self):
        if self.isstarted == True:
            return False
        self.isstarted = True
        RUN_STR = DDPG_RUN_STR
        RUN_STR += ('--num_timesteps=' + self.num_timesteps + ' ')
        RUN_STR += ('--reward_type=' + self.reward_type + ' ')
        RUN_STR += ('--log_file=' + self.log_file + ' ')
        RUN_STR += ('--load_path=' + self.model_file + ' ')
        RUN_STR += ('--save_path=' + self.model_file + ' ')
        RUN_STR += NOISE_TYPE_WORD
        argstr = 'python -m DDPG_Learning.runner ' + RUN_STR
        logger.info('start_run_process: ')
        logger.info(RUN_STR)
        self.p = subprocess.Popen(argstr.split(' '),shell=True,stdout=subprocess.PIPE)
        self.os_id = self.p.pid
        while True:
            time.sleep(1000)
            if self.p.poll() == 0:
                break
            
        self.iscompleted = True
        



class Moniter(object):
    def __init__(self):
        super(Moniter,self).__init__()
        self.processdic = {}
        self.processcout = 0
        
    
    
    def run_process(self,process_id,reward_type):
        datatime_form  = datetime.datetime.now().strftime("sintolrtos-%Y-%m-%d-%H-%M-%S-%f")
        model_file = '../models/wordgame_' + datatime_form
        log_file = '../logs/log_' + datatime_form
        moniter_process = MoniterProcess(process_id,reward_type,'1e6',log_file,model_file)
        self.processdic[process_id] = moniter_process
        moniter_process.start_run_process()
    
    def get_process(self,process_id):
        if self.processdic.has_key(process_id):
            rank_reward_list, key_words_list,iscompleted,isstarted,run_process,os_id = self.processdic[process_id].get_current_process_info()
            retinfo = {}
            retinfo['rank_reward_list'] = rank_reward_list
            retinfo['key_words_list'] = key_words_list
            retinfo['iscompleted'] = iscompleted
            retinfo['isstarted'] = isstarted
            retinfo['run_process'] = run_process
            retinfo['os_id'] = os_id
            return retinfo     
        return None
    
    def end_process(self,process_id):
        return None
        
        
        
        