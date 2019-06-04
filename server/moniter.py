# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:15:21 2019

@author: wangjingyi
"""


import sys 
sys.path.append("..") 

import logger

import _thread
import threading
import datetime
import time
import os
from threading import Lock
mutex=Lock()

global NOISE_TYPE_WORD
NOISE_TYPE_WORD = '--noise_type=adaptive-param_0.2,normal_0.1'

global DDPG_RUN_STR
DDPG_RUN_STR = '--alg=ddpg --env=wordgame --play '

import subprocess    
    

class MoniterProcess(threading.Thread):
    def __init__(self, threadID, name, counter,process_id,reward_type,num_timesteps,log_file,model_file,assert_file):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.process_id = process_id
        self.reward_type = reward_type
        self.num_timesteps = num_timesteps
        self.log_file = log_file
        self.model_file = model_file
        self.run_process = 0.
        self.key_words_list = []
        self.assert_file = assert_file
        self.iscompleted = False
        self.isstarted = False
        self.os_id = 0
        
    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print('Starting thread' + self.name)
        self.start_run_process()
        print('Exiting thread' + self.name)
        
    def get_run_process_id(self):
        return self.process_id
    
    def get_current_process_info(self):
        return self.key_words_list,self.iscompleted,self.isstarted,self.run_process,self.os_id
    
    def start_run_process(self):
        if self.isstarted == True:
            return False
        self.isstarted = True
        RUN_STR = DDPG_RUN_STR
        RUN_STR += ('--num_timesteps=' + str(self.num_timesteps) + ' ')
        RUN_STR += ('--reward_type=' + str(self.reward_type) + ' ')
        RUN_STR += ('--log_file=' + str(self.log_file) + ' ')
        RUN_STR += ('--load_path=' + str(self.model_file) + ' ')
        RUN_STR += ('--save_path=' + str(self.model_file) + ' ')
        RUN_STR += ('--assert_file=' + str(self.assert_file) + ' ')
        RUN_STR += NOISE_TYPE_WORD
        argstr = 'python ../runner.py ' + RUN_STR
        logger.info('start_run_process: ')
        logger.info(argstr)
        try:
            _thread.start_new_thread( self.listener_log_info, (self.log_file,) )
            self.p = subprocess.run(argstr.split(' '))
        except Exception as err:     #进程和线程运行异常
            logger.error('log process_' + str(self.process_id) + ' MoniterProcess error:' + str(err))

        logger.info('complete_run_process:' + str(self.p))   
        self.iscompleted = True
    
    
    def listener_log_info(self,log_file):
        log_file += '/log.txt'
        logger.info('start_listener_log_info: ' + str(log_file))
        self.read_file = None
        self.pos = 0
        self.rank_end = False
        self.rank_start = False
        self.is_getrank = False
        self.is_get_per_process = False;
        while True:
            time.sleep(10)
            if self.iscompleted:
                break
            self.read_file = None
            self.pos = 0
            self.rank_end = False
            self.rank_start = False
            self.is_getrank = False
            self.is_get_per_process = False
            self.key_words_list.clear()
            mutex.acquire()
            if os.path.exists(log_file):
#                logger.info('open log_file:' + str(log_file))
                self.read_file = open(log_file, 'rb')
            else:
                mutex.release()
                continue
                
            self.pos = 0
            while True:
                line_str = self.lastline()
#                logger.info('line_str:' + str(line_str))
                if line_str == False:
                    break
                
                if self.is_get_per_process == False and line_str == 'pervalue':
                    self.is_get_per_process = True
                    continue
                
                if self.is_get_per_process == True:
                    self.is_get_per_process = False
                    self.run_process = line_str
#                    logger.info('current run_process:' + str(self.run_process))
                
                if line_str == 'print_rank_end' and not self.is_getrank:
                    self.rank_end = True
                    continue
                
                if line_str == 'print_rank_start' and self.rank_end and not self.is_getrank:
                    self.rank_start = True
                    continue
                    
                if self.rank_end and not self.rank_start and not self.is_getrank:
                    self.key_words_list.append(line_str)
                    
                if not self.is_getrank and self.rank_start:
                    self.is_getrank = True
#                    logger.info('current log listenr keywordslist:')
#                    logger.info(str(self.key_words_list))
                    self.read_file.close()
                    break
                
                if self.rank_start:
                    break
            
            mutex.release()
         
    def lastline(self):
        while True:
            self.pos = self.pos - 1
            try:
#                logger.info('seek pos:' + str(self.pos))
                self.read_file.seek(self.pos, 2)  #从文件末尾开始读
                w = self.read_file.read(1)
#                logger.info('seek w:' + str(w)) 
                if w == '\n'.encode():
                    break
            except:     #到达文件第一行，直接读取，退出
                self.read_file.seek(0, 0)        
                logger.error('log error:' + str(self.read_file.readline().strip()))
                return False
            
        last_str = self.read_file.readline().strip()
#        logger.info('last_str:' + str(last_str))
        return last_str.decode('gbk')

            
            



class Moniter(object):
    def __init__(self):
        super(Moniter,self).__init__()
        self.processdic = {}
        self.processcout = 0
        
    
    
    def run_process(self,process_id,reward_type,assert_file):
        if self.processdic.__contains__(process_id):
            return False
        
        datatime_form  = datetime.datetime.now().strftime("wordgame-%Y-%m-%d-%H-%M-%S-%f")
        model_file = './models/model_' + datatime_form + '_' + str(reward_type) + '_' + str(process_id)
        log_file = './logs/log_' + datatime_form + '_' + str(reward_type) + '_' + str(process_id)
        assert_file = '../assert/' + assert_file
        if os.path.exists(assert_file) == False:
            logger.info('assert_file is not exists:' + str(assert_file))
            return False
        
        # 创建新线程
        moniter_process = MoniterProcess(process_id,'moniter_' + str(process_id),reward_type,process_id,reward_type,'1e6',log_file,model_file,assert_file)
        self.processdic[process_id] = moniter_process
        moniter_process.start()
        return True
    
    def get_process(self,process_id):
        mutex.acquire()
        if self.processdic.__contains__(process_id):
            key_words_list,iscompleted,isstarted,run_process,os_id = self.processdic[process_id].get_current_process_info()
            retinfo = {}
            retinfo['key_words_list'] = key_words_list
            retinfo['iscompleted'] = iscompleted
            retinfo['isstarted'] = isstarted
            retinfo['run_process'] = run_process
            retinfo['os_id'] = os_id
            mutex.release()
            return retinfo  
        mutex.release()
        return None
    
    def end_process(self,process_id):
        return None
        
        
        
        