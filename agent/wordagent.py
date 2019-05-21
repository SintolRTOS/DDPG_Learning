# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:04:40 2019

@author: wangjingyi
"""

import sys 
sys.path.append("..") 

import numpy as np
np.set_printoptions(suppress=True)
from openpyxl import load_workbook
import logger
from .box import Box

POPULARITY_BOUND = 1000000


class WordAgent(object):
    """docstring for ClassName"""
    def __init__(self,
                 filepath,
                 mode):
        super(WordAgent,self).__init__()
        self.filepath = filepath
        self.mode = mode
        self.parameter_size = 5
        self.result = []
        self.result_keywords = []
        self.select_index = 0;
        self.select_count = 0;
        self.select_process = 0.
        self.select_total = 0
        self.select_total_reward = 0.
        self.max_action = 1.
        self.openExcel()
        self.reset()
        
    def openExcel(self):
        logger.debug('start openpyxl openExcel!')
        self.wb = load_workbook(self.filepath)
        logger.debug('form sheetname: ' + self.wb.sheetnames[0])
        self.keytitle = self.wb.sheetnames[0];
        sheet = self.wb.get_sheet_by_name(str(self.keytitle))
        self.max_row = sheet.max_row
        self.max_column = sheet.max_column
        logger.debug("form max_row: " + str(self.max_row))
        logger.debug("form max_column: " + str(self.max_column))
        self.data = []
        #wish data base
        for row in sheet.iter_rows(min_col=1, min_row=1, max_row=self.max_row, max_col=self.max_column):
            ROW = []
            if row[4].value == 0:
                continue;
            if row[5].value == 0:
                continue;
            if row[8].value == 0 and row[9].value == 0 and row[10].value == 0:
                continue;
            for cell in row:
                ROW.append(cell.value)
            self.data.append(ROW)
        logger.debug("totole data row: " + str(len(self.data)))
        logger.debug("all form data: ")
        logger.debug(self.data)
        
    def reset(self):
        self.init_observation()
        self.init_action_space()
        self.reward = 0.
        self.select_total_reward = 0.
        self.state = np.array([self.reward,self.select_total_reward])
        self.result.clear()
        self.result_keywords.clear()
        return self.observation
    
    def init_observation(self):
        logger.debug('observation-------------------------')
#        self.keywords_length = 10
        self.keywords_length = len(self.data)
        logger.debug('self.keywords_length: '+ str(self.keywords_length))
        self.observation = np.empty(self.keywords_length * self.parameter_size,float)
        popularity_max = self.data[0][4]
        popularity_min = self.data[0][4]
        conversion_max = self.data[0][5]
        conversion_min = self.data[0][5]
        transform_1_max = self.data[0][8]
        transform_1_min = self.data[0][8]
        transform_2_max = self.data[0][9]
        transform_2_min = self.data[0][9]
        transform_3_max = self.data[0][10]
        transform_3_min = self.data[0][10]
        for i in range(self.keywords_length):
            self.observation[i*self.parameter_size] = self.data[i][4]
            if self.observation[i*self.parameter_size] > popularity_max:
                popularity_max = self.observation[i*self.parameter_size]
            if self.observation[i*self.parameter_size] < popularity_min:
                popularity_min = self.observation[i*self.parameter_size]
                
            self.observation[i*self.parameter_size + 1] = self.data[i][5]
            if self.observation[i*self.parameter_size + 1] > conversion_max:
                conversion_max = self.observation[i*self.parameter_size + 1]
            if self.observation[i*self.parameter_size + 1] < conversion_min:
                conversion_min = self.observation[i*self.parameter_size + 1]
                
            self.observation[i*self.parameter_size + 2] = self.data[i][8]
            if self.observation[i*self.parameter_size + 2] > transform_1_max:
                transform_1_max = self.observation[i*self.parameter_size + 2]
            if self.observation[i*self.parameter_size + 2] < transform_1_min:
                transform_1_min = self.observation[i*self.parameter_size + 2]
                
            self.observation[i*self.parameter_size + 3] = self.data[i][9]
            if self.observation[i*self.parameter_size + 3] > transform_2_max:
                transform_2_max = self.observation[i*self.parameter_size + 3]
            if self.observation[i*self.parameter_size + 3] < transform_2_min:
                transform_2_min = self.observation[i*self.parameter_size + 3]
                
            self.observation[i*self.parameter_size + 4] = self.data[i][10]
            if self.observation[i*self.parameter_size + 4] > transform_3_max:
                transform_3_max = self.observation[i*self.parameter_size + 4]
            if self.observation[i*self.parameter_size + 4] < transform_3_min:
                transform_3_min = self.observation[i*self.parameter_size + 4]

        #normal action
        normal_check_popularity = 0.
        normal_check_conversion = 0.
        normal_check_transform_1 = 0.
        normal_check_transform_2 = 0.
        normal_check_transform_3 = 0.
        logger.debug('popularity_max:' + str(popularity_max))
        logger.debug('popularity_min:' + str(popularity_min))
        logger.debug('conversion_max:' + str(conversion_max))
        logger.debug('conversion_min:' + str(conversion_min))
        logger.debug('transform_1_max:' + str(transform_1_max))
        logger.debug('transform_1_min:' + str(transform_1_min))
        logger.debug('transform_2_max:' + str(transform_2_max))
        logger.debug('transform_2_min:' + str(transform_2_min))
        logger.debug('transform_3_max:' + str(transform_3_max))
        logger.debug('transform_3_min:' + str(transform_3_min))
        for j in range(self.keywords_length):
            self.observation[j*self.parameter_size] = (self.observation[j*self.parameter_size] - popularity_min) / (popularity_max - popularity_min)
            normal_check_popularity+=self.observation[j*self.parameter_size]
            
            self.observation[j*self.parameter_size + 1] = (self.observation[j*self.parameter_size + 1] - conversion_min) / (conversion_max - conversion_min)
            normal_check_conversion+=self.observation[j*self.parameter_size + 1]
            
            self.observation[j*self.parameter_size + 2] = (self.observation[j*self.parameter_size + 2] - transform_1_min) / (transform_1_max - transform_1_min)
            normal_check_transform_1+=self.observation[j*self.parameter_size + 2]
            
            self.observation[j*self.parameter_size + 3] = (self.observation[j*self.parameter_size + 3] - transform_2_min) / (transform_2_max - transform_2_min)
            normal_check_transform_2+=self.observation[j*self.parameter_size + 3]
            
            self.observation[j*self.parameter_size + 4] = (self.observation[j*self.parameter_size + 4] - transform_3_min) / (transform_3_max - transform_3_min)
            normal_check_transform_3+=self.observation[j*self.parameter_size + 4]
            
        
        nomal_check_total = normal_check_popularity + normal_check_conversion + normal_check_transform_1 + normal_check_transform_2 + normal_check_transform_3
        logger.debug('nomal_check_total:' + str(nomal_check_total))
        #add mean value
        normal_check_mean = nomal_check_total / self.observation.shape[0]
        logger.debug('normal_check_mean:' + str(normal_check_mean))
        normal_check_max = self.observation[0]
        normal_check_min = self.observation[0]
        for n in range(self.observation.shape[0]):
            self.observation[n] += normal_check_mean
#            logger.debug('self.observation[n]:' + str(self.observation[n]))
            if self.observation[n] > normal_check_max:
                normal_check_max = self.observation[n]
            if self.observation[n] <= normal_check_min:
                normal_check_min = self.observation[n]
        
        logger.debug('normal_check_max:' + str(normal_check_max))
        logger.debug('normal_check_min:' + str(normal_check_min))
        
        #check normal_action
        normal_check_last = 0.
        for x in range(self.observation.shape[0]):
#            logger.debug('self.observation[n]:' + str(self.observation[x]))
            self.observation[x] = (self.observation[x] - normal_check_min) / (normal_check_max - normal_check_min)
#            logger.debug('self.observation[n]_:' + str(self.observation[x]))
            normal_check_last += self.observation[x]
        logger.debug('normal_check_last:' + str(normal_check_last))
            
        
        logger.debug('print total observation--------------------------')
        for a in range(self.keywords_length):
            print_data = []
            print_data.append(self.observation[a*self.parameter_size])
            print_data.append(self.observation[a*self.parameter_size + 1])
            print_data.append(self.observation[a*self.parameter_size + 2])
            print_data.append(self.observation[a*self.parameter_size + 3])
            print_data.append(self.observation[a*self.parameter_size + 4])
            logger.debug(str(print_data))
        logger.debug('end total observation--------------------------')
            
            
            
#        logger.debug(str(self.observation))
        logger.debug('normal_check_popularity:' + str(normal_check_popularity))
        logger.debug('normal_check_conversion:' + str(normal_check_conversion))
        logger.debug('normal_check_transform_1:' + str(normal_check_transform_1))
        logger.debug('normal_check_transform_2:' + str(normal_check_transform_2))
        logger.debug('normal_check_transform_3:' + str(normal_check_transform_3))
        logger.debug('end_observation-------------------------')
        self.observation_space = Box(low = -self.observation,high = self.observation,dtype = np.float32)
        logger.debug('self.observation_space:')
        logger.debug(str(self.observation_space))
        return self.observation
    
    def get_observation(self):
        return self.observation
    
    def init_action_space(self):
        self.max_action = 1.
        self.action_space = Box(low = self.max_action,high = self.max_action,shape=(1,),dtype = np.float32)
        logger.debug('self.action_space:')
        logger.debug(str(self.action_space))
        logger.debug('self.action_space.shape[-1]:'+str(self.action_space.shape[-1]))
    
    def get_action_space(self):
        return self.action_space
    
    def get_result_keywords(self):
        logger.info('--------------get_result_keywords----------------')
        self.result_keywords.clear()
        count = len(self.result)
        self.result.sort()
        logger.debug('self.result: ' + str(self.result))
        logger.debug('result count: ' + str(count))
        for i in range(count):
            key_id = int(self.result[i])
            key_words = self.data[key_id][1]
            self.result_keywords.append(key_words)
            logger.debug('key_id: ' + str(key_id))
            logger.debug('keywords: ' + str(key_words))
        
        logger.debug('result keywords: ' + str(self.result_keywords))
        logger.debug('--------------end get_result_keywords----------------')
    
    def step(self,u):
        logger.debug('wordAgent select: ' + str(u))
        a_value =  float(u[0])
        pervalue = (a_value + 1.) / 2.
        logger.debug('wordAgent pervalue: ' + str(pervalue))
        index = int(pervalue * float(self.keywords_length - 1))
        logger.debug('--------------step----------------')
        logger.debug('wordAgent select index: ' + str(index))
        
        logger.debug('step self.observation index: ' + str(index))
        popularity = self.observation[index*self.parameter_size]
        logger.info('step popularity: ' + str(popularity))
        conversion = self.observation[index*self.parameter_size + 1]
        transform_1 = self.observation[index*self.parameter_size + 2]
        transform_2 = self.observation[index*self.parameter_size + 3]
        transform_3 = self.observation[index*self.parameter_size + 4]
        keywords_id = index
        #add mistake error value
#        self.observation[index*self.parameter_size] = -abs(popularity)
        self.observation[index*self.parameter_size] = -1.
        self.reward = popularity*conversion + popularity*transform_1*2 + popularity*transform_2 + popularity*transform_3
        if popularity == -1.:
            self.reward = -1
        logger.debug('wordAgent self.reward: ' + str(self.reward))
        self.result.append(int(keywords_id))
        self.select_total_reward += self.reward
        
        logger.debug('step self.observation[index*self.parameter_size]: ' + str(self.observation[index*self.parameter_size]))
        logger.debug('step self.observation[index*self.parameter_size + 1]: ' + str(self.observation[index*self.parameter_size + 1]))
        logger.debug('step self.observation[index*self.parameter_size + 2]: ' + str(self.observation[index*self.parameter_size + 2]))
        logger.debug('step self.observation[index*self.parameter_size + 3]: ' + str(self.observation[index*self.parameter_size + 3]))
        logger.debug('step self.observation[index*self.parameter_size + 4]: ' + str(self.observation[index*self.parameter_size + 4]))
#        logger.info('step self.observation[index*self.parameter_size + 5]: ' + str(self.observation[index*self.parameter_size + 5]))
        logger.debug('step self.observation result: ' + str(self.result))
        logger.debug('----------------step end---------------- ')
        self.state[0] = self.reward
        self.state[1] = self.select_total_reward
        return self.get_observation().copy(),self.reward,False,{}
    
    def render(self):
        logger.info('------------------------------------------')
        renderdic = {}
        renderdic['current_reward'] = self.reward
        renderdic['select_total_reward'] = self.select_total_reward
        renderdic['result_idlist'] = self.result
        renderdic['result_keywords'] = self.get_result_keywords()
        logger.info('')
        logger.info(str(renderdic))
        logger.info('')
        logger.info('------------------------------------------')
        logger.info('')
    
    def get_state(self):
        return self.state

#logger.debug('test openpyxl sucessful!')
#agent = WordAgent('../assert/keyword.xlsx','xlsx')
#agent.openExcel()
#agent.reset()

    
    
    