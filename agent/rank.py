# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:10:44 2019

@author: wangjingyi
"""

MAX_SELECT_STEPS = 10

class KeyWordRank(object):

    def __init__(self):
        super(KeyWordRank,self).__init__()
        self.rank_list = []
        self.rank_value = []
        self.JUMP_INDEX = []
    
    def checkrank(self,rewardvalue,keylist):
#        logger.info('start self.keyword_rank_list: ' + str(self.rank_list))
        rankcout = len(self.rank_value)
        flag = False
        cur_index = 0
        new_rank_value = []
        new_keyword_rank_list = []
        for i in range(rankcout):
            if i >= MAX_SELECT_STEPS:
                break
            value = self.rank_value[i]
            
            if value == rewardvalue and str(keylist) == self.rank_list[i]:
                return
       
            if value < rewardvalue:
                cur_index = i
                flag = True
                break
    
        if flag:
            new_rank_value.clear()
            new_keyword_rank_list.clear()
            for j in range(rankcout):
                if j >= MAX_SELECT_STEPS:
                    break
                if j == cur_index:
                    new_rank_value.append(rewardvalue)
                    new_keyword_rank_list.append(str(keylist))
            
                if len(new_rank_value) < MAX_SELECT_STEPS:
                    new_rank_value.append(self.rank_value[j])
                    new_keyword_rank_list.append(self.rank_list[j])
            
#            logger.info('new_keyword_rank_list: ' + str(new_keyword_rank_list))
            self.rank_value.clear();
            self.rank_list.clear()
            self.rank_value = new_rank_value
            self.rank_list = new_keyword_rank_list
            
        else:
            if len(self.rank_value) < MAX_SELECT_STEPS:
                self.rank_value.append(rewardvalue)
                self.rank_list.append(str(keylist))
    
    def check_jump_index(self,index):
        for i in range(len(self.JUMP_INDEX)):
            value = self.JUMP_INDEX[i]
            if value == index:
                return True
        
        return False