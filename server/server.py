# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:59:26 2019

@author: wangjingyi
"""


import sys 
sys.path.append("..") 


import time
import http.server
import json
import logger

HOST_NAME = '127.0.0.1' 
PORT_NUMBER = 9090
CONIFRM_PATH = '/tmp'
global PROCESS_ID
PROCESS_ID = 0
from moniter import Moniter
global moniterimp
moniterimp = Moniter()

from enum import Enum

class ActionType(Enum):
    #启动操作进程
    START_ACTION_PROCESS = 1
    #关闭操作进程
    END_ACTION_PROCESS = 2
    #查询进程结果
    QUERY_ACTION_PROCESS = 3

class DDPG_Server(http.server.BaseHTTPRequestHandler):    
    

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type','application/json')
        self.end_headers()
    
        
    def _post_handler(self,data):
        try:
            global PROCESS_ID
            json_objects = json.loads(str(data))
            logger.info('_post_handler:')
            logger.info(json_objects)
            starttime = time.time()
            localtime = time.localtime()
            logdir = 'sintolrtos_' + str(starttime)
            ret_code = 0
            id = json_objects['id']
            action_id = json_objects['action_id']
            num_timesteps = json_objects['num_timesteps']
            reward_type = json_objects['reward_type']
            action_ret = self.handler(action_id,json_objects,reward_type)
            if action_ret['retcode'] == -1:
                ret_code = -1
            json_ret = {
                    'id' : id,
                    'action_id' : action_id,
                    'retinfo' : action_ret['retinfo'],
                    'retcode' : ret_code,
                    'process_id' : PROCESS_ID,
                    'reward_type' : reward_type,
                    'actiontime': str(time.strftime("%Y-%m-%d %H:%M:%S",localtime)),
                    'logdir' : logdir,
                    'num_timesteps' : num_timesteps
                    }
        except Exception as e:
            logger.info('_post_handler except:', e)
            json_ret = {
                    'retcode' : 1,
                    'errormsg' : str(e)
                    }
        PROCESS_ID += 1
        return json.dumps(json_ret)
    
    
    def handler(self,action_id,json_data,reward_type = None,process_id = None):
        
        global PROCESS_ID
        action_ret = {}
        action_ret['retcode'] = 0
        action_ret['retinfo'] = None
        if action_id == int(ActionType.START_ACTION_PROCESS.value):
            assert_file = json_data['assert_file']
            if moniterimp.run_process(PROCESS_ID,reward_type,assert_file=assert_file) == False:
                action_ret['retcode'] = -1
        elif action_id == int(ActionType.QUERY_ACTION_PROCESS.value):
            retinfo = moniterimp.get_process(process_id)
            if retinfo is None:
                action_ret['retcode'] = -1
            action_ret['retinfo'] = retinfo
        elif action_id == int(ActionType.END_ACTION_PROCESS.value):
            retinfo = moniterimp.end_process(process_id)
            if retinfo is None:
                action_ret['retcode'] = -1
            action_ret['retinfo'] = retinfo
        return action_ret
    
    def do_HEAD(self):
        self._set_headers()
    
    def do_GET(self):
        self._set_headers()
        #get request params
#        path = self.path
#        query = urllib.splitquery(path)
#        self._get_handler(query[1]);
        
    def do_POST(self):
        self._set_headers()
        #get post data
        length = self.headers['content-length'];
        nbytes = int(length)
        post_data = self.rfile.read(nbytes) 
        post_str = post_data.decode(encoding='utf-8')
        jsonobj_ret = self._post_handler(post_str)
        self.wfile.write(jsonobj_ret.encode())

if __name__ == '__main__':
    server_class = http.server.HTTPServer
    server_address = (HOST_NAME,PORT_NUMBER)
    httpd = server_class(server_address, DDPG_Server)
    logger.info(str(time.asctime()) + ' Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logger.info(str(time.asctime()), ' Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))
        
        
        
        
        