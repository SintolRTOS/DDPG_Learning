# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:15:06 2019

@author: wangjingyi
"""

from  runner import runner_imp


NOISE_TYPE_POPART = '--normalize_returns=True --popart=True --play'
NOISE_TYPE_NORMAL = '--noise_type=normal_0.1 --play'
NOISE_TYPE_OU = '--noise_type=ou_0.1 --play'
NOISE_TYPE_ADAPTIVE = '--noise_type=adaptive-param_0.2,normal_0.1 --play'

DDPG_RUN_STR = '--alg=ddpg --env=wordgame --num_timesteps=1e6 '

def main():
    print('smoke main funtion')
    argstr = DDPG_RUN_STR + NOISE_TYPE_OU
    print('smoketest ddpg_learning argstr: ' + argstr)

    ru = runner_imp()
    ru.run(argstr.split(' '))


if __name__ == '__main__':
    main()
    