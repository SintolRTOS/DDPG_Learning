# -*- coding: utf-8 -*-
"""
Created on Mon May 20 19:15:06 2019

@author: wangjingyi
"""

from  runner import runner_imp


NOISE_TYPE_POPART = '--normalize_returns=True --popart=True'
NOISE_TYPE_NORMAL = '--noise_type=normal_0.1'
NOISE_TYPE_OU = '--noise_type=ou_0.1'
NOISE_TYPE_ADAPTIVE = '--noise_type=adaptive-param_0.2,normal_0.1'
NOISE_TYPE_WORD = '--noise_type=adaptive-param_0.2,normal_0.1'

DDPG_RUN_STR = '--alg=ddpg --env=wordgame --num_timesteps=1e6 --save_path=./models/wordgame --play'
DDPG_RUN_STR2 = '--alg=ddpg --env=Pendulum-v0 --num_timesteps=1e6  --save_path=./models/Pendulum_0527 --load_path=./models/Pendulum_05247 --play '
DDPG_RUN_STR3 = '--alg=ddpg --env=wordgame --reward_type=4 --num_timesteps=1e6 --save_path=./models/wordgame_0531 --load_path=./models/wordgame_0531 --assert_file=./assert/collagen2019-05-28.xlsx --play '
DDPG_RUN_STR4 = '--alg=ppo2 --env=Pendulum-v0 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play '

def main():
    print('smoke main funtion')
    argstr = DDPG_RUN_STR3 + NOISE_TYPE_WORD
    print('smoketest ddpg_learning argstr: ' + argstr)

    ru = runner_imp()
    ru.run(argstr.split(' '))


if __name__ == '__main__':
    main()
    