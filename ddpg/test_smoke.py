import sys 
sys.path.append("..") 

from common.tests.util import smoketest
def _run(argstr):
    smoketest('--alg=ddpg --env=Pendulum-v0 --num_timesteps=0 ' + argstr)

def test_popart():
    print('run test_popart')
    _run('--normalize_returns=True --popart=True --play')

def test_noise_normal():
    print('run test_noise_normal')
    _run('--noise_type=normal_0.1 --play')

def test_noise_ou():
    print('run test_noise_ou')
    _run('--noise_type=ou_0.1 --play')

def test_noise_adaptive():
    print('run test_noise_adaptive')
    _run('--noise_type=adaptive-param_0.2,normal_0.1 --play')


def main():
    print('this program is for testing smook')
    test_noise_ou()
    


if __name__ == '__main__':
    main()