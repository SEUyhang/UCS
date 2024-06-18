from runner import Runner
from env_ucs.Env import EnvUCS
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import os
import torch
import warnings
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    for i in range(8):
        args = get_common_args()
        if args.alg.find('coma') > -1:
            args = get_coma_args(args)
        elif args.alg.find('central_v') > -1:
            args = get_centralv_args(args)
        elif args.alg.find('reinforce') > -1:
            args = get_reinforce_args(args)
        else:
            args = get_mixer_args(args)
        if args.alg.find('commnet') > -1:
            args = get_commnet_args(args)
        if args.alg.find('g2anet') > -1:
            args = get_g2anet_args(args)
        env = EnvUCS({   
        'test_mode': False,
        'save_path': '.',
        "controller_mode": True,
        "seed": 1,
        "action_mode": 0,
        "weighted_mode":True,
        "mip_mode":False,
        "noisy_power":-90,
        "tx_power":20,
        "render_mode":True,
        
        #
        "user_data_amount":1,
        "num_uav":3,
        "emergency_threshold":100,
        "collect_range":500,
        })
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        args.device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
