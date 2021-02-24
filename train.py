import os
from datetime import datetime
from sacd.agent import SacdAgent
from environment import Env


def run(args):
    name = 'sacd'
    config = args

    # Create environments.
    monitor = False
    env = Env(mode='train', monitor=monitor)
    test_env = Env(mode='test', monitor=monitor)
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'GammaRay', f'{name}-seed{0}-{time}')

    # Create the agent.
    Agent = SacdAgent
    agent = Agent(
        env=env, test_env=test_env, log_dir=log_dir,
        **config)
    agent.run()

    return agent.best_eval_score, log_dir


if __name__ == '__main__':

    num_steps = [36000]
    batch_size = [8]
    lr = [0.0005]
    gamma = [0.65]
    multi_step = [3]
    start_steps = [0]
    update_interval = [8]
    target_update_interval = [36000]

    best_score = -float('inf')
    for n_s in num_steps:
        for b_s in batch_size:
            for l_r in lr:
                for gam in gamma:
                    for m_s in multi_step:
                        for s_s in start_steps:
                            for u_i in update_interval:
                                for t_u_i in target_update_interval:
                                    args =  {'num_steps': n_s, 'batch_size': b_s, 'lr': l_r,
                                             'memory_size': 300000, 'gamma': gam, 'multi_step': m_s,
                                             'target_entropy_ratio': 0.98, 'start_steps': s_s, 'update_interval': u_i,
                                             'target_update_interval': t_u_i, 'use_per': True, 'dueling_net': False,
                                             'num_eval_steps': 300, 'max_episode_steps': 27000,
                                             'log_interval': 10, 'eval_interval': 6000, 'seed':0
                                             ,'cuda': True}
                                    score, log_name = run(args)
                                    if score > best_score:
                                        best_score = score
                                        f = open("sota.txt", "a")
                                        f.write(log_name + ' ')
                                        f.write(str(best_score) + '\n')
                                        f.write(str(args) + '\n')
                                        f.close()



