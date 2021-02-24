import os
from datetime import datetime
from sacd.agent import SacdAgent
from environment import Env


def run(args):
    name = 'sacd'
    config = args

    # Create environments.
    monitor = True
    test_env = Env(mode='test', monitor=monitor)

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', 'GammaRay-test')

    # Create the agent.
    Agent = SacdAgent
    agent = Agent(
        env=test_env, test_env=test_env, log_dir=log_dir,
        **config)
    agent.policy.my_load_state_dict('./logs/GammaRay/sacd-seed0-20210224-0406/model/best/policy.pth')

    total_return = 0
    num_episodes = 0
    average_episode_length = 0
    num_success = 0
    for i in range(config['num_steps']):
        episode_return = 0.
        episode_steps = 0
        done = False
        state = agent.test_env.reset()
        while (not done):
            action = agent.exploit(state)
            next_state, reward, done, info = agent.test_env.step(action)
            episode_steps +=1
            episode_return += reward
            state = next_state

        num_episodes += 1
        average_episode_length = episode_steps + average_episode_length
        total_return += episode_return
        print('num_test:', i)
        print('num_episodes:', num_episodes)

        if info == 'find' or info == 'over step but find':
            num_success = num_success + 1
        if num_episodes == config['num_steps']:
            break

    mean_return = total_return / num_episodes
    accuracy = num_success / num_episodes
    average_episode_length = average_episode_length / num_episodes
    print('mean_return: ', mean_return)
    print('accuracy: ', accuracy)
    print('average_episode_length: ', average_episode_length)




if __name__ == '__main__':

    num_steps = [3]
    batch_size = [8]
    lr = [0.0005]
    gamma = [0.7]
    multi_step = [3]
    start_steps = [0]
    update_interval = [4]
    target_update_interval = [40000]

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
                                             'log_interval': 10, 'eval_interval': 5000, 'seed':0
                                             ,'cuda': True}
                                    run(args)



