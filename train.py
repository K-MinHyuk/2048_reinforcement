import numpy as np
import torch
import pylab

from DQN_Agent import DQNAgent
if __name__ == '__main__':
    model_path = '2nd_512_model'
    visualize = True
    if visualize:
        from render import Env_visualize as Env
    else:
        from render import Env_for_train as Env
    env = Env()
    state_size = 16
    action_size = 4
    load_model = False
    agent = DQNAgent(state_size, action_size, load_model, model_path, visualize)
    Top = -np.inf
    TopScore = -np.inf
    scores, episodes = [], []
    loss_list = []
    EPISODES=5000
    for e in range(EPISODES):
        done=False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        maxScore = -np.inf
        move = 0
        while not done:
            env.render()
            move += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state, env)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.append_sample(state, action, reward, next_state, done)
            if agent.load_model is False and (len(agent.memory) >= agent.train_start):
                agent.train_model()
                
            score += reward
            state = next_state
            if(reward > maxScore):
                maxScore = reward
            if done:
                # every episode update the target model to be same with model
                if agent.load_model is False:
                    agent.update_target_model()
                loss_list.append(agent.train_loss/move)
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, " move:", move,"  memory length:", len(agent.memory), "  epsilon:", agent.epsilon, ' loss:', agent.train_loss/move)
                if (maxScore > Top) and not (load_model):
                    Top = maxScore
                    print('Weights save... Top reward = {}'.format(maxScore))
                    torch.save(agent.model.state_dict(), './save_model/topReward.pth')
                if (score > TopScore) and not (load_model):
                    TopScore = score
                    print('Weights save... Top score = {}'.format(TopScore))
                    torch.save(agent.model.state_dict(), './save_model/topScore.pth')
    print('Weights save... last_model')
    torch.save(agent.model.state_dict(), './save_model/last_model.pth')
    pylab.plot(episodes, scores, 'b')
    pylab.title('Score per episode')
    pylab.savefig('save_plt/Learning_rate={},epsilon_min={},Discount_factor={},episod={}.png'.format(agent.learning_rate, agent.epsilon_min, agent.discount_factor, e+1), dpi=300, facecolor='w')
         