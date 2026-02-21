from game import Game
from agent import Agent
from pathlib import Path

game = Game(do_display=False)
agent = Agent(input_dim=11, output_dim=3)

num_episodes = 1000
max_steps = 500

checkpoints_folder = ("saved/checkpoints")
final_folder = Path("saved/final")

checkpoints_folder.mkdir(parents=True, exist_ok=True)
final_folder.mkdir(parents=True, exist_ok=True)

for episode in range(num_episodes):
    state = game.get_game_state()
    state_vector = state.vectorize() 
    done = False
    total_reward = 0

    for step in range(max_steps):

        # game.print_data()

        action = agent.act(state_vector)

        # apply action
        next_state, collected_apple, collided = game.update(action)
        next_state_vector = next_state.vectorize()

        reward = 1 if collected_apple else -1 if collided else 0
        total_reward += reward

        agent.store(state_vector, action, reward, next_state_vector, collided)

        agent.train_from_buffer(batch_size=64)

        state_vector = next_state_vector

        if collided:
            break

    print(f"Episode {episode+1}/{num_episodes} ({(episode+1)/num_episodes*100:.2f}%): Total reward = {total_reward}")
    game.reset()

    if (episode + 1) % 100 == 0:
        
        agent.save(checkpoints_folder / f"snake_dqn_ep{episode+1}.pth")

agent.save(final_folder / "snake_dqn_final.pth")