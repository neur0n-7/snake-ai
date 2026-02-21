"""
  ________  _____  ___        __       __   ___  _______           __        __     
 /"       )(\"   \|"  \      /""\     |/"| /  ")/"     "|         /""\      |" \    
(:   \___/ |.\\   \    |    /    \    (: |/   /(: ______)        /    \     ||  |   
 \___  \   |: \.   \\  |   /' /\  \   |    __/  \/    |         /' /\  \    |:  |   
  __/  \\  |.  \    \. |  //  __'  \  (// _  \  // ___)_       //  __'  \   |.  |   
 /" \   :) |    \    \ | /   /  \\  \ |: | \  \(:      "|     /   /  \\  \  /\  |\  
(_______/   \___|\____\)(___/    \___)(__|  \__)\_______)    (___/    \___)(__\_|_) 
                                                                                    

Anish Gupta
https://github.com/neur0n-7

CLI tool to test a saved snake model.
Format: python testmodel.py [model path]

"""

import torch
from game import Game
from agent import Agent
import time
import sys
import pygame

DELAY = 0.05

def run(agent):
    game = Game(do_display=True)

    state = game.get_game_state()
    state_vector = state.vectorize()

    while True:

        done = False

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = agent.act(state_vector)

            next_state, collected_apple, collided = game.update(action)
            # game.print_data()

            state_vector = next_state.vectorize()

            done = collided

            time.sleep(DELAY)

        print(f"Game over, final score = {game.points}")
        print("Starting a new game in 3 seconds...")
        time.sleep(3)
        game.reset()


if len(sys.argv) != 2:
    print("Usage: python testmodel.py model.pth")
    sys.exit(1)


model_path = sys.argv[1]
agent = Agent(input_dim=11, output_dim=3)

agent.model.load_state_dict(torch.load(model_path))
agent.model.eval()

agent.epsilon = 0.0
print("Snake AI Model Testing")
print("Press CTRL+C at any time to stop the program.")

try:
    run(agent)
except KeyboardInterrupt:
    print("Keyboard interrupt detected. ")