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

Class representing the snake game logic

"""

import random
import pygame
import sys
from dataclasses import dataclass


# CONFIGS ################################################################

# This is in clockwise order to simplify things, 
# e.g. moving right is js incrementing the index by 1
DIRECTIONS = {
    "right" : (1, 0),
    "down": (0, 1),
    "left" : (-1, 0),
    "up" : (0, -1)
}

GRID_DIMENSIONS = (20, 20)
WINDOW_DIMENSIONS = (600, 600)

COLORS = {
    "EMPTY1" : (170, 215, 81),
    "EMPTY2" : (162, 209, 73),
    "SNAKE" : (71, 117, 235),
    "APPLE" : (231, 71, 29)
}


# GAMESTATE CLASS ########################################################
class GameState:
    def __init__(
            self,
            danger_straight, danger_left, danger_right,
            moving_up, moving_down, moving_left, moving_right,
            food_up, food_down, food_left, food_right
    ):
        self.danger_straight = danger_straight
        self.danger_left = danger_left
        self.danger_right = danger_right
        self.moving_up = moving_up
        self.moving_down = moving_down
        self.moving_left = moving_left
        self.moving_right = moving_right
        self.food_up = food_up
        self.food_down = food_down
        self.food_left = food_left
        self.food_right = food_right
    
    def vectorize(self):
        return (self.danger_straight, self.danger_left, self.danger_right, 
                self.moving_up, self.moving_down, self.moving_left, self.moving_right,
                self.food_up, self.food_down, self.food_left, self.food_right)


# GAME CLASS #############################################################

class Game:

    def __init__(self, do_display=False,):

        self.grid_width = GRID_DIMENSIONS[0]
        self.grid_height = GRID_DIMENSIONS[1]

        self.direction = DIRECTIONS["right"]

        self.snake = [(self.grid_width//2-1, self.grid_height//2)]
        self.food = (self.snake[0][0] + 3, self.snake[0][1])


        self.points = 0

        self.do_display = do_display

        if self.do_display:
            pygame.init()
            self.CELL_SIZE = min(WINDOW_DIMENSIONS[0] // GRID_DIMENSIONS[0], WINDOW_DIMENSIONS[1] // GRID_DIMENSIONS[1])
            self.screen = pygame.display.set_mode((self.CELL_SIZE*GRID_DIMENSIONS[0], self.CELL_SIZE*GRID_DIMENSIONS[1]))
            pygame.display.set_caption("neur0n-7 | Snake AI (0 points)")

        # matrix is (row, column) which is (y, x)
        self.grid = [[0]*self.grid_width for _ in range(self.grid_height)]
        self.grid[self.snake[0][1]][self.snake[0][0]] = 1
        self.grid[self.food[1]][self.food[0]] = 2

        
    def reset(self):
        self.points = 0
        self.snake = [(self.grid_width//2-1, self.grid_height//2-1)]
        self.food = (random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1))
        if self.do_display:
            self.grid = [[0]*self.grid_width for _ in range(self.grid_height)]


    def draw(self):
        assert self.screen

        # background
        self.screen.fill(COLORS["EMPTY1"])

        squares_to_draw = [(i, j) for i in range(self.grid_width) for j in range(self.grid_height) if i%2!=j%2]

        for (x, y) in squares_to_draw:
            pygame.draw.rect(
                self.screen,
                COLORS["EMPTY2"],
                (x*self.CELL_SIZE, y*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE),
                0
            )

        # snake
        for (x, y) in self.snake:
            pygame.draw.rect(
                self.screen,
                COLORS["SNAKE"],
                (x*self.CELL_SIZE, y*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            )

        # food
        fx, fy = self.food
        pygame.draw.rect(
            self.screen,
            COLORS["APPLE"],
            (fx*self.CELL_SIZE, fy*self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        )
        
        pygame.display.flip()
        
    def check_collided(self):
        return any(
            (
                self.snake[0][0] >= self.grid_width,
                self.snake[0][0] < 0,
                self.snake[0][1] >= self.grid_height,
                self.snake[0][1] < 0,
                len(set(self.snake)) != len(self.snake)
            )
        )

    def get_game_state(self):
        # danger straight/left/right
        dir_values = list(DIRECTIONS.values())
        dir_index = dir_values.index(self.direction)
        left_direction = dir_values[(dir_index - 1) % 4]
        right_direction = dir_values[(dir_index + 1) % 4]
        head_x, head_y = self.snake[0]
        head_pos_straight = (head_x + self.direction[0], head_y + self.direction[1])
        head_pos_left = (head_x + left_direction[0], head_y + left_direction[1])
        head_pos_right = (head_x + right_direction[0], head_y + right_direction[1])
        
        head_positions = (head_pos_straight, head_pos_left, head_pos_right)
        is_danger = []
        for pos in head_positions:

            out_of_bounds = any((
                pos[0] >= self.grid_width,
                pos[1] >= self.grid_height,
                pos[0] < 0,
                pos[1] < 0
            ))

            if out_of_bounds:
                is_danger.append(True)
            elif self.grid[pos[1]][pos[0]] == 1:
                is_danger.append(True)
            else:
                is_danger.append(False)

        danger_straight, danger_left, danger_right = is_danger

        # direction and food location
        moving_up = self.direction == DIRECTIONS["up"]
        moving_down = self.direction == DIRECTIONS["down"]
        moving_left = self.direction == DIRECTIONS["left"]
        moving_right = self.direction == DIRECTIONS["right"]

        food_up = self.food[1] < self.snake[0][1]
        food_down = self.food[1] > self.snake[0][1]
        food_left = self.food[0] < self.snake[0][0]
        food_right = self.food[0] > self.snake[0][0]
        
        return GameState(
            danger_straight,
            danger_left,
            danger_right,
            moving_up,
            moving_down,
            moving_left,
            moving_right,
            food_up,
            food_down,
            food_left,
            food_right
        )
        

    def update(self, action):
        # action: 0 = straight, 1 =left, 2 = right

        # update direction and the snake
        # okay maybe i'll stay away from one-liners for a bit
        # self.direction = DIRECTIONS[list(DIRECTIONS.keys())[(list(DIRECTIONS.values()).index(self.direction) - 1) % 4]] if action == 1 else DIRECTIONS[list(DIRECTIONS.keys())[(list(DIRECTIONS.values()).index(self.direction) + 1) % 4]]
        
        dir_values = list(DIRECTIONS.values())
        dir_index = dir_values.index(self.direction)

        if action == 1:
            self.direction = dir_values[(dir_index - 1) % 4]
        elif action == 2:
            self.direction = dir_values[(dir_index + 1) % 4]

        self.snake.insert(0, tuple(sum(pos) for pos in zip(self.snake[0], self.direction)))

        collided = self.check_collided()
        collected_apple = False

        if not collided:
            # update food
            if self.snake[0] == self.food:
                self.points += 1
                collected_apple = True
                self.food = random.choice(
                    [(x, y) for y in range(self.grid_height) for x in range(self.grid_width) if self.grid[y][x] == 0]
                )
        
        if not collected_apple:
            self.snake.pop(-1)
        
        if not collided:
            # update grid
            self.grid = [[0]*self.grid_width for _ in range(self.grid_height)]
            for (x, y) in self.snake:
                self.grid[y][x] = 1
            self.grid[self.food[1]][self.food[0]] = 2
        


        if self.do_display:
            self.draw()
            pygame.display.set_caption(f"neur0n-7 | Snake AI ({self.points} points)")


        return self.get_game_state(), collected_apple, collided


    def print_data(self):
        print(f"Snake: {self.snake}")
        print(f"Food: {self.food}")
        print(f"Direction: {self.direction}")
        print(f"Points: {self.points}")
