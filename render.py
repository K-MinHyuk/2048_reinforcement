import tkinter as tk
from tkinter import *
import numpy as np
import time

class Env_visualize(tk.Tk):
    def __init__(self, width =4, height =4):
        super(Env_visualize, self).__init__()
        self.FONT = ("Verdana", 40, "bold")
        ACTION_STRING = {0 : 'left', 1:'up' , 2:'right', 3:'down'}
        self.width = width
        self.height = height
        self.board = None
        self.score = 0
        self.title('2048')
        self.geometry('{0}x{1}'.format(700, 800))
        self.SIZE = 500
        self.GRID_LEN = 4
        self.GRID_PADDING = 10
        self.BACKGROUND_COLOR_GAME = "#92877d"
        self.BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
        self.BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
                                      32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
                                      512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}

        self.CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                                32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
                                512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}
        self.grid_cells = []
        self.init_grid()
        self.init_matrix()
        self.action_size = len(ACTION_STRING)
        

    def init_grid(self):
        self.background = tk.Frame(self, bg=self.BACKGROUND_COLOR_GAME, width=self.SIZE, height=self.SIZE)
        self.background.grid()
        for i in range(self.GRID_LEN):
            self.grid_row = []
            for j in range(self.GRID_LEN):
                self.cell = tk.Frame(self.background, bg=self.BACKGROUND_COLOR_CELL_EMPTY,
                                     width=self.SIZE / self.GRID_LEN, height=self.SIZE / self.GRID_LEN)
                self.cell.grid(row=i, column=j, padx=self.GRID_PADDING, pady=self.GRID_PADDING)
                # font = Font(size=FONT_SIZE, family=FONT_FAMILY, weight=FONT_WEIGHT)
                t = Label(master=self.cell, text="", bg=self.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=self.FONT,
                          width=4, height=2)
                t.grid()
                self.grid_row.append(t)

            self.grid_cells.append(self.grid_row)
        self.grid_row = []
        self.cell = tk.Frame(self.background, bg=self.BACKGROUND_COLOR_CELL_EMPTY, width=self.SIZE,
                             height=self.SIZE / self.GRID_LEN)
        self.cell.grid(row=4, columnspan=4, padx=self.GRID_PADDING, pady=self.GRID_PADDING, sticky=W + E + N + S)
        t = Label(master=self.cell, text="", bg=self.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=self.FONT, width=4,
                  height=2)
        t.grid(columnspan=4, ipadx=250)
        self.grid_row.append(t)
        self.grid_cells.append(self.grid_row)

    def init_matrix(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((self.width,self.height),dtype = np.int64)
        self.place_random_tiles(self.board,cnt=2)
        self.score = 0
        return self.board
        
    def render(self):
        time.sleep(0.01)
        self.update()
    
    def close(self):
        self.destroy()
        
    def canMove(self, action:int, state):
        state = state.reshape((4,4))
        rotated_obs = np.rot90(state, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        bord = np.rot90(updated_obs, k=4 - action).reshape(1,-1)
        return bord
    
    def canMove_t(self, action:int, state):
        state_ = state.to('cpu').numpy().copy()[0][0]
        state.to('cuda')
        rotated_obs = np.rot90(state_, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        bord = np.rot90(updated_obs, k=4 - action)
        return state_.reshape(-1,16), bord.reshape(-1,16)
    
    def step(self, action:int):
        rotated_obs = np.rot90(self.board, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        self.board = np.rot90(updated_obs, k=4 - action)
        
        self.place_random_tiles(self.board, cnt=1)
        done = self.is_done()
        self.score += reward
        for i in range(self.GRID_LEN):
            for j in range(self.GRID_LEN):
                new_number = self.board[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=self.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=self.BACKGROUND_COLOR_DICT[new_number], fg=self.CELL_COLOR_DICT[new_number])
        self.grid_cells[4][0].configure(text="Game Score : {}".format(self.score), bg=self.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER)
        if done:
            self.grid_cells[1][1].configure(text="You", bg=self.BACKGROUND_COLOR_CELL_EMPTY)
            self.grid_cells[1][2].configure(text="Lose!", bg=self.BACKGROUND_COLOR_CELL_EMPTY)
            
        return self.board, reward,done,{}
    
    def is_done(self):
        temp = self.board.copy()
        if not temp.all():
            return False
        
        for action in range(4):
            rotated_obs = np.rot90(temp,k=action)
            _,updated_obs = self.slide_left_and_merge(rotated_obs)
            if not updated_obs.all():
                return False
        
        return True
            
            
        
    def sample_tiles(self, cnt=1):

        choices = [2, 4]
        probs = [0.9, 0.1]

        tiles = np.random.choice(choices,size=cnt, p=probs)
        return tiles.tolist()

    def place_random_tiles(self,board,cnt =1):
        if not board.all():
            tiles = self.sample_tiles(cnt)
            tile_locations = self.sample_tile_locations(board, cnt)
            board[tuple(tile_locations)] = tiles
    
    def sample_tile_locations(self, board, cnt=1):
        zero_locations = np.argwhere(board==0)
        zero_index = np.random.choice(len(zero_locations) , size = cnt)
        
        zero_position = zero_locations[zero_index]
        zero_position = list(zip(*zero_position))
        
        return zero_position
    
    
    def slide_left_and_merge(self, board):
        result=[]
        score = 0
        for row in board:
            row = np.extract(row>0,row)
            score_, result_row = self.try_merge(row)
            
            score+= score_
            row = np.pad(np.array(result_row), (0, self.width - len(result_row)), 'constant', constant_values=(0,))
            result.append(row)
            
        return score, np.array(result, dtype=np.int64)
    
    @staticmethod
    def try_merge(row):
        score = 0
        result_row = []
        i=1
        while i <len(row):
            if row[i] == row[i-1]:
                score += row[i]*2
                result_row.append(row[i]*2)
                i+=2
            else:
                result_row.append(row[i-1])
                i+=1
        if i==len(row):
            result_row.append(row[i-1])
        return score, result_row
    

class Env_for_train():
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3

    ACTION_STRING = {LEFT : 'left', UP:'up' , RIGHT:'right', DOWN:'down'}

    def __init__(self, width =4, height =4):
        self.width = width
        self.height = height
        
        self.board = None
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.width,self.height),dtype = np.int64)
        self.place_random_tiles(self.board,cnt=2)
        return self.board
        
    def canMove(self, action:int, state):
        state = state.reshape((4,4))
        rotated_obs = np.rot90(state, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        bord = np.rot90(updated_obs, k=4 - action).reshape(1,-1)
        return bord
    
    def canMove_t(self, action:int, state):
        state_ = state.to('cpu').numpy().copy()[0][0]
        state.to('cuda')
        rotated_obs = np.rot90(state_, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        bord = np.rot90(updated_obs, k=4 - action)
        return state_.reshape(-1,16), bord.reshape(-1,16)
    
    def step(self,action:int):
        rotated_obs = np.rot90(self.board, k=action)
        reward, updated_obs = self.slide_left_and_merge(rotated_obs)
        self.board = np.rot90(updated_obs, k=4 - action)
        
        self.place_random_tiles(self.board, cnt=1)
        done = self.is_done()
        
        return self.board, reward,done,{}
    
    def is_done(self):
        temp = self.board.copy()
        if not temp.all():
            return False
        
        for action in range(4):
            rotated_obs = np.rot90(temp,k=action)
            _,updated_obs = self.slide_left_and_merge(rotated_obs)
            if not updated_obs.all():
                return False
        
        return True
            
            
        
    def sample_tiles(self, cnt=1):

        choices = [2, 4]
        probs = [0.9, 0.1]

        tiles = np.random.choice(choices,size=cnt, p=probs)
        return tiles.tolist()

    def place_random_tiles(self,board,cnt =1):
        if not board.all():
            tiles = self.sample_tiles(cnt)
            tile_locations = self.sample_tile_locations(board, cnt)
            board[tuple(tile_locations)] = tiles
    
    def sample_tile_locations(self, board, cnt=1):
        zero_locations = np.argwhere(board==0)
        zero_index = np.random.choice(len(zero_locations) , size = cnt)
        
        zero_position = zero_locations[zero_index]
        zero_position = list(zip(*zero_position))
        
        return zero_position
    
    
    def slide_left_and_merge(self, board):
        result=[]
        score = 0
        for row in board:
            row = np.extract(row>0,row)
            score_, result_row = self.try_merge(row)
            
            score+= score_
            row = np.pad(np.array(result_row), (0, self.width - len(result_row)), 'constant', constant_values=(0,))
            result.append(row)
            
        return score, np.array(result, dtype=np.int64)
    
    @staticmethod
    def try_merge(row):
        score = 0
        result_row = []
        i=1
        while i <len(row):
            if row[i] == row[i-1]:
                score += row[i]*2
                result_row.append(row[i]*2)
                i+=2
            else:
                result_row.append(row[i-1])
                i+=1
        if i==len(row):
            result_row.append(row[i-1])
        return score, result_row
    
    def render(self):
        return self.board