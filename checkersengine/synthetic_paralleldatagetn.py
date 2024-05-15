from checkers.game import Game
import numpy as np
import random
from tqdm import tqdm
from threading import Thread, Lock

lock = Lock() 

def possible_capture_move(move):
    x = move[0]
    y = move[1]
    if x<y and (((x-1) // 4) + 2) == ((y-1) //4):
        return True
    elif ((x-1) // 4)  == (((y-1) //4) +2):
        return True
    else:
        return False


def synthetic_games(game, moves):
    
    game_string = ";"
    for i in range(1, moves+1):
        game_string +=  str(i) + ". "
        for _ in range(2):
            possible_moves = game.get_possible_moves()
            random_index = random.randint(0, len(possible_moves) - 1)
            random_element = possible_moves[random_index]
            x = random_element[0]
            y = random_element[1]
            game.move([x, y])
            if game.is_over():
                return game_string
            if possible_capture_move(random_element):
                game_string+= str(x) + 'x' + str(y)
                double_capture = game.get_possible_moves()
                if (len(double_capture) != 1 or double_capture[0][0] != y):
                    game_string+=" "
                else:
                    while (len(double_capture) == 1 and double_capture[0][0] == y):
                        game.move([double_capture[0][0],double_capture[0][1]])
                        game_string+= "x"+str(double_capture[0][1])
                        y = double_capture[0][1]
                        double_capture = game.get_possible_moves()
                        if game.is_over():
                            return game_string+" "
                    game_string+=" "  
            else:
                game_string+= str(x) + '-' + str(y) + " "
    return game_string


def generate_data(file, num_games):
    for _ in tqdm(range(num_games)):
        game = Game()
        game_data = synthetic_games(game, 20)
        with lock:
            file.write(game_data + '\n')



def main():
    num_games = 16000000
    num_threads = 1000
    num_games_per_thread = int(num_games / num_threads)  # Total 16,000,000 games, divided equally among threads
    with open('synthetic_dataparallel16M.txt', 'w') as file:
        threads = []
        for _ in range(num_threads):
            thread = Thread(target=generate_data, args=(file, num_games_per_thread))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()  # Wait for all threads to finish before closing the file

if __name__ == "__main__":
    main()