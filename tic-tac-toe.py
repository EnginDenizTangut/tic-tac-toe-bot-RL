import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  
        self.current_player = 'X'  

    def print_board(self):
        for i in range(0, 9, 3):
            print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
            if i < 6: print("---------")

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
            return True
        return False

    def check_winner(self):

        lines = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for a, b, c in lines:
            if self.board[a] != ' ' and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        return None if ' ' in self.board else 'Draw'

class QLearningRobot:
    def __init__(self):
        self.q_table = {}  
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  

    def get_state_key(self, board):
        return tuple(board)  

    def choose_action(self, board, available_actions):
        state_key = self.get_state_key(board)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(9)

        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            q_values = self.q_table[state_key]
            return max(available_actions, key=lambda a: q_values[a])

    def learn(self, board, action, reward, next_board):
        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(9)

        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + self.discount_factor * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.learning_rate * td_error

def train_robot(episodes=10000):
    game = TicTacToe()
    robot = QLearningRobot()
    results = {'X': 0, 'O': 0, 'Draw': 0}

    for _ in range(episodes):
        game.__init__()  
        while True:
            available_actions = [i for i, cell in enumerate(game.board) if cell == ' ']
            if not available_actions:
                break

            if game.current_player == 'X':
                action = np.random.choice(available_actions)  
            else:
                action = robot.choose_action(game.board, available_actions)

            prev_board = game.board.copy()
            game.make_move(action)
            winner = game.check_winner()

            if winner == 'O':
                reward = 1
                results['O'] += 1
            elif winner == 'X':
                reward = -1
                results['X'] += 1
            elif winner == 'Draw':
                reward = 0.5
                results['Draw'] += 1
            else:
                reward = 0

            if game.current_player == 'X':  
                robot.learn(prev_board, action, reward, game.board)

            if winner is not None:
                break

    print("Eğitim sonuçları:", results)
    return robot

def play_against_robot(robot):
    game = TicTacToe()
    print("Oyuna hoş geldiniz! Sen 'X', robot 'O' oynuyor.")

    while True:
        game.print_board()
        winner = game.check_winner()
        if winner is not None:
            print(f"Oyun bitti! Kazanan: {winner}")
            break

        if game.current_player == 'X':
            try:
                action = int(input("Hamlenizi girin (0-8): "))
                if not game.make_move(action):
                    print("Geçersiz hamle!")
            except ValueError:
                print("Lütfen 0-8 arası bir sayı girin.")
        else:
            available_actions = [i for i, cell in enumerate(game.board) if cell == ' ']
            action = robot.choose_action(game.board, available_actions)
            game.make_move(action)
            print(f"Robot hamlesi: {action}")

if __name__ == "__main__":
    print("Robot eğitiliyor...")
    trained_robot = train_robot(episodes=1000000)
    print("\nEğitim tamamlandı! İşte oyun zamanı:")
    play_against_robot(trained_robot)