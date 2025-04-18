from logic.dudo_game import DudoGame
from collections import Counter


if __name__ == "__main__":
    # Example: simulate 100 games of 5 players
    results = Counter()
    for _ in range(100):
        game = DudoGame(num_players=5)
        winner = game.play_game()
        results[winner] += 1
    print("Win counts:", results)