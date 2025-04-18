from logic.dudo_game import DudoGame, DudoAnalytics, RandomStrategy, NoisyProbabilisticStrategy, AdaptiveStrategy
from collections import Counter
import random



if __name__ == "__main__":
    
    num_players = 10
    # List of available strategy classes
    strategy_classes = [
        RandomStrategy,
        NoisyProbabilisticStrategy,
        AdaptiveStrategy
    ]

    # Assign a random strategy to each player
    strategies = [random.choice(strategy_classes)() for _ in range(num_players)]

    game = DudoGame(num_players=num_players, strategies=strategies)
    game = DudoGame(num_players=num_players, strategies=strategies)
    winner_idx, winner_strat, dice_left = game.play_game()
    print(f"Player {winner_idx} wins with strategy "
      f"{winner_strat.__class__.__name__} and has {dice_left} dice remaining.")

    analytics = DudoAnalytics(game)
    print("Call Summary:", analytics.summarize_calls())
    analytics.plot_call_summary()
    analytics.plot_probability_trend()
    analytics.plot_dice_evolution()

