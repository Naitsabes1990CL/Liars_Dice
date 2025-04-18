

import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind

from logic.dudo_game import (
    DudoGame,
    DudoAnalytics,
    RandomStrategy,
    NoisyProbabilisticStrategy,
    AdaptiveStrategy,
    OpponentModelStrategy
)


def pairwise_win_rate_tests(df: pd.DataFrame,
                            strategy_col: str = 'Winner Strategy',
                            total_games: int = 500) -> pd.DataFrame:
    """
    Perform pairwise two-proportion z‑tests on win counts between strategies,
    and apply Bonferroni correction to p-values.
    """
    win_counts = df[strategy_col].value_counts().to_dict()
    strategies = list(win_counts.keys())

    records = []
    for a, b in combinations(strategies, 2):
        count = np.array([win_counts[a], win_counts[b]])
        nobs  = np.array([total_games, total_games])
        zstat, pval = proportions_ztest(count, nobs)
        records.append({
            'Strategy A': a,
            'Strategy B': b,
            'z_stat': zstat,
            'p_value': pval
        })

    result = pd.DataFrame(records)
    m = len(result)
    result['p_adjusted'] = np.minimum(result['p_value'] * m, 1.0)
    return result


def pairwise_metric_tests(df: pd.DataFrame,
                          metric_col: str,
                          strategy_col: str = 'Winner Strategy') -> pd.DataFrame:
    """
    Perform pairwise Welch's t‑tests on a continuous metric between strategies,
    and apply Bonferroni correction to p-values.
    """
    strategies = df[strategy_col].unique()
    records = []
    for a, b in combinations(strategies, 2):
        vals_a = df.loc[df[strategy_col] == a, metric_col]
        vals_b = df.loc[df[strategy_col] == b, metric_col]
        tstat, pval = ttest_ind(vals_a, vals_b, equal_var=False)
        records.append({
            'Strategy A': a,
            'Strategy B': b,
            't_stat': tstat,
            'p_value': pval,
            'Metric': metric_col
        })

    result = pd.DataFrame(records)
    m = len(result)
    result['p_adjusted'] = np.minimum(result['p_value'] * m, 1.0)
    return result


def simulate_and_visualize(
    num_games: int = 500,
    num_players: int = 4
):
    """
    Runs num_games of Dudo with num_players, collects metrics,
    prints a summary, performs pairwise hypothesis tests with Bonferroni correction,
    and shows four performance plots.
    """
    strategy_factories = [
        lambda: RandomStrategy(),
        lambda: NoisyProbabilisticStrategy(),
        lambda: OpponentModelStrategy(margin=0.1),
        lambda: AdaptiveStrategy()
    ]

    records = []
    for game_ix in range(1, num_games + 1):
        strategies = [random.choice(strategy_factories)() for _ in range(num_players)]
        game = DudoGame(num_players=num_players, strategies=strategies)
        winner_idx, winner_strat, dice_left = game.play_game()
        rounds_played = game.round_number

        dudo_calls = calzo_calls = 0
        bid_sizes = []
        for r in game.get_log():
            for e in r['bids']:
                if e.get('call') == 'dudo':
                    dudo_calls += 1
                elif e.get('call') == 'calzo':
                    calzo_calls += 1
                elif 'bid' in e:
                    bid_sizes.append(e['bid'][0])

        avg_bid_size = (sum(bid_sizes) / len(bid_sizes)) if bid_sizes else 0.0

        records.append({
            'Game':            game_ix,
            'Winner Strategy': winner_strat.__class__.__name__,
            'Dice Left':       dice_left,
            'Rounds Played':   rounds_played,
            'Dudo Calls':      dudo_calls,
            'Calzo Calls':     calzo_calls,
            'Avg Bid Size':    avg_bid_size
        })

    df = pd.DataFrame(records)

    # aggregate with means and standard errors
    summary = df.groupby('Winner Strategy').agg(
        Wins                = ('Winner Strategy', 'count'),
        Average_Dice_Left   = ('Dice Left', 'mean'),
        SE_Dice_Left        = ('Dice Left', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        Average_Rounds      = ('Rounds Played', 'mean'),
        SE_Rounds           = ('Rounds Played', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        Total_Dudo_Calls    = ('Dudo Calls', 'sum'),
        Total_Calzo_Calls   = ('Calzo Calls', 'sum'),
        Avg_Bid_Size        = ('Avg Bid Size', 'mean'),
        SE_Avg_Bid_Size     = ('Avg Bid Size', lambda x: x.std(ddof=1)/np.sqrt(len(x)))
    ).reset_index()

    print("\n=== Summary by Strategy ===")
    print(summary.to_string(index=False))

    # Pairwise hypothesis tests
    win_tests     = pairwise_win_rate_tests(df, total_games=num_games)
    dice_tests    = pairwise_metric_tests(df, 'Dice Left')
    rounds_tests  = pairwise_metric_tests(df, 'Rounds Played')
    bidsize_tests = pairwise_metric_tests(df, 'Avg Bid Size')

    print("\n=== Pairwise Win Rate Z-Tests (Bonferroni corrected) ===")
    print(win_tests.to_string(index=False))

    print("\n=== Pairwise Dice Left T-Tests (Bonferroni corrected) ===")
    print(dice_tests.to_string(index=False))

    print("\n=== Pairwise Rounds Played T-Tests (Bonferroni corrected) ===")
    print(rounds_tests.to_string(index=False))

    print("\n=== Pairwise Avg Bid Size T-Tests (Bonferroni corrected) ===")
    print(bidsize_tests.to_string(index=False))

    # 1) Win Counts
    plt.figure()
    plt.bar(summary['Winner Strategy'], summary['Wins'])
    plt.title('Win Counts by Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 2) Average Dice Left ±SE
    plt.figure()
    plt.errorbar(
        summary['Winner Strategy'],
        summary['Average_Dice_Left'],
        yerr=summary['SE_Dice_Left'],
        fmt='o',
        capsize=5
    )
    plt.title('Average Dice Left by Strategy (±SE)')
    plt.xlabel('Strategy')
    plt.ylabel('Average Dice Left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 3) Average Rounds ±SE
    plt.figure()
    plt.errorbar(
        summary['Winner Strategy'],
        summary['Average_Rounds'],
        yerr=summary['SE_Rounds'],
        fmt='o',
        capsize=5
    )
    plt.title('Average Rounds by Strategy (±SE)')
    plt.xlabel('Strategy')
    plt.ylabel('Average Number of Rounds')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # 4) Average Bid Size ±SE
    plt.figure()
    plt.errorbar(
        summary['Winner Strategy'],
        summary['Avg_Bid_Size'],
        yerr=summary['SE_Avg_Bid_Size'],
        fmt='o',
        capsize=5
    )
    plt.title('Average Bid Size by Strategy (±SE)')
    plt.xlabel('Strategy')
    plt.ylabel('Average Bid Quantity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_and_visualize(num_games=500, num_players=10)

