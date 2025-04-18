import random
import math
from collections import Counter
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import random
import math
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Abstract base class for defining strategies in DudoGame.

    Subclasses must implement the decide method.
    """
    @abstractmethod
    def decide(self, game, player):
        """
        Determine the next action for a player in the game.

        Args:
            game (DudoGame): Current game instance.
            player (int): Player index.

        Returns:
            tuple: 
              - ('bid', (qty, face)) to place a bid,
              - ('dudo', None) to call Dudo (challenge),
              - ('calzo', None) to call Calzo (exact).
        """
        pass


class RandomStrategy(BaseStrategy):
    """
    Strategy that makes random decisions:
      - 15% chance to call Dudo,
      - 5% chance to call Calzo,
      - otherwise places a random valid bid.
    """
    def decide(self, game, player):
        if game.current_bid is None:
            return ('bid', random.choice(game.get_valid_bids()))
        r = random.random()
        if r < 0.15:
            return ('dudo', None)
        if r < 0.20:
            return ('calzo', None)
        return ('bid', random.choice(game.get_valid_bids()))


class ProbabilisticStrategy(BaseStrategy):
    """
    Basic probability‐based strategy:
      - calls Dudo if P(last_bid true) < dudo_threshold,
      - calls Calzo if |P(last_bid true) − 0.5| < calzo_tol,
      - otherwise bids to move probability toward 50%.
    """
    def __init__(self, dudo_threshold=0.2, calzo_tol=0.05):
        """
        Args:
            dudo_threshold (float): probability below which to call Dudo.
            calzo_tol      (float): tolerance around 0.5 to call Calzo.
        """
        self.dudo_threshold = dudo_threshold
        self.calzo_tol = calzo_tol

    def decide(self, game, player):
        if game.current_bid is None:
            return ('bid', game._choose_bid())

        qty, face = game.current_bid
        p_true = game._prob_at_least(qty, face)

        if p_true < self.dudo_threshold:
            return ('dudo', None)
        if abs(p_true - 0.5) < self.calzo_tol:
            return ('calzo', None)
        return ('bid', game._choose_bid())


class RealisticProbabilisticStrategy(ProbabilisticStrategy):
    """
    Scales thresholds by fraction of dice remaining:
      - becomes more cautious (higher threshold) when many dice,
      - more conservative when few dice.
    """
    def decide(self, game, player):
        tot  = game.total_dice()
        frac = tot / game.initial_total

        # dynamic thresholds
        dudo_thr  = self.dudo_threshold * frac
        calzo_tol = self.calzo_tol      * frac

        if game.current_bid is None:
            return ('bid', game._choose_bid())

        qty, face = game.current_bid
        p_true = game._prob_at_least(qty, face)

        if p_true < dudo_thr:
            return ('dudo', None)
        if abs(p_true - 0.5) < calzo_tol:
            return ('calzo', None)
        return ('bid', game._choose_bid())


class ExtremeProbabilisticStrategy(ProbabilisticStrategy):
    """
    Like ProbabilisticStrategy, but:
      1. Immediately calls Dudo if qty > total dice (impossible bid).
      2. Otherwise uses dynamic thresholds (dice‐remaining scaling).
    """
    def decide(self, game, player):
        if game.current_bid is None:
            return ('bid', game._choose_bid())

        qty, face = game.current_bid
        total     = game.total_dice()

        # impossible bid -> guaranteed Dudo
        if qty > total:
            return ('dudo', None)

        # scale thresholds
        frac      = total / game.initial_total
        dudo_thr  = self.dudo_threshold * frac
        calzo_tol = self.calzo_tol      * frac

        p_true = game._prob_at_least(qty, face)
        if p_true < dudo_thr:
            return ('dudo', None)
        if abs(p_true - 0.5) < calzo_tol:
            return ('calzo', None)
        return ('bid', game._choose_bid())


class NoisyProbabilisticStrategy(ExtremeProbabilisticStrategy):
    """
    Adds random jitter to the Dudo threshold for unpredictability.
    """
    def __init__(self, dudo_threshold=0.2, calzo_tol=0.05, jitter=0.05):
        super().__init__(dudo_threshold=dudo_threshold, calzo_tol=calzo_tol)
        self.jitter = jitter

    def decide(self, game, player):
        if game.current_bid is None:
            return ('bid', game._choose_bid())

        qty, face = game.current_bid
        total     = game.total_dice()

        if qty > total:
            return ('dudo', None)

        frac      = total / game.initial_total
        base_thr  = self.dudo_threshold * frac
        calzo_tol = self.calzo_tol      * frac

        # uniform noise
        noise    = random.uniform(-self.jitter, self.jitter)
        dudo_thr = max(0.0, min(1.0, base_thr + noise))

        p_true = game._prob_at_least(qty, face)
        if p_true < dudo_thr:
            return ('dudo', None)
        if abs(p_true - 0.5) < calzo_tol:
            return ('calzo', None)
        return ('bid', game._choose_bid())


class AdaptiveStrategy(BaseStrategy):
    """
    Adjusts its Dudo threshold based on past wins/losses:
      - increases threshold when it loses,
      - decreases when others lose.
    """
    def __init__(self, initial_dudo_threshold=0.2, initial_calzo_tol=0.05, adapt_rate=0.02):
        self.dudo_threshold = initial_dudo_threshold
        self.calzo_tol      = initial_calzo_tol
        self.adapt_rate     = adapt_rate
        self.last_log_len   = 0

    def decide(self, game, player):
        # adapt threshold from new log entries
        logs = game.get_log()
        for r in logs[self.last_log_len:]:
            if r['loser'] == player:
                self.dudo_threshold = min(1.0, self.dudo_threshold + self.adapt_rate)
            elif r['call_type'] in ('dudo', 'calzo') and r['loser'] != player:
                self.dudo_threshold = max(0.0, self.dudo_threshold - self.adapt_rate)
        self.last_log_len = len(logs)

        # delegate actual decision to a fresh ProbabilisticStrategy
        strat = ProbabilisticStrategy(self.dudo_threshold, self.calzo_tol)
        return strat.decide(game, player)
    

class OpponentModelStrategy(BaseStrategy):
    """
    Strategy that builds a simple model of each opponent's bidding aggressiveness,
    then calls Dudo when the current bid exceeds what that opponent typically risks.

    - Tracks, for each player, the average qty they bid (as a fraction of total dice).
    - If the current bid's qty/total_dice is more than their historical average + margin,
    we call Dudo.
    - Otherwise, fall back to bidding toward the 50% probability point.
    """
    def __init__(self, margin: float = 0.1):
        """
        Args:
            margin (float): extra fraction above opponent's average to trigger Dudo.
                            (e.g. 0.1 means 10% above their norm)
        """
        self.margin = margin

    def decide(self, game, player):
        # 1) opening bid
        if game.current_bid is None:
            return ('bid', game._choose_bid())

        qty, face = game.current_bid
        total     = game.total_dice()
        last_bidder = game.last_bidder

        # 2) compute opponent's historical aggression
        logs = game.get_log()
        ratios = []
        for r in logs:
            for e in r['bids']:
                if e['player'] == last_bidder and 'bid' in e:
                    q, _ = e['bid']
                    ratios.append(q / r['initial_total'] if 'initial_total' in r else q / total)
        # if no history, fall back to balanced bidding
        if not ratios:
            return ('bid', game._choose_bid())

        avg_ratio = sum(ratios) / len(ratios)

        # 3) if current bid unusually high for that opponent → Dudo
        if (qty / total) > (avg_ratio + self.margin):
            return ('dudo', None)

        # 4) else check basic probability for Calzo
        p_true = game._prob_at_least(qty, face)
        if abs(p_true - 0.5) < 0.05:
            return ('calzo', None)

        # 5) otherwise raise toward 50%
        return ('bid', game._choose_bid())



class DudoGame:
    """
    Efficient simulator for Chilean Liar's Dice (Dudo) with pluggable strategies.

    Attributes:
        num_players (int): Number of players (2-10).
        strategies (list): One strategy instance per player.
        dice_counts (list): Current dice count per player.
        face_counts (list): Count of face values from last roll.
        log (list): Detailed log of each round.
    """
    def __init__(self, num_players: int, strategies=None):
        assert 2 <= num_players <= 10, "Players must be between 2 and 10"
        self.num_players   = num_players
        self.initial_dice  = [5] * num_players
        # total dice at start, used to enforce Calzo rule 
        self.initial_total = sum(self.initial_dice)
        self.strategies    = strategies or [ProbabilisticStrategy() for _ in range(num_players)]
        assert len(self.strategies) == num_players, "Strategy list length must equal players"
        self.reset()

    def reset(self):
        """Reset game state and precompute probability CDFs."""
        self.dice_counts     = self.initial_dice.copy()
        self.players         = list(range(self.num_players))
        self.current_bid     = None
        self.starting_player = 0
        self.last_bidder     = None
        self.log             = []
        self.round_number    = 0

        # Precompute CDFs for wild (1) and other faces
        max_total = self.initial_total
        self.cdf_wild = {}
        self.cdf_face = {}
        for T in range(1, max_total + 1):
            pmf1 = [math.comb(T, k)*(1/6)**k*(5/6)**(T-k) for k in range(T+1)]
            pmf2 = [math.comb(T, k)*(2/6)**k*(4/6)**(T-k) for k in range(T+1)]
            self.cdf_wild[T] = [sum(pmf1[k:]) for k in range(T+1)]
            self.cdf_face[T] = [sum(pmf2[k:]) for k in range(T+1)]

    def roll_dice(self):
        """Roll each player's dice and aggregate face counts."""
        self.face_counts = [0]*7
        self.rolls = []
        for p in self.players:
            for _ in range(self.dice_counts[p]):
                d = random.randint(1,6)
                self.rolls.append((p, d))
                self.face_counts[d] += 1

    def total_dice(self) -> int:
        """Total dice currently in play."""
        return sum(self.dice_counts)

    def get_valid_bids(self) -> list:
        """List all valid bids above the current bid according to Dudo rules."""
        tot = self.total_dice()
        bids = []
        if not self.current_bid:
            for q in range(1, tot+1):
                for f in range(2,7):
                    bids.append((q,f))
        else:
            q0, f0 = self.current_bid
            for q in range(q0+1, tot+1):
                bids.append((q, f0))
            for f in range(f0+1, 7):
                bids.append((q0, f))
            if f0 != 1:
                bids.append((math.ceil(q0/2), 1))
            else:
                for q in range(q0+1, tot+1):
                    bids.append((q, 1))
                dbl = q0*2 + 1
                if dbl <= tot:
                    for f in range(2,7):
                        bids.append((dbl, f))
        return bids

    def _prob_at_least(self, qty: int, face: int) -> float:
        """Probability at least qty dice match face (or wild)."""
        tot = self.total_dice()
        return self.cdf_wild[tot][qty] if face == 1 else self.cdf_face[tot][qty]

    def _choose_bid(self) -> tuple:
        """Select the bid whose probability is closest to 0.5."""
        return min(
            self.get_valid_bids(),
            key=lambda b: abs(self._prob_at_least(*b) - 0.5)
        )

    def resolve_challenge(self, challenger: int, bidder: int) -> int:
        """Resolve a Dudo call: loser loses one die."""
        qty, face = self.current_bid
        cnt = self.face_counts[face] + self.face_counts[1]
        return bidder if cnt < qty else challenger

    def resolve_calzo(self, challenger: int) -> int or None:
        """Resolve a Calzo call: exact guess gains a die, else loses one."""
        qty, face = self.current_bid
        cnt = self.face_counts[face] + self.face_counts[1]
        if cnt == qty:
            self.dice_counts[challenger] = min(5, self.dice_counts[challenger] + 1)
            return None
        self.dice_counts[challenger] -= 1
        return challenger

    def play_round(self) -> tuple:
        """Play one round: roll, bidding sequence, challenge resolution."""
        self.round_number += 1
        self.roll_dice()
        if self.starting_player not in self.players:
            self.starting_player = self.players[0]

        round_data = {
            'round':  self.round_number,
            'start':  self.starting_player,
            'initial': dict(zip(self.players, self.dice_counts)),
            'bids':   []
        }

        idx = self.players.index(self.starting_player)
        self.current_bid = None

        # track turns to force a Dudo if nobody ever challenges
        turns_without_challenge = 0
        max_turns = len(self.players)

        while True:
            p = self.players[idx]
            action, param = self.strategies[p].decide(self, p)

            # enforce Calzo legality
            if action == 'calzo' and self.total_dice() < self.initial_total / 2:
                action = 'bid'
                param  = self._choose_bid()

            entry = {'player': p}

            # ---- BID ----
            if action == 'bid':
                self.current_bid = param
                self.last_bidder = p
                entry['bid']    = param
                entry['p_true'] = self._prob_at_least(*param)
                round_data['bids'].append(entry)

                idx += 1
                idx %= len(self.players)
                turns_without_challenge += 1

            # ---- DUDO ----
            elif action == 'dudo':
                entry['call'] = 'dudo'
                round_data['bids'].append(entry)

                # resolve and subtract one die from loser
                loser     = self.resolve_challenge(p, self.last_bidder)
                self.dice_counts[loser] -= 1
                call_type = 'dudo'
                break

            # ---- CALZO ----
            else:  # action == 'calzo' (and legal here)
                entry['call'] = 'calzo'
                round_data['bids'].append(entry)

                loser     = self.resolve_calzo(p)
                call_type = 'calzo'
                break

            # force a Dudo if we've cycled through everyone with only bids
            if turns_without_challenge >= max_turns:
                ff = self.players[idx]
                forced = {'player': ff, 'call': 'dudo'}
                round_data['bids'].append(forced)

                loser     = self.resolve_challenge(ff, self.last_bidder)
                self.dice_counts[loser] -= 1
                call_type = 'dudo'
                break

        # finalize round data
        qty, face = self.current_bid or (None, None)
        actual    = None if qty is None else (self.face_counts[face] + self.face_counts[1])
        round_data.update({
            'call_type': call_type,
            'loser':      loser,
            'actual':     actual,
            'post':       dict(zip(self.players, self.dice_counts))
        })
        self.log.append(round_data)

        # eliminate player if out of dice
        if isinstance(loser, int) and self.dice_counts[loser] <= 0:
            self.players.remove(loser)

        # next round starts with loser if Dudo (and still in), else same bidder
        self.starting_player = (loser 
                                if call_type == 'dudo' and loser in self.players 
                                else p)
        return loser, call_type



    def play_game(self) -> tuple[int, BaseStrategy, int]:
        """
        Run full game until one player remains.

        Returns:
            tuple:
              - winner_index (int): index of the winning player
              - winner_strategy (BaseStrategy): the strategy instance they used
              - winner_dice (int): how many dice they have left at the end
        """
        self.reset()
        while len(self.players) > 1:
            self.play_round()

        winner = self.players[0]
        return winner, self.strategies[winner], self.dice_counts[winner]
    
    def get_log(self) -> list:
        """Retrieve the detailed log of all rounds."""
        return self.log



class DudoAnalytics:
    """
    Analytics and visualization for a completed DudoGame instance.
    """
    def __init__(self, game: DudoGame):
        self.log = game.get_log()
        self.num_players = game.num_players

    def summarize_calls(self) -> Counter:
        """Count occurrences of each call type ('dudo' or 'calzo')."""
        return Counter(r['call_type'] for r in self.log)

    def average_bid_probability(self) -> float:
        """Compute the average P(True) across all bids placed."""
        probs = [e['p_true'] for r in self.log for e in r['bids'] if 'p_true' in e]
        return sum(probs) / len(probs) if probs else 0.0

    def plot_call_summary(self):
        """Bar chart of call type frequencies."""
        data = self.summarize_calls()
        plt.figure()
        plt.bar(*zip(*data.items()))
        plt.title('Call Frequency')
        plt.xlabel('Call Type')
        plt.ylabel('Count')
        plt.show()

    def plot_probability_trend(self):
        """Line plot of P(True) for each bid event."""
        probs = [e['p_true'] for r in self.log for e in r['bids'] if 'p_true' in e]
        plt.figure()
        plt.plot(range(1, len(probs)+1), probs)
        plt.title('Bid True Probability Trend')
        plt.xlabel('Event')
        plt.ylabel('Probability')
        plt.show()

    def plot_dice_evolution(self):
        """Line chart of each player's dice count over rounds."""
        history = {p: [] for p in range(self.num_players)}
        for r in self.log:
            for p in range(self.num_players):
                history[p].append(r['post'].get(p, 0))
        plt.figure()
        for p, counts in history.items():
            plt.plot(counts, label=f'Player {p}')
        plt.title('Dice Count Over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Dice Count')
        plt.legend()
        plt.show()












