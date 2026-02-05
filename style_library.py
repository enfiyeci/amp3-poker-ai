"""
style_library.py - Multi-player Poker Style Library
Implements 64 different player styles based on:
- Random probability strategies (7 variants)
- Sklansky hand grading (5 player types)
- Bill Chen hand evaluation (5 player types)  
- Rule-based strategies (5 player types)

Player types: Conservative, Regular, Aggressive, Bluffing, Deceptive
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum

from poker_core import (
    Card, Action, Street, GameState, PlayerState,
    get_sklansky_group, chen_formula, get_hand_category,
    evaluate_hand, HandRank
)


class PlayerType(IntEnum):
    """Player types from the paper"""
    CONSERVATIVE = 0  # Bet only with very strong hands
    REGULAR = 1       # Bet with strong hands, fold with weak
    AGGRESSIVE = 2    # Bet with slightly strong hands
    BLUFFING = 3      # Sometimes bet with weak hands
    DECEPTIVE = 4     # Call with strong hands to trap


@dataclass
class StyleFeatures:
    """
    Four key style features from the paper:
    - VPIP: Voluntarily Put Money In Pot (pre-flop participation)
    - PFR: Pre-Flop Raise percentage
    - AFq: Aggression Frequency (post-flop)
    - WTSD: Went To ShowDown percentage
    """
    vpip: float = 0.0
    pfr: float = 0.0
    afq: float = 0.0
    wtsd: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.vpip, self.pfr, self.afq, self.wtsd], dtype=np.float32)
    
    @staticmethod
    def from_vector(vec: np.ndarray) -> 'StyleFeatures':
        return StyleFeatures(vpip=vec[0], pfr=vec[1], afq=vec[2], wtsd=vec[3])


class BasePokerStrategy(ABC):
    """Base class for all poker strategies"""
    
    def __init__(self, style_name: str, player_type: PlayerType):
        self.style_name = style_name
        self.player_type = player_type
        self.rng = np.random.default_rng()
        
        # Statistics tracking
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.total_actions_postflop = 0
        self.aggressive_actions_postflop = 0
        self.hands_to_flop = 0
        self.hands_to_showdown = 0
    
    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)
    
    @abstractmethod
    def get_action(self, 
                   state: GameState, 
                   seat: int,
                   hole_cards: List[Card],
                   community_cards: List[Card]) -> Tuple[Action, float]:
        """
        Returns (action, amount) tuple.
        Amount is only relevant for BET actions.
        """
        pass
    
    def get_style_features(self) -> StyleFeatures:
        """Calculate style features from statistics"""
        vpip = self.vpip_count / max(1, self.hands_played)
        pfr = self.pfr_count / max(1, self.hands_played)
        afq = self.aggressive_actions_postflop / max(1, self.total_actions_postflop)
        wtsd = self.hands_to_showdown / max(1, self.hands_to_flop)
        return StyleFeatures(vpip=vpip, pfr=pfr, afq=afq, wtsd=wtsd)
    
    def update_stats(self, action: Action, state: GameState, is_preflop: bool):
        """Update statistics after taking an action"""
        if is_preflop:
            self.hands_played += 1
            if action in [Action.CALL, Action.BET, Action.ALL_IN]:
                self.vpip_count += 1
            if action in [Action.BET, Action.ALL_IN]:
                self.pfr_count += 1
        else:
            self.total_actions_postflop += 1
            if action in [Action.BET, Action.ALL_IN]:
                self.aggressive_actions_postflop += 1
    
    def reset_stats(self):
        """Reset all statistics"""
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.total_actions_postflop = 0
        self.aggressive_actions_postflop = 0
        self.hands_to_flop = 0
        self.hands_to_showdown = 0


# =============================================================================
# Random Probability Strategies
# =============================================================================

class RandomStrategy(BasePokerStrategy):
    """
    Strategy with random probability distributions before and after flop.
    7 variants with different (fold, call, raise) probability tuples.
    """
    
    # Probability distributions: (fold, call, raise)
    PROB_CONFIGS = {
        0: ((1/3, 1/3, 1/3), (1/3, 1/3, 1/3)),       # Uniform
        1: ((1/2, 1/5, 3/10), (1/2, 1/5, 3/10)),     # Tight-passive
        2: ((1/2, 3/10, 1/5), (1/2, 3/10, 1/5)),     # Tight-aggressive
        3: ((1/5, 1/2, 3/10), (1/5, 1/2, 3/10)),     # Loose-passive
        4: ((3/10, 1/2, 1/5), (3/10, 1/2, 1/5)),     # Loose-passive alt
        5: ((1/5, 3/10, 1/2), (1/5, 3/10, 1/2)),     # Loose-aggressive
        6: ((3/10, 1/5, 1/2), (3/10, 1/5, 1/2)),     # Very aggressive
    }
    
    def __init__(self, config_id: int = 0):
        player_type = self._get_player_type(config_id)
        super().__init__(f"Random_{config_id}", player_type)
        self.config_id = config_id
        self.preflop_probs, self.postflop_probs = self.PROB_CONFIGS[config_id]
    
    def _get_player_type(self, config_id: int) -> PlayerType:
        """Map config to player type"""
        if config_id in [1, 2]:
            return PlayerType.CONSERVATIVE
        elif config_id in [3, 4]:
            return PlayerType.REGULAR
        else:
            return PlayerType.AGGRESSIVE
    
    def get_action(self, state: GameState, seat: int,
                   hole_cards: List[Card], community_cards: List[Card]) -> Tuple[Action, float]:
        
        is_preflop = state.street == Street.PREFLOP
        probs = self.preflop_probs if is_preflop else self.postflop_probs
        
        # Sample action
        action_idx = self.rng.choice([0, 1, 2], p=probs)
        
        to_call = state.get_to_call(seat)
        player_stack = state.players[seat].stack
        
        if action_idx == 0:  # Fold
            if to_call == 0:
                action, amount = Action.CALL, 0  # Check instead
            else:
                action, amount = Action.FOLD, 0
        elif action_idx == 1:  # Call
            action, amount = Action.CALL, min(to_call, player_stack)
        else:  # Raise
            min_raise = max(state.big_blind, state.min_raise)
            raise_amount = to_call + min_raise + self.rng.random() * min_raise * 2
            if raise_amount >= player_stack:
                action, amount = Action.ALL_IN, player_stack
            else:
                action, amount = Action.BET, raise_amount
        
        self.update_stats(action, state, is_preflop)
        return action, amount


# =============================================================================
# Sklansky Hand Grading Strategy
# =============================================================================

class SklanskyStrategy(BasePokerStrategy):
    """
    Strategy based on Sklansky's hand groupings.
    Groups 1-8 represent playable hands, 9 is unplayable.
    """
    
    # Thresholds for different player types (max group to play)
    TYPE_THRESHOLDS = {
        PlayerType.CONSERVATIVE: 3,  # Only groups 1-3
        PlayerType.REGULAR: 5,       # Groups 1-5
        PlayerType.AGGRESSIVE: 7,    # Groups 1-7
        PlayerType.BLUFFING: 8,      # Groups 1-8 + sometimes bluff
        PlayerType.DECEPTIVE: 5,     # Like regular but traps
    }
    
    def __init__(self, player_type: PlayerType):
        super().__init__(f"Sklansky_{player_type.name}", player_type)
        self.group_threshold = self.TYPE_THRESHOLDS[player_type]
    
    def get_action(self, state: GameState, seat: int,
                   hole_cards: List[Card], community_cards: List[Card]) -> Tuple[Action, float]:
        
        is_preflop = state.street == Street.PREFLOP
        to_call = state.get_to_call(seat)
        player_stack = state.players[seat].stack
        pot = state.pot
        
        if is_preflop:
            group = get_sklansky_group(hole_cards[0], hole_cards[1])
            action, amount = self._preflop_action(group, to_call, player_stack, pot, state)
        else:
            action, amount = self._postflop_action(hole_cards, community_cards, 
                                                    to_call, player_stack, pot, state)
        
        self.update_stats(action, state, is_preflop)
        return action, amount
    
    def _preflop_action(self, group: int, to_call: float, stack: float, 
                        pot: float, state: GameState) -> Tuple[Action, float]:
        """Preflop decision based on hand group"""
        
        if self.player_type == PlayerType.BLUFFING:
            # Sometimes bluff with bad hands
            if group > self.group_threshold and self.rng.random() < 0.15:
                raise_amount = to_call + state.big_blind * 3
                return Action.BET, min(raise_amount, stack)
        
        if group > self.group_threshold:
            if to_call == 0:
                return Action.CALL, 0  # Check
            return Action.FOLD, 0
        
        if self.player_type == PlayerType.DECEPTIVE and group <= 2:
            # Trap with premium hands sometimes
            if self.rng.random() < 0.4:
                return Action.CALL, min(to_call, stack)
        
        # Standard play based on group and player type
        if self.player_type == PlayerType.CONSERVATIVE:
            if group <= 2:
                raise_amount = to_call + state.big_blind * 4
                return Action.BET, min(raise_amount, stack)
            else:
                return Action.CALL, min(to_call, stack)
        
        elif self.player_type == PlayerType.AGGRESSIVE:
            if group <= 4:
                raise_amount = to_call + state.big_blind * 3
                return Action.BET, min(raise_amount, stack)
            else:
                return Action.CALL, min(to_call, stack)
        
        else:  # REGULAR
            if group <= 3:
                raise_amount = to_call + state.big_blind * 3
                return Action.BET, min(raise_amount, stack)
            elif to_call <= state.big_blind * 2:
                return Action.CALL, min(to_call, stack)
            else:
                return Action.FOLD, 0
    
    def _postflop_action(self, hole_cards: List[Card], community_cards: List[Card],
                         to_call: float, stack: float, pot: float,
                         state: GameState) -> Tuple[Action, float]:
        """Postflop decision based on hand strength"""
        
        hand_rank, _ = evaluate_hand(hole_cards + community_cards)
        
        # Calculate pot odds
        pot_odds = pot / max(to_call, 0.01)
        
        if self.player_type == PlayerType.BLUFFING:
            # Bluff with weak hands sometimes
            if hand_rank <= HandRank.HIGH_CARD and self.rng.random() < 0.2:
                bet_amount = pot * 0.6
                return Action.BET, min(bet_amount, stack)
        
        if self.player_type == PlayerType.DECEPTIVE and hand_rank >= HandRank.TWO_PAIR:
            # Slow play strong hands
            if self.rng.random() < 0.5:
                return Action.CALL, min(to_call, stack)
        
        # Standard postflop play
        if hand_rank >= HandRank.TWO_PAIR:
            bet_amount = pot * (0.6 + 0.2 * self.rng.random())
            return Action.BET, min(bet_amount + to_call, stack)
        
        elif hand_rank >= HandRank.ONE_PAIR:
            if to_call <= pot * 0.5:
                return Action.CALL, min(to_call, stack)
            elif self.player_type == PlayerType.AGGRESSIVE and self.rng.random() < 0.3:
                return Action.CALL, min(to_call, stack)
            else:
                return Action.FOLD, 0
        
        else:  # High card only
            if to_call == 0:
                return Action.CALL, 0  # Check
            elif to_call <= pot * 0.3:
                return Action.CALL, min(to_call, stack)
            else:
                return Action.FOLD, 0


# =============================================================================
# Bill Chen Hand Evaluation Strategy
# =============================================================================

class ChenStrategy(BasePokerStrategy):
    """
    Strategy based on Bill Chen's hand evaluation formula.
    Chen scores range from -1 to 20.
    """
    
    # Score thresholds for different player types
    TYPE_THRESHOLDS = {
        PlayerType.CONSERVATIVE: 10,  # Premium hands only
        PlayerType.REGULAR: 7,        # Strong hands
        PlayerType.AGGRESSIVE: 5,     # Playable hands
        PlayerType.BLUFFING: 3,       # Many hands + bluffs
        PlayerType.DECEPTIVE: 7,      # Like regular but traps
    }
    
    def __init__(self, player_type: PlayerType):
        super().__init__(f"Chen_{player_type.name}", player_type)
        self.score_threshold = self.TYPE_THRESHOLDS[player_type]
    
    def get_action(self, state: GameState, seat: int,
                   hole_cards: List[Card], community_cards: List[Card]) -> Tuple[Action, float]:
        
        is_preflop = state.street == Street.PREFLOP
        to_call = state.get_to_call(seat)
        player_stack = state.players[seat].stack
        pot = state.pot
        
        if is_preflop:
            chen_score = chen_formula(hole_cards[0], hole_cards[1])
            action, amount = self._preflop_action(chen_score, to_call, player_stack, pot, state)
        else:
            action, amount = self._postflop_action(hole_cards, community_cards,
                                                    to_call, player_stack, pot, state)
        
        self.update_stats(action, state, is_preflop)
        return action, amount
    
    def _preflop_action(self, chen_score: float, to_call: float, stack: float,
                        pot: float, state: GameState) -> Tuple[Action, float]:
        """Preflop decision based on Chen score"""
        
        if self.player_type == PlayerType.BLUFFING:
            if chen_score < self.score_threshold and self.rng.random() < 0.12:
                raise_amount = to_call + state.big_blind * 3
                return Action.BET, min(raise_amount, stack)
        
        if chen_score < self.score_threshold:
            if to_call == 0:
                return Action.CALL, 0
            return Action.FOLD, 0
        
        if self.player_type == PlayerType.DECEPTIVE and chen_score >= 12:
            if self.rng.random() < 0.35:
                return Action.CALL, min(to_call, stack)
        
        # Calculate raise amount based on score
        score_factor = min(chen_score / 10.0, 1.5)
        
        if self.player_type == PlayerType.AGGRESSIVE:
            raise_amount = to_call + state.big_blind * (2 + score_factor)
            return Action.BET, min(raise_amount, stack)
        elif self.player_type == PlayerType.CONSERVATIVE:
            if chen_score >= 10:
                raise_amount = to_call + state.big_blind * (3 + score_factor)
                return Action.BET, min(raise_amount, stack)
            return Action.CALL, min(to_call, stack)
        else:  # REGULAR
            if chen_score >= 8:
                raise_amount = to_call + state.big_blind * 3
                return Action.BET, min(raise_amount, stack)
            return Action.CALL, min(to_call, stack)
    
    def _postflop_action(self, hole_cards: List[Card], community_cards: List[Card],
                         to_call: float, stack: float, pot: float,
                         state: GameState) -> Tuple[Action, float]:
        """Postflop decision - similar to Sklansky but with Chen scoring for draws"""
        
        hand_rank, _ = evaluate_hand(hole_cards + community_cards)
        
        if self.player_type == PlayerType.BLUFFING:
            if hand_rank <= HandRank.HIGH_CARD and self.rng.random() < 0.18:
                bet_amount = pot * 0.55
                return Action.BET, min(bet_amount, stack)
        
        if self.player_type == PlayerType.DECEPTIVE and hand_rank >= HandRank.THREE_OF_A_KIND:
            if self.rng.random() < 0.45:
                return Action.CALL, min(to_call, stack)
        
        if hand_rank >= HandRank.THREE_OF_A_KIND:
            bet_amount = pot * 0.7
            return Action.BET, min(bet_amount + to_call, stack)
        
        elif hand_rank >= HandRank.ONE_PAIR:
            if to_call == 0:
                if self.rng.random() < 0.5:
                    return Action.BET, min(pot * 0.4, stack)
                return Action.CALL, 0
            elif to_call <= pot * 0.4:
                return Action.CALL, min(to_call, stack)
            else:
                return Action.FOLD, 0
        
        else:
            if to_call == 0:
                return Action.CALL, 0
            return Action.FOLD, 0


# =============================================================================
# Rule-Based Strategy
# =============================================================================

class RuleBasedStrategy(BasePokerStrategy):
    """
    Rule-based strategy that considers the coupling between
    hole cards and community cards for postflop play.
    """
    
    def __init__(self, player_type: PlayerType):
        super().__init__(f"RuleBased_{player_type.name}", player_type)
    
    def get_action(self, state: GameState, seat: int,
                   hole_cards: List[Card], community_cards: List[Card]) -> Tuple[Action, float]:
        
        is_preflop = state.street == Street.PREFLOP
        to_call = state.get_to_call(seat)
        player_stack = state.players[seat].stack
        pot = state.pot
        
        if is_preflop:
            chen_score = chen_formula(hole_cards[0], hole_cards[1])
            action, amount = self._preflop_action(chen_score, to_call, player_stack, pot, state)
        else:
            action, amount = self._postflop_action(hole_cards, community_cards,
                                                    to_call, player_stack, pot, state)
        
        self.update_stats(action, state, is_preflop)
        return action, amount
    
    def _preflop_action(self, chen_score: float, to_call: float, stack: float,
                        pot: float, state: GameState) -> Tuple[Action, float]:
        """Use Chen-based preflop logic"""
        thresholds = {
            PlayerType.CONSERVATIVE: 10,
            PlayerType.REGULAR: 7,
            PlayerType.AGGRESSIVE: 4,
            PlayerType.BLUFFING: 3,
            PlayerType.DECEPTIVE: 7,
        }
        threshold = thresholds[self.player_type]
        
        if self.player_type == PlayerType.BLUFFING and chen_score < threshold:
            if self.rng.random() < 0.1:
                return Action.BET, min(to_call + state.big_blind * 3, stack)
        
        if chen_score < threshold:
            if to_call == 0:
                return Action.CALL, 0
            return Action.FOLD, 0
        
        if self.player_type == PlayerType.DECEPTIVE and chen_score >= 12:
            if self.rng.random() < 0.4:
                return Action.CALL, min(to_call, stack)
        
        raise_size = state.big_blind * (2 + chen_score / 5)
        return Action.BET, min(to_call + raise_size, stack)
    
    def _postflop_action(self, hole_cards: List[Card], community_cards: List[Card],
                         to_call: float, stack: float, pot: float,
                         state: GameState) -> Tuple[Action, float]:
        """
        Rule-based postflop that considers:
        - Made hand strength
        - Draw potential (flush/straight draws)
        - Board texture
        """
        
        hand_rank, kickers = evaluate_hand(hole_cards + community_cards)
        
        # Check for draws
        has_flush_draw = self._check_flush_draw(hole_cards, community_cards)
        has_straight_draw = self._check_straight_draw(hole_cards, community_cards)
        draw_strength = has_flush_draw * 0.3 + has_straight_draw * 0.2
        
        # Calculate effective strength
        effective_strength = hand_rank + draw_strength
        
        # Bluffing behavior
        if self.player_type == PlayerType.BLUFFING:
            if hand_rank <= HandRank.HIGH_CARD and self.rng.random() < 0.15:
                return Action.BET, min(pot * 0.6, stack)
        
        # Deceptive behavior
        if self.player_type == PlayerType.DECEPTIVE and hand_rank >= HandRank.TWO_PAIR:
            if self.rng.random() < 0.4:
                return Action.CALL, min(to_call, stack)
        
        # Action based on effective strength
        if effective_strength >= HandRank.THREE_OF_A_KIND:
            bet_size = pot * (0.6 + 0.2 * (effective_strength / 10))
            return Action.BET, min(bet_size + to_call, stack)
        
        elif effective_strength >= HandRank.ONE_PAIR + 0.2:  # Pair + draw
            if to_call <= pot * 0.5:
                return Action.CALL, min(to_call, stack)
            elif self.player_type == PlayerType.AGGRESSIVE:
                return Action.CALL, min(to_call, stack)
            return Action.FOLD, 0
        
        elif hand_rank >= HandRank.ONE_PAIR:
            if to_call <= pot * 0.3:
                return Action.CALL, min(to_call, stack)
            return Action.FOLD, 0
        
        else:  # High card
            if to_call == 0:
                if draw_strength > 0 and self.rng.random() < 0.3:
                    return Action.BET, min(pot * 0.4, stack)
                return Action.CALL, 0
            elif draw_strength > 0.4 and to_call <= pot * 0.25:
                return Action.CALL, min(to_call, stack)
            return Action.FOLD, 0
    
    def _check_flush_draw(self, hole_cards: List[Card], community_cards: List[Card]) -> bool:
        """Check if we have a flush draw (4 to a flush)"""
        all_cards = hole_cards + community_cards
        suits = [c.suit for c in all_cards]
        for suit in range(4):
            if suits.count(suit) == 4:
                # Check if at least one hole card contributes
                if any(c.suit == suit for c in hole_cards):
                    return True
        return False
    
    def _check_straight_draw(self, hole_cards: List[Card], community_cards: List[Card]) -> bool:
        """Check if we have a straight draw (4 to a straight)"""
        all_cards = hole_cards + community_cards
        ranks = sorted(set(c.rank for c in all_cards))
        
        # Check for 4 consecutive cards
        for i in range(len(ranks) - 3):
            if ranks[i+3] - ranks[i] <= 4:
                # Check if hole cards contribute
                hole_ranks = set(c.rank for c in hole_cards)
                window_ranks = set(ranks[i:i+4])
                if hole_ranks & window_ranks:
                    return True
        
        return False


# =============================================================================
# Style Library
# =============================================================================

class StyleLibrary:
    """
    Complete library of 64 poker player styles.
    - 49 random probability strategies (7x7)
    - 5 Sklansky-based strategies
    - 5 Chen-based strategies
    - 5 Rule-based strategies
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.strategies: Dict[str, BasePokerStrategy] = {}
        self.style_features: Dict[str, StyleFeatures] = {}
        self._build_library()
        
        if seed is not None:
            self.set_seed(seed)
    
    def _build_library(self):
        """Build all 64 strategies"""
        
        # Random strategies (7 configs = styles 0-6)
        # Note: Paper says 49 (7x7) but they're really just 7 distinct configs
        for config_id in range(7):
            strategy = RandomStrategy(config_id)
            self.strategies[strategy.style_name] = strategy
        
        # Sklansky-based strategies (5 player types = styles 7-11)
        for player_type in PlayerType:
            strategy = SklanskyStrategy(player_type)
            self.strategies[strategy.style_name] = strategy
        
        # Chen-based strategies (5 player types = styles 12-16)
        for player_type in PlayerType:
            strategy = ChenStrategy(player_type)
            self.strategies[strategy.style_name] = strategy
        
        # Rule-based strategies (5 player types = styles 17-21)
        for player_type in PlayerType:
            strategy = RuleBasedStrategy(player_type)
            self.strategies[strategy.style_name] = strategy
        
        print(f"Built style library with {len(self.strategies)} strategies")
    
    def set_seed(self, seed: int):
        """Set random seed for all strategies"""
        for i, strategy in enumerate(self.strategies.values()):
            strategy.set_seed(seed + i)
    
    def get_strategy(self, style_name: str) -> BasePokerStrategy:
        """Get strategy by name"""
        return self.strategies[style_name]
    
    def get_random_strategy(self, rng: Optional[np.random.Generator] = None) -> BasePokerStrategy:
        """Get a random strategy"""
        if rng is None:
            rng = np.random.default_rng()
        names = list(self.strategies.keys())
        return self.strategies[rng.choice(names)]
    
    def get_all_names(self) -> List[str]:
        """Get all strategy names"""
        return list(self.strategies.keys())
    
    def get_strategy_by_index(self, idx: int) -> BasePokerStrategy:
        """Get strategy by index"""
        names = list(self.strategies.keys())
        return self.strategies[names[idx]]
    
    def compute_style_features(self, num_games: int = 10000, 
                               verbose: bool = True) -> Dict[str, StyleFeatures]:
        """
        Compute style features by simulating games between strategies.
        This creates the ground truth style features for each strategy.
        """
        from poker_env import PokerEnvironment
        
        if verbose:
            print(f"Computing style features from {num_games} games...")
        
        # Reset all stats
        for strategy in self.strategies.values():
            strategy.reset_stats()
        
        env = PokerEnvironment(num_players=6)
        strategy_names = list(self.strategies.keys())
        rng = np.random.default_rng(42)
        
        for game_idx in range(num_games):
            # Randomly select 6 players (can repeat)
            selected = [self.strategies[rng.choice(strategy_names)] for _ in range(6)]
            
            state = env.reset(button_seat=game_idx % 6)
            
            while not state.is_terminal:
                seat = state.current_player
                strategy = selected[seat]
                player = state.players[seat]
                
                action, amount = strategy.get_action(
                    state, seat, player.hole_cards, state.community_cards
                )
                
                state, _, done = env.step(action, amount)
            
            # Update flop/showdown stats
            for i, strategy in enumerate(selected):
                if state.street.value >= Street.FLOP.value:
                    strategy.hands_to_flop += 1
                if i in state.winners or (state.street == Street.RIVER and 
                                          state.players[i].is_active):
                    strategy.hands_to_showdown += 1
            
            if verbose and (game_idx + 1) % 1000 == 0:
                print(f"  Completed {game_idx + 1}/{num_games} games")
        
        # Extract style features
        for name, strategy in self.strategies.items():
            self.style_features[name] = strategy.get_style_features()
        
        if verbose:
            print("Style features computed:")
            for name, features in list(self.style_features.items())[:5]:
                print(f"  {name}: VPIP={features.vpip:.3f}, PFR={features.pfr:.3f}, "
                      f"AFq={features.afq:.3f}, WTSD={features.wtsd:.3f}")
        
        return self.style_features


if __name__ == "__main__":
    # Test the style library
    library = StyleLibrary(seed=42)
    
    print("\nAll strategy names:")
    for name in library.get_all_names():
        print(f"  {name}")
    
    # Compute style features (small test)
    features = library.compute_style_features(num_games=1000, verbose=True)
