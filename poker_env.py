"""
poker_env.py - Texas Hold'em Game Environment
Implements full 6-player No-Limit Texas Hold'em with proper betting mechanics.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from copy import deepcopy

from poker_core import (
    Card, Deck, Action, Street, PlayerState, GameState,
    evaluate_hand, compare_hands, HandRank
)


class PokerEnvironment:
    """
    6-player No-Limit Texas Hold'em environment.
    Follows standard poker rules with proper betting rounds.
    """
    
    def __init__(self,
                 num_players: int = 6,
                 small_blind: float = 0.5,
                 big_blind: float = 1.0,
                 starting_stack: float = 100.0,
                 seed: Optional[int] = None):
        
        self.num_players = num_players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.starting_stack = starting_stack
        self.deck = Deck(seed)
        
        self.state: Optional[GameState] = None
        self.rng = np.random.default_rng(seed)
        
    def reset(self, button_seat: int = 0) -> GameState:
        """Start a new hand"""
        self.deck.reset()
        
        # Initialize players
        players = [
            PlayerState(
                seat=i,
                stack=self.starting_stack,
                bet_this_round=0.0,
                bet_this_street=0.0,
                is_active=True,
                has_acted=False,
                hole_cards=[]
            )
            for i in range(self.num_players)
        ]
        
        # Create game state
        self.state = GameState(
            num_players=self.num_players,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            players=players,
            button_seat=button_seat,
            street=Street.PREFLOP,
            pot=0.0,
            current_bet=0.0,
            min_raise=self.big_blind,
            action_history=[],
            street_history={Street.PREFLOP: []},
            is_terminal=False,
            winners=[]
        )
        
        # Post blinds
        self._post_blinds()
        
        # Deal hole cards
        for player in self.state.players:
            player.hole_cards = self.deck.deal(2)
        
        # Set first to act (UTG in 6-max, left of BB)
        self.state.current_player = (button_seat + 3) % self.num_players
        
        return self.state.copy()
    
    def _post_blinds(self):
        """Post small and big blinds"""
        sb_seat = (self.state.button_seat + 1) % self.num_players
        bb_seat = (self.state.button_seat + 2) % self.num_players
        
        # Small blind
        sb_amount = min(self.small_blind, self.state.players[sb_seat].stack)
        self.state.players[sb_seat].stack -= sb_amount
        self.state.players[sb_seat].bet_this_street = sb_amount
        self.state.players[sb_seat].bet_this_round = sb_amount
        self.state.pot += sb_amount
        
        # Big blind
        bb_amount = min(self.big_blind, self.state.players[bb_seat].stack)
        self.state.players[bb_seat].stack -= bb_amount
        self.state.players[bb_seat].bet_this_street = bb_amount
        self.state.players[bb_seat].bet_this_round = bb_amount
        self.state.pot += bb_amount
        
        self.state.current_bet = bb_amount
        self.state.min_raise = self.big_blind
    
    def step(self, action: Action, amount: float = 0.0) -> Tuple[GameState, Dict[int, float], bool]:
        """
        Execute an action and return (new_state, rewards, done).
        Rewards dict maps player seat to reward (only non-zero at terminal).
        """
        if self.state.is_terminal:
            raise ValueError("Game is already over")
        
        seat = self.state.current_player
        player = self.state.players[seat]
        
        # Execute action
        if action == Action.FOLD:
            self._execute_fold(seat)
        elif action == Action.CALL:
            self._execute_call(seat)
        elif action == Action.BET:
            self._execute_bet(seat, amount)
        elif action == Action.ALL_IN:
            self._execute_all_in(seat)
        
        # Record action
        self.state.action_history.append((seat, action, amount))
        self.state.street_history[self.state.street].append((seat, action, amount))
        
        # Mark player as having acted
        player.has_acted = True
        
        # Check for hand completion
        active_players = self.state.get_active_players()
        if len(active_players) == 1:
            # All but one folded
            self._award_pot([active_players[0]])
            return self.state.copy(), self._get_rewards(), True
        
        # Move to next player or next street
        if self._is_betting_complete():
            if self.state.street == Street.RIVER:
                # Showdown
                self._showdown()
                return self.state.copy(), self._get_rewards(), True
            else:
                self._advance_street()
        else:
            self._advance_player()
        
        return self.state.copy(), self._get_rewards(), self.state.is_terminal
    
    def _execute_fold(self, seat: int):
        """Player folds"""
        self.state.players[seat].is_active = False
    
    def _execute_call(self, seat: int):
        """Player calls or checks"""
        player = self.state.players[seat]
        to_call = self.state.get_to_call(seat)
        call_amount = min(to_call, player.stack)
        
        player.stack -= call_amount
        player.bet_this_street += call_amount
        player.bet_this_round += call_amount
        self.state.pot += call_amount
    
    def _execute_bet(self, seat: int, amount: float):
        """Player bets or raises"""
        player = self.state.players[seat]
        
        # Amount is the total amount being put in this street
        actual_amount = min(amount, player.stack)
        
        # Calculate raise amount (for min_raise update)
        old_bet = self.state.current_bet
        new_bet = player.bet_this_street + actual_amount
        raise_amount = new_bet - old_bet
        
        player.stack -= actual_amount
        player.bet_this_street += actual_amount
        player.bet_this_round += actual_amount
        self.state.pot += actual_amount
        
        self.state.current_bet = new_bet
        self.state.min_raise = max(self.state.min_raise, raise_amount)
        
        # Reset has_acted for other active players (reopens betting)
        for p in self.state.players:
            if p.seat != seat and p.is_active and p.stack > 0:
                p.has_acted = False
    
    def _execute_all_in(self, seat: int):
        """Player goes all-in"""
        self._execute_bet(seat, self.state.players[seat].stack)
    
    def _is_betting_complete(self) -> bool:
        """Check if betting round is complete"""
        active_with_chips = [
            p for p in self.state.players
            if p.is_active and p.stack > 0
        ]
        
        # All active players with chips have acted and matched current bet
        for player in active_with_chips:
            if not player.has_acted:
                return False
            if player.bet_this_street < self.state.current_bet:
                return False
        
        return True
    
    def _advance_player(self):
        """Move to next active player"""
        seat = self.state.current_player
        
        for _ in range(self.num_players):
            seat = (seat + 1) % self.num_players
            player = self.state.players[seat]
            if player.is_active and player.stack > 0 and not player.has_acted:
                self.state.current_player = seat
                return
            # Also include players who need to act due to a raise
            if player.is_active and player.stack > 0:
                if player.bet_this_street < self.state.current_bet:
                    self.state.current_player = seat
                    return
        
        # Shouldn't reach here if betting isn't complete
        self.state.current_player = seat
    
    def _advance_street(self):
        """Move to next street"""
        # Reset betting
        for player in self.state.players:
            player.bet_this_street = 0.0
            player.has_acted = False
        
        self.state.current_bet = 0.0
        self.state.min_raise = self.big_blind
        
        # Deal community cards
        if self.state.street == Street.PREFLOP:
            self.state.street = Street.FLOP
            self.state.community_cards.extend(self.deck.deal(3))
        elif self.state.street == Street.FLOP:
            self.state.street = Street.TURN
            self.state.community_cards.extend(self.deck.deal(1))
        elif self.state.street == Street.TURN:
            self.state.street = Street.RIVER
            self.state.community_cards.extend(self.deck.deal(1))
        
        # Initialize street history
        self.state.street_history[self.state.street] = []
        
        # First to act is left of button
        seat = self.state.button_seat
        for _ in range(self.num_players):
            seat = (seat + 1) % self.num_players
            if self.state.players[seat].is_active and self.state.players[seat].stack > 0:
                self.state.current_player = seat
                return
    
    def _showdown(self):
        """Determine winner(s) at showdown"""
        active_players = self.state.get_active_players()
        
        if len(active_players) == 1:
            self._award_pot(active_players)
            return
        
        # Evaluate all hands
        hand_values = []
        for seat in active_players:
            player = self.state.players[seat]
            full_hand = player.hole_cards + self.state.community_cards
            hand_eval = evaluate_hand(full_hand)
            hand_values.append((seat, hand_eval))
        
        # Find best hand(s)
        best_eval = max(hv[1] for hv in hand_values)
        winners = [seat for seat, eval_val in hand_values if eval_val == best_eval]
        
        self._award_pot(winners)
    
    def _award_pot(self, winners: List[int]):
        """Award pot to winner(s)"""
        # Simple pot distribution (doesn't handle side pots)
        pot_share = self.state.pot / len(winners)
        
        for seat in winners:
            self.state.players[seat].stack += pot_share
        
        self.state.winners = winners
        self.state.is_terminal = True
    
    def _get_rewards(self) -> Dict[int, float]:
        """Calculate rewards (profit/loss) for each player"""
        rewards = {}
        for player in self.state.players:
            # Reward is change in stack from starting
            rewards[player.seat] = player.stack - self.starting_stack
        return rewards
    
    def get_observation(self, seat: int) -> Dict[str, Any]:
        """
        Get observation for a specific player.
        Contains all public info + player's private hole cards.
        """
        player = self.state.players[seat]
        
        return {
            # Player's private info
            'hole_cards': player.hole_cards,
            'stack': player.stack,
            'bet_this_round': player.bet_this_round,
            'position': self._get_position(seat),
            
            # Public info
            'street': self.state.street,
            'community_cards': self.state.community_cards,
            'pot': self.state.pot,
            'current_bet': self.state.current_bet,
            'to_call': self.state.get_to_call(seat),
            
            # Other players (visible info only)
            'players': [
                {
                    'seat': p.seat,
                    'stack': p.stack,
                    'bet_this_street': p.bet_this_street,
                    'is_active': p.is_active,
                    'position': self._get_position(p.seat),
                }
                for p in self.state.players
            ],
            
            # Action history
            'action_history': self.state.action_history,
            'street_history': self.state.street_history,
            
            # Game state
            'num_active': len(self.state.get_active_players()),
            'legal_actions': self.state.get_legal_actions(),
        }
    
    def _get_position(self, seat: int) -> str:
        """Get position name for a seat"""
        relative = (seat - self.state.button_seat) % self.num_players
        positions = ['BTN', 'SB', 'BB', 'UTG', 'HJ', 'CO']
        return positions[relative] if relative < len(positions) else f'P{relative}'
    
    def render(self):
        """Print current game state"""
        if self.state is None:
            print("No game in progress")
            return
        
        print("=" * 60)
        print(f"Street: {self.state.street.name}")
        print(f"Pot: {self.state.pot:.1f} | Current bet: {self.state.current_bet:.1f}")
        print(f"Community: {self.state.community_cards}")
        print("-" * 60)
        
        for player in self.state.players:
            status = "ACTIVE" if player.is_active else "FOLDED"
            position = self._get_position(player.seat)
            marker = ">>>" if player.seat == self.state.current_player else "   "
            print(f"{marker} P{player.seat} ({position}): Stack={player.stack:.1f}, "
                  f"Bet={player.bet_this_street:.1f}, {status}")
            if player.is_active:
                print(f"      Hole: {player.hole_cards}")
        
        print("=" * 60)


class PokerEnvVec:
    """Vectorized poker environment for parallel training"""
    
    def __init__(self, num_envs: int, **kwargs):
        self.envs = [PokerEnvironment(**kwargs) for _ in range(num_envs)]
        self.num_envs = num_envs
    
    def reset(self) -> List[GameState]:
        return [env.reset(button_seat=i % 6) for i, env in enumerate(self.envs)]
    
    def step(self, actions: List[Tuple[Action, float]]) -> Tuple[List[GameState], List[Dict], List[bool]]:
        results = []
        for env, (action, amount) in zip(self.envs, actions):
            results.append(env.step(action, amount))
        
        states = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        
        return states, rewards, dones


if __name__ == "__main__":
    # Test game
    env = PokerEnvironment(num_players=6, seed=42)
    state = env.reset()
    
    print("Initial state:")
    env.render()
    
    # Play a simple hand
    actions = [
        (Action.FOLD, 0),   # UTG folds
        (Action.FOLD, 0),   # HJ folds
        (Action.CALL, 0),   # CO calls
        (Action.FOLD, 0),   # BTN folds
        (Action.CALL, 0),   # SB calls
        (Action.BET, 3.0),  # BB raises
    ]
    
    for action, amount in actions:
        if env.state.is_terminal:
            break
        print(f"\nPlayer {env.state.current_player} takes action: {action.name}, amount: {amount}")
        state, rewards, done = env.step(action, amount)
        env.render()
        
        if done:
            print("\nHand complete!")
            print(f"Winners: {state.winners}")
            print(f"Rewards: {rewards}")
            break
