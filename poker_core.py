"""
poker_core.py - Core poker game mechanics for Texas Hold'em
Implements card representation, hand evaluation, and game state management.
"""

import numpy as np
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from itertools import combinations

# =============================================================================
# Card Representation
# =============================================================================

class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

SUIT_SYMBOLS = {Suit.CLUBS: '♣', Suit.DIAMONDS: '♦', Suit.HEARTS: '♥', Suit.SPADES: '♠'}
RANK_SYMBOLS = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
                9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}

@dataclass(frozen=True)
class Card:
    rank: int
    suit: int
    
    def __repr__(self):
        return f"{RANK_SYMBOLS[self.rank]}{SUIT_SYMBOLS[Suit(self.suit)]}"
    
    def to_int(self) -> int:
        """Convert to integer 0-51"""
        return (self.rank - 2) * 4 + self.suit
    
    @staticmethod
    def from_int(i: int) -> 'Card':
        """Create card from integer 0-51"""
        return Card(rank=(i // 4) + 2, suit=i % 4)
    
    @staticmethod
    def from_str(s: str) -> 'Card':
        """Create card from string like 'Ah' or 'Ts'"""
        rank_map = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3, 
                   '♣': 0, '♦': 1, '♥': 2, '♠': 3}
        return Card(rank=rank_map[s[0].upper()], suit=suit_map[s[1].lower()])


class Deck:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()
    
    def reset(self):
        self.cards = [Card.from_int(i) for i in range(52)]
        self.rng.shuffle(self.cards)
        self.index = 0
    
    def deal(self, n: int = 1) -> List[Card]:
        dealt = self.cards[self.index:self.index + n]
        self.index += n
        return dealt
    
    def deal_one(self) -> Card:
        return self.deal(1)[0]


# =============================================================================
# Hand Evaluation (optimized for speed)
# =============================================================================

class HandRank(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
    """
    Evaluate a 5-7 card hand and return (hand_rank, kickers).
    Returns the best 5-card combination from the given cards.
    """
    if len(cards) < 5:
        raise ValueError("Need at least 5 cards")
    
    if len(cards) > 5:
        # Find best 5-card combination
        best_rank = (-1, [])
        for combo in combinations(cards, 5):
            rank = _evaluate_5_cards(list(combo))
            if rank > best_rank:
                best_rank = rank
        return best_rank
    
    return _evaluate_5_cards(cards)


def _evaluate_5_cards(cards: List[Card]) -> Tuple[int, List[int]]:
    """Evaluate exactly 5 cards"""
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    
    # Check for flush
    is_flush = len(set(suits)) == 1
    
    # Check for straight (including wheel: A-2-3-4-5)
    is_straight = False
    straight_high = 0
    unique_ranks = sorted(set(ranks), reverse=True)
    
    if len(unique_ranks) >= 5:
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i] - unique_ranks[i+4] == 4:
                is_straight = True
                straight_high = unique_ranks[i]
                break
        
        # Check for wheel (A-2-3-4-5)
        if not is_straight and set([14, 5, 4, 3, 2]).issubset(set(ranks)):
            is_straight = True
            straight_high = 5  # 5-high straight
    
    # Count rank occurrences
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    
    # Sort by count then by rank
    count_rank = sorted([(count, rank) for rank, count in rank_counts.items()],
                       key=lambda x: (x[0], x[1]), reverse=True)
    
    # Determine hand type
    if is_straight and is_flush:
        if straight_high == 14:
            return (HandRank.ROYAL_FLUSH, [14])
        return (HandRank.STRAIGHT_FLUSH, [straight_high])
    
    if count_rank[0][0] == 4:
        quad_rank = count_rank[0][1]
        kicker = count_rank[1][1]
        return (HandRank.FOUR_OF_A_KIND, [quad_rank, kicker])
    
    if count_rank[0][0] == 3 and count_rank[1][0] >= 2:
        trip_rank = count_rank[0][1]
        pair_rank = count_rank[1][1]
        return (HandRank.FULL_HOUSE, [trip_rank, pair_rank])
    
    if is_flush:
        return (HandRank.FLUSH, ranks[:5])
    
    if is_straight:
        return (HandRank.STRAIGHT, [straight_high])
    
    if count_rank[0][0] == 3:
        trip_rank = count_rank[0][1]
        kickers = sorted([r for r in ranks if r != trip_rank], reverse=True)[:2]
        return (HandRank.THREE_OF_A_KIND, [trip_rank] + kickers)
    
    if count_rank[0][0] == 2 and count_rank[1][0] == 2:
        high_pair = count_rank[0][1]
        low_pair = count_rank[1][1]
        kicker = [r for r in ranks if r != high_pair and r != low_pair][0]
        return (HandRank.TWO_PAIR, [high_pair, low_pair, kicker])
    
    if count_rank[0][0] == 2:
        pair_rank = count_rank[0][1]
        kickers = sorted([r for r in ranks if r != pair_rank], reverse=True)[:3]
        return (HandRank.ONE_PAIR, [pair_rank] + kickers)
    
    return (HandRank.HIGH_CARD, ranks[:5])


def compare_hands(hand1: List[Card], hand2: List[Card]) -> int:
    """
    Compare two hands. Returns:
    1 if hand1 wins, -1 if hand2 wins, 0 if tie
    """
    eval1 = evaluate_hand(hand1)
    eval2 = evaluate_hand(hand2)
    
    if eval1 > eval2:
        return 1
    elif eval1 < eval2:
        return -1
    return 0


# =============================================================================
# Hand Strength Calculation (Monte Carlo)
# =============================================================================

def calculate_hand_strength(hole_cards: List[Card], 
                           community_cards: List[Card],
                           num_opponents: int = 5,
                           num_simulations: int = 1000,
                           seed: Optional[int] = None) -> float:
    """
    Estimate hand strength using Monte Carlo simulation.
    Returns probability of winning against random opponent hands.
    """
    rng = np.random.default_rng(seed)
    
    # Create remaining deck
    used = set(c.to_int() for c in hole_cards + community_cards)
    remaining = [Card.from_int(i) for i in range(52) if i not in used]
    
    wins = 0
    ties = 0
    
    for _ in range(num_simulations):
        # Shuffle remaining cards
        rng.shuffle(remaining)
        idx = 0
        
        # Deal remaining community cards
        cards_needed = 5 - len(community_cards)
        sim_community = community_cards + remaining[idx:idx + cards_needed]
        idx += cards_needed
        
        # Deal opponent hands
        hero_hand = hole_cards + sim_community
        hero_eval = evaluate_hand(hero_hand)
        
        best_opponent = (-1, [])
        for _ in range(num_opponents):
            opp_hole = remaining[idx:idx + 2]
            idx += 2
            opp_hand = opp_hole + sim_community
            opp_eval = evaluate_hand(opp_hand)
            if opp_eval > best_opponent:
                best_opponent = opp_eval
        
        if hero_eval > best_opponent:
            wins += 1
        elif hero_eval == best_opponent:
            ties += 1
    
    return (wins + 0.5 * ties) / num_simulations


# =============================================================================
# Game State
# =============================================================================

class Action(IntEnum):
    FOLD = 0
    CALL = 1  # Also CHECK when to_call is 0
    BET = 2   # Also RAISE
    ALL_IN = 3


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass
class PlayerState:
    """State of a single player"""
    seat: int
    stack: float
    bet_this_round: float = 0.0
    bet_this_street: float = 0.0
    is_active: bool = True  # Still in the hand
    has_acted: bool = False
    hole_cards: List[Card] = field(default_factory=list)
    
    def copy(self) -> 'PlayerState':
        return PlayerState(
            seat=self.seat,
            stack=self.stack,
            bet_this_round=self.bet_this_round,
            bet_this_street=self.bet_this_street,
            is_active=self.is_active,
            has_acted=self.has_acted,
            hole_cards=self.hole_cards.copy()
        )


@dataclass
class GameState:
    """Complete game state for Texas Hold'em"""
    num_players: int
    small_blind: float
    big_blind: float
    
    # Player states
    players: List[PlayerState] = field(default_factory=list)
    
    # Position info
    button_seat: int = 0
    current_player: int = 0
    
    # Street info
    street: Street = Street.PREFLOP
    community_cards: List[Card] = field(default_factory=list)
    
    # Betting info
    pot: float = 0.0
    current_bet: float = 0.0
    min_raise: float = 0.0
    
    # Action history
    action_history: List[Tuple[int, Action, float]] = field(default_factory=list)
    street_history: Dict[Street, List[Tuple[int, Action, float]]] = field(default_factory=dict)
    
    # Terminal
    is_terminal: bool = False
    winners: List[int] = field(default_factory=list)
    
    def copy(self) -> 'GameState':
        """Deep copy of game state"""
        new_state = GameState(
            num_players=self.num_players,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            players=[p.copy() for p in self.players],
            button_seat=self.button_seat,
            current_player=self.current_player,
            street=self.street,
            community_cards=self.community_cards.copy(),
            pot=self.pot,
            current_bet=self.current_bet,
            min_raise=self.min_raise,
            action_history=self.action_history.copy(),
            street_history={k: v.copy() for k, v in self.street_history.items()},
            is_terminal=self.is_terminal,
            winners=self.winners.copy()
        )
        return new_state
    
    def get_active_players(self) -> List[int]:
        """Get list of active player seats"""
        return [p.seat for p in self.players if p.is_active]
    
    def get_to_call(self, seat: int) -> float:
        """Amount needed to call for a player"""
        return max(0, self.current_bet - self.players[seat].bet_this_street)
    
    def get_pot_odds(self, seat: int) -> float:
        """Calculate pot odds for a player"""
        to_call = self.get_to_call(seat)
        if to_call == 0:
            return float('inf')
        return self.pot / to_call
    
    def get_spr(self, seat: int) -> float:
        """Stack-to-pot ratio"""
        if self.pot == 0:
            return float('inf')
        return self.players[seat].stack / self.pot
    
    def get_legal_actions(self) -> List[Tuple[Action, float, float]]:
        """
        Returns list of (action, min_amount, max_amount) tuples.
        For FOLD/CALL, amounts are fixed.
        For BET/RAISE, amounts specify valid range.
        """
        seat = self.current_player
        player = self.players[seat]
        to_call = self.get_to_call(seat)
        
        actions = []
        
        # Can always fold (unless it's a free check)
        if to_call > 0:
            actions.append((Action.FOLD, 0, 0))
        
        # Call or Check
        call_amount = min(to_call, player.stack)
        actions.append((Action.CALL, call_amount, call_amount))
        
        # Bet/Raise
        if player.stack > to_call:
            if self.current_bet == 0:
                # Opening bet
                min_bet = self.big_blind
                max_bet = player.stack
            else:
                # Raise
                min_raise_to = self.current_bet + self.min_raise
                min_bet = min_raise_to - player.bet_this_street
                max_bet = player.stack
            
            if max_bet >= min_bet:
                actions.append((Action.BET, min_bet, max_bet))
        
        # All-in is always an option (as a special bet size)
        if player.stack > 0:
            actions.append((Action.ALL_IN, player.stack, player.stack))
        
        return actions


# =============================================================================
# Hand Strength Estimation (Sklansky / Chen methods)
# =============================================================================

# Sklansky hand groups (1 = best, 8 = worst playable, 9 = unplayable)
SKLANSKY_GROUPS = {
    # Group 1
    ('A', 'A'): 1, ('K', 'K'): 1, ('Q', 'Q'): 1, ('J', 'J'): 1, ('A', 'K', 's'): 1,
    # Group 2
    ('T', 'T'): 2, ('A', 'K'): 2, ('A', 'Q', 's'): 2, ('A', 'J', 's'): 2, ('K', 'Q', 's'): 2,
    # Group 3
    ('9', '9'): 3, ('J', 'T', 's'): 3, ('Q', 'J', 's'): 3, ('K', 'J', 's'): 3, 
    ('A', 'T', 's'): 3, ('A', 'Q'): 3,
    # Group 4
    ('T', '9', 's'): 4, ('K', 'Q'): 4, ('8', '8'): 4, ('Q', 'T', 's'): 4, 
    ('9', '8', 's'): 4, ('J', '9', 's'): 4, ('A', 'J'): 4, ('K', 'T', 's'): 4,
    # Group 5
    ('7', '7'): 5, ('8', '7', 's'): 5, ('Q', '9', 's'): 5, ('T', '8', 's'): 5,
    ('K', 'J'): 5, ('Q', 'J'): 5, ('J', 'T'): 5, ('7', '6', 's'): 5, ('9', '7', 's'): 5,
    ('A', 'x', 's'): 5, ('6', '5', 's'): 5,
    # Group 6
    ('6', '6'): 6, ('A', 'T'): 6, ('5', '5'): 6, ('8', '6', 's'): 6, ('K', 'T'): 6,
    ('Q', 'T'): 6, ('5', '4', 's'): 6, ('K', '9', 's'): 6, ('J', '8', 's'): 6, ('7', '5', 's'): 6,
    # Group 7
    ('4', '4'): 7, ('J', '9'): 7, ('6', '4', 's'): 7, ('T', '9'): 7, ('5', '3', 's'): 7,
    ('3', '3'): 7, ('9', '8'): 7, ('4', '3', 's'): 7, ('2', '2'): 7, ('K', 'x', 's'): 7,
    ('T', '7', 's'): 7, ('Q', '8', 's'): 7,
    # Group 8
    ('8', '7'): 8, ('A', '9'): 8, ('Q', '9'): 8, ('7', '6'): 8, ('4', '2', 's'): 8,
    ('3', '2', 's'): 8, ('9', '6', 's'): 8, ('8', '5', 's'): 8, ('J', '8'): 8, ('J', '7', 's'): 8,
    ('6', '5'): 8, ('5', '4'): 8, ('7', '4', 's'): 8, ('K', '9'): 8, ('T', '8'): 8,
}


def get_sklansky_group(card1: Card, card2: Card) -> int:
    """Get Sklansky hand group (1-8, 9 for unplayable)"""
    r1, r2 = card1.rank, card2.rank
    suited = card1.suit == card2.suit
    
    # Normalize so r1 >= r2
    if r1 < r2:
        r1, r2 = r2, r1
    
    r1_sym = RANK_SYMBOLS[r1]
    r2_sym = RANK_SYMBOLS[r2]
    
    # Check for pairs
    if r1 == r2:
        key = (r1_sym, r2_sym)
        return SKLANSKY_GROUPS.get(key, 9)
    
    # Check suited vs offsuit
    if suited:
        key = (r1_sym, r2_sym, 's')
        if key in SKLANSKY_GROUPS:
            return SKLANSKY_GROUPS[key]
        # Check for Ax suited or Kx suited
        if r1 == 14:
            return SKLANSKY_GROUPS.get(('A', 'x', 's'), 9)
        if r1 == 13:
            return SKLANSKY_GROUPS.get(('K', 'x', 's'), 9)
    else:
        key = (r1_sym, r2_sym)
        if key in SKLANSKY_GROUPS:
            return SKLANSKY_GROUPS[key]
    
    return 9  # Unplayable


def chen_formula(card1: Card, card2: Card) -> float:
    """
    Bill Chen's formula for evaluating starting hands.
    Higher score = stronger hand.
    """
    r1, r2 = card1.rank, card2.rank
    suited = card1.suit == card2.suit
    
    # Normalize so r1 >= r2
    if r1 < r2:
        r1, r2 = r2, r1
    
    # Base score is highest card
    def card_score(r):
        if r == 14:  # Ace
            return 10
        elif r == 13:  # King
            return 8
        elif r == 12:  # Queen
            return 7
        elif r == 11:  # Jack
            return 6
        else:
            return r / 2.0
    
    score = card_score(r1)
    
    # Pair bonus
    if r1 == r2:
        score = max(score * 2, 5)
    
    # Suited bonus
    if suited:
        score += 2
    
    # Gap penalty
    gap = r1 - r2 - 1
    if gap == 1:
        score -= 1
    elif gap == 2:
        score -= 2
    elif gap == 3:
        score -= 4
    elif gap >= 4:
        score -= 5
    
    # Straight potential bonus for connected/gapped cards
    if gap <= 1 and r1 < 12:  # Can make straight
        score += 1
    
    return score


def get_chen_score(card1: Card, card2: Card) -> float:
    """Alias for chen_formula for backward compatibility."""
    return chen_formula(card1, card2)


def get_hand_strength_monte_carlo(
    hole_cards: List[Card],
    community_cards: List[Card],
    num_opponents: int,
    num_simulations: int = 1000
) -> float:
    """
    Estimate hand strength using Monte Carlo simulation.
    Returns probability of winning against num_opponents random hands.
    """
    if len(hole_cards) != 2:
        return 0.5

    wins = 0
    ties = 0

    # Get cards that are already dealt
    dealt_cards = set(c.to_int() for c in hole_cards + community_cards)
    available = [i for i in range(52) if i not in dealt_cards]

    rng = np.random.default_rng()

    for _ in range(num_simulations):
        # Shuffle available cards
        sim_deck = available.copy()
        rng.shuffle(sim_deck)

        # Complete the board
        cards_needed = 5 - len(community_cards)
        sim_community = community_cards + [Card.from_int(i) for i in sim_deck[:cards_needed]]
        deck_idx = cards_needed

        # Evaluate hero's hand
        hero_full_hand = hole_cards + sim_community
        hero_hand = evaluate_hand(hero_full_hand)

        # Simulate opponent hands
        beats_all = True
        ties_any = False

        for _ in range(num_opponents):
            if deck_idx + 2 > len(sim_deck):
                break
            opp_hole = [Card.from_int(sim_deck[deck_idx]), Card.from_int(sim_deck[deck_idx + 1])]
            deck_idx += 2

            opp_full_hand = opp_hole + sim_community
            opp_hand = evaluate_hand(opp_full_hand)

            # Compare evaluated results directly
            if hero_hand > opp_hand:
                comparison = 1
            elif hero_hand < opp_hand:
                comparison = -1
            else:
                comparison = 0

            if comparison < 0:  # Opponent wins
                beats_all = False
                break
            elif comparison == 0:  # Tie
                ties_any = True

        if beats_all:
            if ties_any:
                ties += 1
            else:
                wins += 1

    return (wins + 0.5 * ties) / num_simulations if num_simulations > 0 else 0.5


def get_hand_category(hole_cards: List[Card], community_cards: List[Card]) -> int:
    """
    Get hand category (0-9) based on current cards.
    Useful for rule-based strategies.
    """
    if len(community_cards) == 0:
        # Preflop - use Chen formula
        chen = chen_formula(hole_cards[0], hole_cards[1])
        if chen >= 10:
            return 0  # Premium
        elif chen >= 7:
            return 1  # Strong
        elif chen >= 5:
            return 2  # Playable
        elif chen >= 3:
            return 3  # Marginal
        else:
            return 4  # Weak
    
    # Postflop - evaluate actual hand
    hand_rank, _ = evaluate_hand(hole_cards + community_cards)
    return hand_rank


if __name__ == "__main__":
    # Test hand evaluation
    cards = [
        Card.from_str("Ah"),
        Card.from_str("Kh"),
        Card.from_str("Qh"),
        Card.from_str("Jh"),
        Card.from_str("Th"),
    ]
    print(f"Hand: {cards}")
    print(f"Evaluation: {evaluate_hand(cards)}")
    
    # Test Chen formula
    hole = [Card.from_str("As"), Card.from_str("Ks")]
    print(f"\nHole cards: {hole}")
    print(f"Chen score: {chen_formula(hole[0], hole[1])}")
    print(f"Sklansky group: {get_sklansky_group(hole[0], hole[1])}")
