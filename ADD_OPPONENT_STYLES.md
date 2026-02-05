# How to Add More Opponent Styles to OSM

## Current Setup

You currently have **22 opponent styles**:
- 7 Random strategies
- 5 Sklansky-based (Conservative, Regular, Aggressive, Bluffing, Deceptive)
- 5 Chen-based (same 5 types)
- 5 Rule-based (same 5 types)

## Option 1: Add More Player Types (Easiest)

Add more player archetypes to `style_library.py`:

### Step 1: Expand PlayerType Enum

Find this section in `style_library.py` (around line 24):

```python
class PlayerType(IntEnum):
    """Player types from the paper"""
    CONSERVATIVE = 0  # Bet only with very strong hands
    REGULAR = 1       # Bet with strong hands, fold with weak
    AGGRESSIVE = 2    # Bet with slightly strong hands
    BLUFFING = 3      # Sometimes bet with weak hands
    DECEPTIVE = 4     # Call with strong hands to trap
```

**Add new types:**

```python
class PlayerType(IntEnum):
    """Expanded player types"""
    CONSERVATIVE = 0   # Bet only with very strong hands
    REGULAR = 1        # Bet with strong hands, fold with weak
    AGGRESSIVE = 2     # Bet with slightly strong hands
    BLUFFING = 3       # Sometimes bet with weak hands
    DECEPTIVE = 4      # Call with strong hands to trap

    # NEW ADDITIONS
    MANIAC = 5         # Extremely aggressive, raises constantly
    ROCK = 6           # Ultra tight, only plays premium hands
    CALLING_STATION = 7 # Rarely folds, calls too much
    TAG = 8            # Tight-Aggressive (classic strong player)
    LAG = 9            # Loose-Aggressive (dangerous player)
    WEAK_TIGHT = 10    # Plays few hands, folds to aggression
    WILD = 11          # Unpredictable, mixes all styles
```

### Step 2: Update Strategy Thresholds

For each strategy class (Sklansky, Chen, RuleBased), add thresholds for new types.

**Example for SklanskyStrategy** (around line 200):

```python
TYPE_THRESHOLDS = {
    PlayerType.CONSERVATIVE: 3,
    PlayerType.REGULAR: 5,
    PlayerType.AGGRESSIVE: 7,
    PlayerType.BLUFFING: 8,
    PlayerType.DECEPTIVE: 6,

    # ADD NEW ONES
    PlayerType.MANIAC: 9,        # Plays almost any hand
    PlayerType.ROCK: 2,          # Only best hands
    PlayerType.CALLING_STATION: 8,
    PlayerType.TAG: 4,           # Tight range but aggressive
    PlayerType.LAG: 7,
    PlayerType.WEAK_TIGHT: 3,
    PlayerType.WILD: 6,
}
```

**Example for ChenStrategy** (around line 330):

```python
TYPE_THRESHOLDS = {
    PlayerType.CONSERVATIVE: 8.0,
    PlayerType.REGULAR: 6.0,
    PlayerType.AGGRESSIVE: 4.0,
    PlayerType.BLUFFING: 3.0,
    PlayerType.DECEPTIVE: 5.0,

    # ADD NEW ONES
    PlayerType.MANIAC: 0.0,         # Plays everything
    PlayerType.ROCK: 12.0,          # Very high threshold
    PlayerType.CALLING_STATION: 2.0,
    PlayerType.TAG: 9.0,
    PlayerType.LAG: 5.0,
    PlayerType.WEAK_TIGHT: 10.0,
    PlayerType.WILD: 4.0,
}
```

### Step 3: Update RuleBased Strategy

For RuleBasedStrategy, you'd need to add different betting behaviors for each new type.

This is more complex - you'd modify the `get_action` method to have different logic per player type.

## Option 2: Add Completely New Strategy Types

Create entirely new strategy classes:

### GTO Strategy (Game Theory Optimal)

```python
class GTOStrategy(BasePokerStrategy):
    """
    Game Theory Optimal strategy - balanced, unexploitable
    """

    def __init__(self, player_type: PlayerType):
        super().__init__(f"GTO_{player_type.name}", player_type)

    def get_action(self, state, seat, hole_cards, community_cards):
        # Implement balanced GTO-style play
        # Mix of betting and checking with optimal frequencies
        hand_strength = self._evaluate_hand_strength(hole_cards, community_cards)

        if hand_strength > 0.8:
            # Strong hands: mostly bet, sometimes check
            return (Action.BET, state.pot * 0.66) if self.rng.random() < 0.75 else (Action.CALL, 0)
        elif hand_strength > 0.5:
            # Medium hands: balanced mix
            if self.rng.random() < 0.4:
                return (Action.BET, state.pot * 0.5)
            else:
                return (Action.CALL, 0)
        else:
            # Weak hands: mostly fold, sometimes bluff
            return (Action.FOLD, 0) if self.rng.random() < 0.85 else (Action.BET, state.pot * 0.3)
```

### Position-Aware Strategy

```python
class PositionalStrategy(BasePokerStrategy):
    """
    Strategy that heavily considers table position
    """

    def __init__(self, player_type: PlayerType):
        super().__init__(f"Positional_{player_type.name}", player_type)

    def get_action(self, state, seat, hole_cards, community_cards):
        # Get position (0=button, 5=UTG)
        position = (seat - state.button_seat) % state.num_players

        # Play tighter in early position, looser in late position
        hand_strength = self._evaluate_hand_strength(hole_cards, community_cards)

        if position <= 2:  # Early position - play tight
            threshold = 0.7
        elif position <= 4:  # Middle position
            threshold = 0.5
        else:  # Late position - play loose
            threshold = 0.3

        if hand_strength > threshold:
            return (Action.BET, state.pot * 0.75)
        else:
            return (Action.FOLD, 0)
```

### Exploitative Strategy

```python
class ExploitativeStrategy(BasePokerStrategy):
    """
    Adapts to opponent tendencies
    """

    def __init__(self, player_type: PlayerType):
        super().__init__(f"Exploitative_{player_type.name}", player_type)
        self.opponent_fold_freq = {}  # Track opponent fold frequencies

    def get_action(self, state, seat, hole_cards, community_cards):
        hand_strength = self._evaluate_hand_strength(hole_cards, community_cards)

        # Check how often opponents fold
        avg_fold_freq = np.mean(list(self.opponent_fold_freq.values())) if self.opponent_fold_freq else 0.5

        # If opponents fold a lot, bluff more
        if avg_fold_freq > 0.6:
            bluff_threshold = 0.3
        else:
            bluff_threshold = 0.8

        if hand_strength > 0.6 or (hand_strength < bluff_threshold and self.rng.random() < avg_fold_freq):
            return (Action.BET, state.pot * 0.8)
        elif hand_strength > 0.4:
            return (Action.CALL, 0)
        else:
            return (Action.FOLD, 0)
```

## Option 3: Add Hybrid Strategies

Combine existing strategies:

```python
class HybridStrategy(BasePokerStrategy):
    """
    Switches between strategies based on game state
    """

    def __init__(self, player_type: PlayerType):
        super().__init__(f"Hybrid_{player_type.name}", player_type)
        self.tight_strategy = SklanskyStrategy(PlayerType.CONSERVATIVE)
        self.loose_strategy = RandomStrategy(5)

    def get_action(self, state, seat, hole_cards, community_cards):
        # Play tight when short-stacked, loose when deep
        my_stack = state.stacks[seat]
        avg_stack = np.mean(state.stacks)

        if my_stack < avg_stack * 0.5:
            # Short stack - play tight
            return self.tight_strategy.get_action(state, seat, hole_cards, community_cards)
        else:
            # Deep stack - play loose
            return self.loose_strategy.get_action(state, seat, hole_cards, community_cards)
```

## Update StyleLibrary to Include New Strategies

After creating new strategy classes, add them to `StyleLibrary._build_library()`:

```python
def _build_library(self):
    """Build all strategies"""

    # Existing strategies...
    for config_id in range(7):
        strategy = RandomStrategy(config_id)
        self.strategies[strategy.style_name] = strategy

    for player_type in PlayerType:
        self.strategies[SklanskyStrategy(player_type).style_name] = SklanskyStrategy(player_type)
        self.strategies[ChenStrategy(player_type).style_name] = ChenStrategy(player_type)
        self.strategies[RuleBasedStrategy(player_type).style_name] = RuleBasedStrategy(player_type)

        # ADD NEW STRATEGY TYPES HERE
        self.strategies[GTOStrategy(player_type).style_name] = GTOStrategy(player_type)
        self.strategies[PositionalStrategy(player_type).style_name] = PositionalStrategy(player_type)
        self.strategies[ExploitativeStrategy(player_type).style_name] = ExploitativeStrategy(player_type)

    print(f"Built style library with {len(self.strategies)} strategies")
```

## How Many Styles Will You Have?

**Option 1** (Expand PlayerType to 12 types):
- 7 Random + (12 × 3 strategy types) = **43 total styles**

**Option 2** (Add 3 new strategy classes):
- 7 Random + (5 types × 6 strategy classes) = **37 total styles**

**Both Options Combined**:
- 7 Random + (12 types × 6 strategy classes) = **79 total styles**

## Recommendation

Start with **Option 1** - it's easier and gives you more variety:

1. Add MANIAC, ROCK, CALLING_STATION, TAG, LAG player types
2. Update thresholds in existing strategy classes
3. This gives you **37 total styles** (up from 22)

Then later, if you want even more variety, add new strategy classes (Option 2).

## After Adding Styles

You'll need to retrain OSM to recognize the new styles:

```bash
# Regenerate style features with new styles
python3 train_amp3.py --stage osm
```

This will take several hours but will teach AMP3 to recognize all the new opponent types!
