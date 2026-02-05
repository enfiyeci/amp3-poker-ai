# AMP3 Opponent Style Classification

## Overview

AMP3 includes real-time opponent modeling that classifies players into distinct playing styles. During the demo, you'll see AMP3 detect and adapt to different opponent types.

## 6 Opponent Playing Styles

### 1. **Tight-Passive**
- **Description**: Plays few hands, rarely raises
- **Behavior**: Very selective about entering pots, prefers calling to raising
- **AMP3 Adaptation**: Can apply more pressure with weaker hands, steal more often

### 2. **Tight-Aggressive**
- **Description**: Selective but aggressive when playing
- **Behavior**: Only plays premium hands but bets/raises with them
- **AMP3 Adaptation**: Respects their raises, folds marginal hands against aggression

### 3. **Loose-Passive**
- **Description**: Plays many hands, calls often
- **Behavior**: Enters many pots, likes to call and see cards
- **AMP3 Adaptation**: Value bets more thinly, bluffs less frequently

### 4. **Loose-Aggressive**
- **Description**: Plays many hands, raises frequently
- **Behavior**: Very active, applies constant pressure
- **AMP3 Adaptation**: Tightens up, waits for strong hands to trap with

### 5. **Balanced**
- **Description**: Mix of strategies, adapts to situation
- **Behavior**: No clear exploitable pattern, well-rounded play
- **AMP3 Adaptation**: Uses game theory optimal (GTO) approach

### 6. **Aggressive**
- **Description**: High raise frequency, applies pressure
- **Behavior**: Constantly betting and raising, forces decisions
- **AMP3 Adaptation**: Increases calling frequency, uses opponent's aggression against them

## How AMP3 Uses This Information

During each decision, AMP3:
1. **Observes** opponent betting patterns and hand frequencies
2. **Classifies** the opponent into one of the 6 styles
3. **Adapts** its strategy to exploit opponent weaknesses
4. **Adjusts** action probabilities based on opponent profile

## In the Demo

You'll see output like:
```
→ Processing opponent tendencies...
→ Opponent Style Detected: Tight-Aggressive
→ Calculating optimal strategy...

AMP3 DECISION:
  Recommended Action: FOLD
  Confidence: 82.3%

  Opponent Profile: Tight-Aggressive
  Tendency: Selective but aggressive when playing
```

This shows AMP3's opponent modeling in action - understanding who it's playing against and making strategic adjustments accordingly.

## Why This Matters

- **Adaptability**: One of poker's core skills is adjusting to opponents
- **Exploitation**: Different strategies work against different player types
- **Real-world Applicability**: Professional poker requires opponent modeling
- **AI Sophistication**: Shows AMP3 goes beyond just evaluating hand strength
