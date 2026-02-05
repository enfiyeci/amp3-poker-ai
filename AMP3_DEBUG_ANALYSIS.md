# AMP3 Training Debug Analysis
**Date**: January 20, 2026

---

## 🔍 Complete System Analysis

### Working Components ✅

1. **AMP3Agent class** (`amp3_network.py`)
   - `__init__()` - ✅ Correct API
   - `predict_opponent_styles()` - ✅ Returns (6, 4) style array
   - `get_action()` - ✅ Returns `(Action, float, float)` = (action, amount, log_prob)
   - `update()` - ✅ Takes batch_size parameter

2. **State Encoding** (`amp3_network.py`)
   - `encode_amp3_state()` - ✅ Converts GameState → AMP3State
   - `state_to_tensors()` - ✅ Converts AMP3State → Dict[str, Tensor]
   - Returns dict with keys: `personal`, `public`, `position`, `action_history`, `style_features`

3. **Data Structures**
   - `AMP3State` - ✅ Dataclass with all fields
   - `Experience` - ✅ Dataclass for replay buffer
   - `ReplayBuffer` - ✅ Working deque-based buffer

---

## ❌ Broken Components & Issues

### Issue 1: ReplayBuffer.push() API Mismatch (CRITICAL)

**Location**: `train_amp3.py` line 711-717

**Problem**:
```python
# Training loop tries to call:
replay_buffer.push(
    trans['state'].to_vector(),    # ❌ Dict doesn't have .to_vector()
    trans['action'],
    reward,
    trans['state'].to_vector(),    # ❌ Same issue
    done
)
```

**Actual ReplayBuffer.push() signature**:
```python
def push(self, experience: Experience):
    # Expects a single Experience object, not 5 separate arguments!
```

**Why it's broken**:
1. `trans['state']` is a Dict[str, Tensor] from `state_to_tensors()`
2. Dicts don't have `.to_vector()` method
3. `push()` expects an `Experience` object, not individual arguments
4. The call signature is completely wrong

---

### Issue 2: Experience Construction Missing

**Problem**: Training loop doesn't create proper `Experience` objects

**What's needed**:
```python
experience = Experience(
    state=trans['state'],              # Dict[str, np.ndarray]
    critic_state=critic_state_dict,    # Dict[str, np.ndarray]
    action=action_idx,                 # int
    reward=reward,                     # float
    next_state=next_state_dict,        # Dict[str, np.ndarray] or None
    next_critic_state=next_critic,     # Dict[str, np.ndarray] or None
    done=done,                         # bool
    log_prob=trans['log_prob']         # float
)
```

**What's missing**:
- No critic_state encoding
- No next_state encoding
- No next_critic_state encoding
- Trying to call non-existent methods

---

### Issue 3: AMP3Agent.update() API Mismatch

**Location**: `train_amp3.py` line 728-732

**Training loop calls**:
```python
amp3_agent.update(
    replay_buffer,              # ❌ Wrong! Passes buffer object
    batch_size=config['amp3_batch_size'],
    entropy_coef=config['amp3_entropy_coef']  # ❌ Doesn't exist
)
```

**Actual update() signature**:
```python
def update(self, batch_size: int = 256) -> Tuple[float, float]:
    # Takes only batch_size, not replay_buffer or entropy_coef!
    # Uses self.replay_buffer internally
```

**Why it's broken**:
1. `update()` doesn't take `replay_buffer` as parameter (it uses `self.replay_buffer`)
2. `update()` doesn't have `entropy_coef` parameter
3. The training loop is passing wrong arguments

---

### Issue 4: Action Type Mismatch

**Location**: Throughout training loop

**Problem**: Confusion between Action enum and action index

```python
# trans['action'] stores Action enum (e.g., Action.CALL)
# But Experience.action expects int (action index 0-3)
# And update() expects action indices, not Action enums
```

**Need to store**: `action_idx` (int) not `action` (Action enum)

---

### Issue 5: Missing Critic State Encoding

**Problem**: Training loop never creates critic states

**What critic needs**:
- All players' hole cards (not just hero's)
- Global view of table
- Different input structure than actor

**Current state**: Not implemented in training loop

---

### Issue 6: Tensor vs NumPy Confusion

**Problem**: Mixing tensor and numpy array formats

**From `state_to_tensors()`**: Returns Dict[str, **Tensor**]
**For `Experience`**: Needs Dict[str, **np.ndarray**]

**Need to**:
1. Convert tensors back to numpy for storage
2. Or store as tensors and handle in Experience

---

## 📋 Complete List of Fixes Needed

### Priority 1: Critical Fixes (Required for Training to Start)

1. **Fix replay_buffer.push() calls**
   - Create proper `Experience` objects
   - Don't call `.to_vector()` on dicts
   - Pass single Experience object to push()

2. **Fix AMP3Agent.update() calls**
   - Remove `replay_buffer` argument
   - Remove `entropy_coef` argument
   - Just pass `batch_size`

3. **Store action_idx not Action enum**
   - Change `trans['action']` to `trans['action_idx']`
   - Keep Action enum separate for env.step()

4. **Convert tensors to numpy for Experience**
   - Add `.cpu().numpy()` when storing states
   - Or modify Experience to accept tensors

### Priority 2: Functionality Fixes (Required for Training to Work)

5. **Implement critic state encoding**
   - Create `encode_critic_state()` function
   - Include all players' information
   - Store in experience

6. **Implement next_state encoding**
   - Encode state at t+1 for TD learning
   - Handle terminal states (None)

7. **Fix calculate_reward() function**
   - Verify it exists and works
   - Returns appropriate reward signal

### Priority 3: Polish (Nice to Have)

8. **Add proper error handling**
9. **Add validation checks**
10. **Improve logging**

---

## 🔧 Root Cause Analysis

### Why These Issues Exist

**Different developers/modules**:
- `amp3_network.py` was written with one API design
- `train_amp3.py` was written assuming a different API
- They were never tested together

**Key disconnects**:
1. **ReplayBuffer API**: Training assumes `push(*args)`, actual is `push(Experience)`
2. **Update API**: Training assumes `update(buffer, ...)`, actual is `update(batch_size)`
3. **State format**: Training doesn't understand Dict[str, Tensor] format
4. **Critic states**: Training doesn't know how to encode them

---

## ✅ What Actually Works

Despite these issues, the core components are solid:

1. ✅ **Actor/Critic networks** - Architecture is correct
2. ✅ **State encoding functions** - Work correctly
3. ✅ **AMP3Agent logic** - Internally consistent
4. ✅ **ReplayBuffer** - Works if used correctly
5. ✅ **get_action()** - Returns correct values

**The problem**: Training loop doesn't use them correctly!

---

## 🎯 Fix Strategy

### Option A: Fix Training Loop (Recommended)

Rewrite the training loop section to properly:
1. Create Experience objects
2. Call ReplayBuffer.push() correctly
3. Call update() correctly
4. Handle state encoding properly

**Estimated time**: 2-3 hours
**Success probability**: High (90%+)

### Option B: Modify AMP3Agent to Match Training Loop

Change AMP3Agent API to match what training expects:
1. Modify ReplayBuffer.push() to accept separate args
2. Modify update() to accept replay_buffer
3. Add entropy_coef parameter

**Estimated time**: 3-4 hours
**Success probability**: Medium (requires changing working code)

---

## 📝 Recommended Next Steps

1. **Fix ReplayBuffer usage** (30 min)
   - Create proper Experience objects
   - Fix push() calls

2. **Fix update() calls** (15 min)
   - Remove wrong parameters
   - Use correct API

3. **Add critic state encoding** (1 hour)
   - Implement encode_critic_state()
   - Add to experience creation

4. **Test with 100 episodes** (30 min)
   - Quick validation run
   - Fix any runtime errors

5. **Full training run** (3-4 hours)
   - 120,000 episodes
   - Monitor convergence

**Total estimated time**: 2-3 hours debugging + 3-4 hours training = **5-7 hours total**

---

## 🚀 Ready to Fix?

The issues are now clearly identified. All are fixable with systematic code changes.

**Most critical**: Fix the replay buffer and update() API mismatches first, then handle critic states.

Would you like me to proceed with implementing these fixes?
