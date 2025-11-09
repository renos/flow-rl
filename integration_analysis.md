# Tutorial Integration Analysis for Craftax Prompts

## Executive Summary
The tutorial provides critical game progression knowledge that should be integrated into the prompt structure to improve skill proposal quality and sequencing. This document outlines specific integration strategies.

## Current State Analysis

### Existing Prompt Structure
1. **next_task**: Proposes skills based on frontier + knowledge base
2. **next_subtask**: Analyzes requirements/consumption/gains
3. **continue_training_decision**: Checks prerequisite readiness
4. **create_skill_densify_reward_reasoning**: Designs reward functions
5. **create_skill_coding**: Implements task_is_done/task_reward/task_network_number

### What's Missing
- **Floor-specific progression paths** - No guidance on what skills make sense at which floor
- **Ladder mechanics** - 8-creature requirement not mentioned
- **Damage type system** - Physical/fire/ice damage and resistances
- **Resource availability** - What resources exist on which floors
- **Intrinsics management** - Survival mechanics (hunger, thirst, energy, sleeping)
- **Tool/equipment progression** - Natural upgrade paths

## Integration Strategy

### 1. Enhance the Knowledge Base Section

The `$db.knowledge_base$` should be pre-populated with tutorial information structured as:

```markdown
## Game Progression Overview
- 9 floors total (0-8): Overworld → Dungeon → Gnomish Mines → Sewers → Vaults → Troll Mines → Fire Realm → Ice Realm → Graveyard
- Each ladder requires killing 8 creatures on the current floor to open (except Floor 0)
- Player gains 1 experience point per floor descended (can upgrade Dexterity, Strength, or Intelligence)

## Tool Progression
- **Pickaxe tiers**: wood (1) → stone (2) → iron (3) → diamond (4)
- **Sword tiers**: wood (1) → stone (2) → iron (3) → diamond (4)
- **Crafting prerequisites**:
  - Wood tools: crafting_table + wood
  - Stone tools: crafting_table + wood + stone
  - Iron tools: crafting_table + furnace + wood + iron + coal (both must be adjacent)
  - Diamond tools: crafting_table + diamonds

## Floor-Specific Information

### Floor 0: Overworld
**Resources**: trees (wood), stone, coal, iron, grass (seeds), cows (food), lakes (water)
**Enemies**: zombies, skeletons
**Key objectives**:
  - Craft wood tools → stone tools → iron tools
  - Gather food/water supplies
  - Find ladder down
**Special notes**: Only source of abundant wood; collect extra before descending

### Floor 1: Dungeon
**Layout**: Rooms connected by paths
**Resources**: fountains (water), chests (first contains bow), snails (food)
**Enemies**: orc warriors, orc mages (8 kills to open ladder)
**Key objectives**:
  - Find and open first chest (contains bow)
  - Craft arrows (wood + stone)
  - Kill 8 orcs using bow
**Combat**: Bow is critical for safe ranged combat

### Floor 2: Gnomish Mines
**Environment**: Dark (requires torches for vision)
**Resources**: rich ore deposits (coal, iron, diamonds, sapphires, rubies), water pools, bats (food)
**Enemies**: gnomes (stronger than orcs, dangerous in open spaces)
**Key objectives**:
  - Craft and place torches (wood + coal)
  - Mine diamonds (need diamond pickaxe: 3 diamonds)
  - Mine sapphires/rubies (need diamond pickaxe)
  - Craft diamond tools and armor
**Combat**: Avoid being surrounded; use terrain

### Floor 3: Sewers
**Layout**: Dungeon-like with water patches
**Resources**: water patches (must fill with stone), chests (first contains spell book)
**Enemies**: lizards (very dangerous, swim through water), kobolds (throw daggers)
**Key mechanics introduced**:
  - **Spellcasting**: First chest contains book → learn fireball or iceball (costs 2 mana)
  - **Damage types**: Spells do fire/ice damage (not physical)
  - **Ice enchantment table**: Can enchant sword/bow/armor with sapphires (9 mana cost)
**Combat**: High-damage enemies; consider enchantments

### Floor 4: Vaults
**Layout**: Dungeon-like
**Resources**: chests (another spell book), fire enchantment table
**Enemies**: knights, archers (armored - 50% physical resistance)
**Key mechanics**:
  - **Fire enchantment table**: Enchant with rubies (9 mana cost)
  - **Enchantment effects**:
    - Weapons: +50% damage of enchantment type
    - Armor: -20% damage reduction per piece for that type
**Combat**: Physical damage halved; use spells or enchantments

### Floor 5: Troll Mines
**Environment**: Dark caverns (richest ore deposits)
**Resources**: abundant diamonds/sapphires/rubies, water (with deep things)
**Enemies**: trolls (strong, high damage), deep things (weak if hit, dangerous ranged attacks)
**Key objectives**: Craft full diamond armor
**Combat**: Trolls are formidable; armor recommended

### Floor 6: Fire Realm
**Environment**: Islands separated by lava, NO WATER SOURCE
**Resources**: abundant coal and rubies
**Enemies**: pig men, fire elementals (100% fire resistance, high physical resistance)
**Key mechanics**:
  - Build bridges across lava (place + mine stone)
  - Must return to Floor 5 for water periodically
  - **Requires ice damage** to kill enemies effectively
**Combat**: Ice spells or ice enchantments essential

### Floor 7: Ice Realm
**Environment**: Dark level, NO FOOD SOURCE
**Resources**: abundant sapphires and rubies
**Enemies**: frost trolls, ice elementals (strongest in game, require fire damage)
**Key mechanics**:
  - Must return to Floor 6 for food periodically
  - **Requires fire damage** to kill enemies
**Combat**: Fire spells or fire enchantments essential

### Floor 8: Graveyard (Boss Level)
**Environment**: No ladder out, intrinsics don't decay
**Resources**: None
**Enemies**: Necromancer (final boss) + summoned waves
**Mechanics**:
  - Necromancer summons waves corresponding to each previous floor
  - Between waves, necromancer becomes vulnerable
  - Attack vulnerable necromancer to trigger next wave
  - Final wave (ice realm) → defeat → attack necromancer → win game

## Damage Type System
- **Physical damage**: Default for melee weapons and bow
  - Reduced by armor
  - Heavily resisted on Floors 4+ enemies
- **Fire damage**: From fireball spell or fire enchantments
  - Essential for ice enemies (Floor 7)
  - Useless against fire enemies (Floor 6)
- **Ice damage**: From iceball spell or ice enchantments
  - Essential for fire enemies (Floor 6)
  - Useless against ice enemies (Floor 7)

## Attribute System (1 point per floor descended, max level 5)
- **Dexterity**: ↑ max food/water/energy, ↓ decay rate, ↑ bow damage
- **Strength**: ↑ melee damage, ↑ max health
- **Intelligence**: ↑ max mana, ↓ mana decay, ↑ spell damage, ↑ enchantment effectiveness

## Intrinsics System
- **Health**: Recovers when hunger/thirst/energy > 0; decreases when any = 0; death at 0
- **Hunger**: Decreases over time; replenish by eating (cows, plants, bats, snails)
- **Thirst**: Decreases over time; replenish from water, fountains
- **Energy**: Decreases over time; replenish by sleeping
- **Mana**: Used for spells (2) and enchanting (9); naturally recovers

## Survival Tips
- **Sleeping safely**: Block yourself in with stone before sleeping (vulnerable while asleep)
- **Resting**: Execute no-ops until intrinsic hits 0, attacked, or full health
- **Potion system**: 6 colors (red, green, blue, pink, cyan, yellow)
  - Each gives +8 or -3 to health/mana/energy
  - Effects randomized each game (trial and error required)
```

### 2. Modify the `next_task` Prompt

Add a new section after the frontier summary:

```markdown
## Tutorial Progression Context
Based on the tutorial, consider the natural game progression:

**Current Floor Analysis:**
- What floor is the frontier currently on? (check level:player_level)
- What resources are available on this floor vs previous floors?
- What are the known challenges for this floor?

**Skill Sequencing Priorities:**
1. **Survival skills**: Food, water, energy management (always relevant)
2. **Tool progression**: Upgrade tools to access new resources
3. **Combat preparation**: Weapons/armor appropriate for current floor enemies
4. **Floor-specific mechanics**: Torches (Floor 2+), spells (Floor 3+), enchantments (Floor 3+)
5. **Ladder descent**: Kill 8 creatures → open ladder → descend

**Damage Type Considerations:**
- Floors 0-3: Physical damage viable
- Floor 4+: Need elemental damage (fire/ice)
- Floor 6: Ice damage required (fire enemies)
- Floor 7: Fire damage required (ice enemies)

When proposing skills, ensure they:
- Match the current floor's available resources
- Prepare for known upcoming challenges
- Follow logical progression (don't skip required prerequisites)
- Consider damage type requirements for combat skills
```

### 3. Add Floor-Aware Validation to `next_subtask` Prompt

After the "Task Analysis" section:

```markdown
## Floor Feasibility Check
- Can the required resources for this skill be obtained at the current frontier's floor level?
- If the skill involves combat, does the frontier have appropriate damage types?
- If the skill involves crafting, are the necessary crafting stations reachable?

Example validations:
- Diamond tools require Floor 2+ (diamond availability)
- Spell skills require Floor 3+ (book availability)
- Enchantment skills require Floor 3+ (enchantment tables)
- Ice damage skills require Floor 6 combat
- Fire damage skills require Floor 7 combat
```

### 4. Enhance Reward Design with Floor Context

In `create_skill_densify_reward_reasoning`, add:

```markdown
## Floor-Specific Dense Reward Guidance

Consider floor-appropriate dense rewards:
- **Resource gathering**: Distance to target resource blocks
- **Combat skills**:
  - Distance to appropriate enemies
  - Health management (avoid damage)
  - For armored enemies (Floor 4+): prioritize elemental damage
- **Ladder descent**: Progress toward killing 8 creatures
- **Exploration**: Movement toward unexplored areas (fountains, chests, ladders)
- **Survival**: Maintaining healthy intrinsics levels
```

### 5. Knowledge Base Bootstrapping

Create an initial knowledge base file that the system loads with the tutorial content structured as shown above. This ensures the LLM always has access to game mechanics without using tokens on repetition.

## Implementation Priority

1. **High Priority**: Add tutorial content to knowledge base (Sections 1)
2. **High Priority**: Modify `next_task` prompt with progression context (Section 2)
3. **Medium Priority**: Add floor feasibility checks to `next_subtask` (Section 3)
4. **Low Priority**: Enhance reward design guidance (Section 4)

## Expected Benefits

1. **Better skill sequencing**: Skills follow natural game progression
2. **Fewer invalid proposals**: Floor-aware validation prevents impossible skills
3. **Improved combat skills**: Understanding damage types and enemy resistances
4. **Resource efficiency**: Skills gather appropriate resources for current floor
5. **Clearer requirements**: Tutorial provides ground truth for prerequisites
6. **Faster convergence**: Following the tutorial path reduces trial-and-error

## Example Improved Skill Proposals

### Before (Frontier Only)
- Might propose "Craft Diamond Armor" when still on Floor 0 (no diamonds)
- Might propose "Kill Fire Elemental" without ice damage capability
- Might propose "Descend to Floor 2" without killing 8 Floor 1 creatures

### After (Frontier + Tutorial)
- Proposes "Gather Wood Extensively" on Floor 0 (tutorial warns it's rare later)
- Proposes "Learn Ice Spell" before "Enter Fire Realm"
- Proposes "Kill 8 Orcs" explicitly before "Descend to Floor 2"
- Proposes "Craft Full Diamond Armor" specifically on Floor 5 (richest ores)

## Testing Strategy

1. **Baseline comparison**: Run skill generation with/without tutorial integration
2. **Validate skill order**: Check if generated skills follow tutorial progression
3. **Measure failure rate**: Track how often proposed skills have unsatisfiable requirements
4. **Human evaluation**: Compare generated skill sequences to expert tutorial path
