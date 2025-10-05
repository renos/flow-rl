# Craftax Classic Achievement Progression Guide

This guide details the exact requirements for each achievement in Craftax Classic, including prerequisites, resource costs, and location requirements.

## Basic Resource Collection

### COLLECT_WOOD (Achievement #0)
- **Action**: Use DO action next to a tree
- **Requirements**: None (can mine trees with bare hands)
- **Location**: Next to tree blocks
- **Resources gained**: +1 wood
- **Notes**: Essential first step for all progression

### COLLECT_SAPLING (Achievement #3)
- **Action**: Use DO action on grass blocks
- **Requirements**: None
- **Location**: Any grass block
- **Success rate**: 10% chance per grass block interaction
- **Resources gained**: +1 sapling
- **Notes**: Random drop, may take multiple attempts

### COLLECT_DRINK (Achievement #4)
- **Action**: Use DO action next to water
- **Requirements**: None
- **Location**: Next to water blocks
- **Resources gained**: +1 drink, reduces thirst to 0
- **Notes**: Essential for survival

## Crafting Infrastructure

### PLACE_TABLE (Achievement #1)
- **Action**: PLACE_TABLE action
- **Requirements**:
  - 2 wood minimum in inventory
  - Standing next to a valid placement location (not in wall)
- **Cost**: 2 wood
- **Notes**: Required for all crafting recipes

### PLACE_FURNACE (Achievement #16)
- **Action**: PLACE_FURNACE action
- **Requirements**:
  - 1 stone minimum in inventory
  - Standing next to a valid placement location (not in wall)
- **Cost**: 1 stone
- **Prerequisites**: Must have stone (requires wood pickaxe to mine)
- **Notes**: Required for iron-tier crafting

### PLACE_STONE (Achievement #10)
- **Action**: PLACE_STONE action
- **Requirements**:
  - 1 stone minimum in inventory
  - Valid placement location (can place on water or non-solid blocks)
- **Cost**: 1 stone
- **Prerequisites**: Must have stone (requires wood pickaxe to mine)

## Tool Progression

### MAKE_WOOD_PICKAXE (Achievement #5)
- **Action**: MAKE_WOOD_PICKAXE action
- **Requirements**:
  - 1 wood minimum in inventory
  - Standing next to a crafting table
- **Cost**: 1 wood
- **Prerequisites**: PLACE_TABLE achievement
- **Notes**: Unlocks stone and coal mining

### MAKE_STONE_PICKAXE (Achievement #13)
- **Action**: MAKE_STONE_PICKAXE action
- **Requirements**:
  - 1 wood AND 1 stone minimum in inventory
  - Standing next to a crafting table
- **Cost**: 1 wood + 1 stone
- **Prerequisites**:
  - MAKE_WOOD_PICKAXE (to mine stone)
  - COLLECT_STONE achievement
- **Notes**: Unlocks iron mining

### MAKE_IRON_PICKAXE (Achievement #20)
- **Action**: MAKE_IRON_PICKAXE action
- **Requirements**:
  - 1 wood AND 1 stone AND 1 iron AND 1 coal minimum in inventory
  - Standing next to BOTH a crafting table AND a furnace
- **Cost**: 1 wood + 1 stone + 1 iron + 1 coal
- **Prerequisites**:
  - MAKE_STONE_PICKAXE (to mine iron)
  - COLLECT_COAL and COLLECT_IRON achievements
  - PLACE_FURNACE achievement
- **Notes**: Unlocks diamond mining

## Weapon Progression

### MAKE_WOOD_SWORD (Achievement #6)
- **Action**: MAKE_WOOD_SWORD action
- **Requirements**:
  - 1 wood minimum in inventory
  - Standing next to a crafting table
- **Cost**: 1 wood
- **Damage**: 2 (vs 1 bare handed)
- **Prerequisites**: PLACE_TABLE achievement

### MAKE_STONE_SWORD (Achievement #14)
- **Action**: MAKE_STONE_SWORD action
- **Requirements**:
  - 1 wood AND 1 stone minimum in inventory
  - Standing next to a crafting table
- **Cost**: 1 wood + 1 stone
- **Damage**: 3
- **Prerequisites**: COLLECT_STONE achievement

### MAKE_IRON_SWORD (Achievement #21)
- **Action**: MAKE_IRON_SWORD action
- **Requirements**:
  - 1 wood AND 1 stone AND 1 iron AND 1 coal minimum in inventory
  - Standing next to BOTH a crafting table AND a furnace
- **Cost**: 1 wood + 1 stone + 1 iron + 1 coal
- **Damage**: 5
- **Prerequisites**: Same as iron pickaxe

## Advanced Mining

### COLLECT_STONE (Achievement #9)
- **Action**: Use DO action next to stone blocks
- **Requirements**: Must have wood pickaxe equipped
- **Location**: Next to stone blocks
- **Resources gained**: +1 stone
- **Prerequisites**: MAKE_WOOD_PICKAXE achievement

### COLLECT_COAL (Achievement #17)
- **Action**: Use DO action next to coal blocks
- **Requirements**: Must have wood pickaxe equipped
- **Location**: Next to coal blocks
- **Resources gained**: +1 coal
- **Prerequisites**: MAKE_WOOD_PICKAXE achievement

### COLLECT_IRON (Achievement #18)
- **Action**: Use DO action next to iron blocks
- **Requirements**: Must have stone pickaxe equipped
- **Location**: Next to iron blocks
- **Resources gained**: +1 iron
- **Prerequisites**: MAKE_STONE_PICKAXE achievement

### COLLECT_DIAMOND (Achievement #19)
- **Action**: Use DO action next to diamond blocks
- **Requirements**: Must have iron pickaxe equipped
- **Location**: Next to diamond blocks
- **Resources gained**: +1 diamond
- **Prerequisites**: MAKE_IRON_PICKAXE achievement

## Farming & Food

### PLACE_PLANT (Achievement #7)
- **Action**: PLACE_PLANT action
- **Requirements**:
  - 1 sapling minimum in inventory
  - Standing next to grass block for placement
- **Cost**: 1 sapling
- **Prerequisites**: COLLECT_SAPLING achievement
- **Notes**: Plant grows over 600 timesteps to become ripe

### EAT_PLANT (Achievement #11)
- **Action**: Use DO action next to ripe plant
- **Requirements**: Plant must be fully grown (ripe)
- **Location**: Next to ripe plant blocks
- **Benefits**: +4 food, hunger reset to 0
- **Notes**: Reverts ripe plant back to regular plant

## Combat

### EAT_COW (Achievement #2)
- **Action**: Attack cow with DO action until it dies
- **Requirements**: None (can kill with bare hands)
- **Combat**: Cow has 3 health
- **Benefits**: +6 food, hunger reset to 0
- **Notes**: Cows are passive mobs

### DEFEAT_ZOMBIE (Achievement #8)
- **Action**: Attack zombie with DO action until it dies
- **Requirements**: None, but weapons recommended
- **Combat**: Zombie has 5 health, deals 2 damage (7 while sleeping)
- **Notes**: Zombies are aggressive, attack cooldown of 5 turns

### DEFEAT_SKELETON (Achievement #12)
- **Action**: Attack skeleton with DO action until it dies
- **Requirements**: None, but weapons strongly recommended
- **Combat**: Skeleton has 3 health, shoots arrows (2 damage), attack cooldown of 4 turns
- **Notes**: Skeletons maintain distance and shoot arrows

## Rest & Survival

### WAKE_UP (Achievement #15)
- **Action**: Automatic when energy reaches 9 while sleeping OR when attacked while sleeping
- **Requirements**: Must be sleeping (use SLEEP action when energy < 9)
- **Notes**: Sleeping restores energy faster but makes you vulnerable

## Complete Progression Paths

### Basic Survival Path:
1. COLLECT_WOOD → 2. PLACE_TABLE → 3. MAKE_WOOD_PICKAXE → 4. COLLECT_STONE

### Combat Path:
1. MAKE_WOOD_SWORD → 2. DEFEAT_ZOMBIE/EAT_COW → 3. MAKE_STONE_SWORD → 4. DEFEAT_SKELETON

### Advanced Crafting Path:
1. COLLECT_COAL → 2. PLACE_FURNACE → 3. MAKE_STONE_PICKAXE → 4. COLLECT_IRON → 5. MAKE_IRON_PICKAXE → 6. COLLECT_DIAMOND

### Farming Path:
1. COLLECT_SAPLING → 2. PLACE_PLANT → 3. Wait 600 timesteps → 4. EAT_PLANT

## Resource Requirements Summary

| Achievement | Wood | Stone | Coal | Iron | Diamond | Sapling | Stations Required |
|-------------|------|-------|------|------|---------|---------|-------------------|
| Place Table | 2 | 0 | 0 | 0 | 0 | 0 | None |
| Wood Pickaxe | 1 | 0 | 0 | 0 | 0 | 0 | Crafting Table |
| Wood Sword | 1 | 0 | 0 | 0 | 0 | 0 | Crafting Table |
| Stone Pickaxe | 1 | 1 | 0 | 0 | 0 | 0 | Crafting Table |
| Stone Sword | 1 | 1 | 0 | 0 | 0 | 0 | Crafting Table |
| Place Furnace | 0 | 1 | 0 | 0 | 0 | 0 | None |
| Iron Pickaxe | 1 | 1 | 1 | 1 | 0 | 0 | Crafting Table + Furnace |
| Iron Sword | 1 | 1 | 1 | 1 | 0 | 0 | Crafting Table + Furnace |
| Place Plant | 0 | 0 | 0 | 0 | 0 | 1 | None |

## Mining Tool Requirements

| Block | Tool Required | Notes |
|-------|---------------|-------|
| Tree | None | Can mine with bare hands |
| Stone | Wood Pickaxe | Unlocked after crafting wood pickaxe |
| Coal | Wood Pickaxe | Same as stone |
| Iron | Stone Pickaxe | Requires stone pickaxe upgrade |
| Diamond | Iron Pickaxe | Highest tier requirement |

This progression system requires careful resource management and planning to achieve all 22 achievements efficiently.