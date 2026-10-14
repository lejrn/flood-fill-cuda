---
title: Ball screw - Brilliant Mechanism Rotary to Linear Motion
source: Brilliant Mechanism Rotary to Linear Motion.mp4
duration: 00:36
final_invention: Ball screw (recirculating ball screw)
core_function: convert rotation into straight-line motion
chain_type: evolution chain
rungs: 4
---

# Ball screw: hold the nut so turning becomes travel, roll it on balls, then give the balls a way back

**Chain at a glance:** Turn a nut along a bolt -> Hold nut, turn the bolt -> Balls between the threads -> Return tube recirculates balls

**The simple problem:** Not stated up front; implied by a nut travelling along a bolt and named at [00:10]: turn spinning into straight-line movement.

## The ladder

### 1. Turn the nut along the bolt [00:00]
- **Idea:** The everyday case: a turned nut walks along the bolt.
- **Why it works:** The threads are a helix; each turn advances the nut one thread.
- **But:** (implied) The travelling part also spins, so it cannot push a load straight. - `control`
- **So:** Hold the nut so it cannot rotate; turn the bolt instead. [00:00] - `constrain`
- **Shown as:** Desk-level close-up of a hex nut sliding along a bolt held in a small gripper.
![Nut travelling along the bolt](frames/ball-screw/f001_0-00-02.jpg)

### 2. Hold the nut, turn the bolt (plain screw drive) [00:05]
- **Idea:** A gripper clamps the nut while a hand twists the bolt.
- **Why it works:** Engaged threads plus a nut that cannot turn leave only sliding: rotation becomes straight travel. [00:10]
- **But:** (implied) Thread faces rub, so it is stiff; balls are promised to make the nut "move more easily". [00:14] - `side-effect`
- **So:** Put small balls between the threads so they roll instead of slide. [00:14] - `substitute-medium`
- **Shown as:** The engineer, seen only as torso and hand, twists the bolt while the gripper holds the nut.
![Hand turns the bolt while the nut is held](frames/ball-screw/f002_0-00-06.jpg)

### 3. Balls in the thread groove [00:14]
- **Idea:** A row of steel balls rides in the helical groove between screw and nut.
- **Why it works:** Rolling has far less friction than sliding.
- **But:** As the nut moves, balls reaching the end of the groove fall out. [00:20] - `consumable-waste`
- **So:** Connect a tube from the groove's end back to its start so balls keep circulating. [00:24] - `close-the-loop`
- **Shown as:** Cutaway nut with chrome balls in the groove; the camera moves to the open groove end where they would spill.
![Balls reaching the end of the groove](frames/ball-screw/f007_0-00-24.jpg)

### Final: Ball screw with a return tube [00:24]
- **Idea:** Balls roll along the groove, pass through the return tube and re-enter, endlessly, while the held nut travels along the turning screw.
- **Why it solves the chain:** Keeps the held nut (1) and rolling contact (2); the tube stops balls escaping (3).
- **Reveal:** A long real ball screw with a flanged nut at "This is how the ball screw ... was created" [00:30], ending on the screw driving a carriage between two guide rails.
![Real ball screw with flanged nut](frames/ball-screw/f009_0-00-32.jpg)

## Scar map
Every part of the final design and the failure that forced it.
| Part of the final design | Exists because of | Rung |
|---|---|---|
| Nut held against rotation (gripper in the demo; carriage on rails at the end) | The travelling part also spun | 1 |
| Steel balls in a helical groove | Sliding threads are stiff (implied) | 2 |
| Return tube from groove end to start | Balls fell out of the groove | 3 |

## How the video tells it
- **Hook (first ~5 s):** Familiar hardware, a nut moving along a bolt, then at once the twist: hold the nut and turn the bolt.
- **Storyteller:** "We / let's" for the experiments, "engineers" for the fix. Connectives: Now, let's / What happens if / If we / But there was a problem / So engineers / This is how.
- **Visual devices:** Desk-scale nut and bolt; a mechanical gripper as the visible constraint; engineer only as torso and hand (no thinking close-up); cutaway nut with chrome balls in macro; real product and machine context (rails) at the end. No colour coding. No sampled frame shows the return tube; it appears between the 00:24 and 00:31 frames.
- **Pacing:** 4 rungs in 00:36, about 9 s per rung; two of three flaws are only implied.

## Accuracy notes
- Real ball screws return balls via an external tube (as shown), internal deflectors or end caps; sliding lead screws remain common where low cost or self-locking matters.
- Unstated payoff: about 90% efficiency versus roughly 20-50% for sliding lead screws, low wear, and near-zero backlash with a preloaded nut, hence their use in CNC axes. Trade-off: not self-locking, so vertical axes need a brake.
