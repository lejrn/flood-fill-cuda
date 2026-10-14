---
title: Induction cooktop - A Genius Invention That Boils Water Without Fire
source: A Genius Invention That Boils Water Without Fire.mp4
duration: 01:34
final_invention: Induction cooktop (induction hob)
core_function: heat a pot without a flame
chain_type: evolution chain
rungs: 5
---

# Induction cooktop: keep the shaking magnet, remove every moving part, then shake faster than ears can hear

**Chain at a glance:** Hand-shaken magnet under pot -> Motor-spun magnet -> Still coil electromagnet -> Coil on 60 Hz mains -> Silent high-frequency coil

**The simple problem:** A magnet shaken over iron heats the iron, so a pot could be heated with no fire. How do you make that a stove? [00:19]

## The ladder

### 1. Shake a magnet under the pot by hand [00:17]
- **Idea:** Shake a horseshoe magnet under a steel pot.
- **Why it works:** A fast-changing magnetic field agitates the iron, and the agitation becomes heat.
- **But:** Someone must stand there shaking it every time they cook. [00:25] - `manual-effort`
- **So:** Let a motor do it. [00:29] - `automate`
- **Shown as:** Silver atom-balls flush red, then the pot over the red/blue magnet turns red.
![Hand-shaken magnet heats the pot](frames/induction-cooktop/f006_0-00-21.jpg)

### 2. Spin the magnet with a motor [00:29]
- **Idea:** Mount the magnet on a powerful motor's shaft under the pot.
- **Why it works:** The field keeps changing with no human effort.
- **But:** Noise, vibration, worn-out parts - "not exactly kitchen-friendly." [00:37] - `side-effect`
- **So:** Replace the moving magnet with a coil under the surface fed by electricity. [00:43] - `substitute-medium`
- **Shown as:** A blue industrial motor spins the magnet while the engineer stirs, hand on chin; a blurred shaft close-up suggests shaking.
![Motor spinning the magnet](frames/induction-cooktop/f008_0-00-31.jpg)

### 3. A still coil electromagnet [00:43]
- **Idea:** A flat spiral coil under the surface becomes an electromagnet.
- **Why it works:** Current in a coil makes a magnetic field with no moving parts.
- **But:** A perfectly still magnet heats nothing. [00:49] - `doesn't-work`
- **So:** Flip the current's direction instead of moving the magnet. [00:55] - `field-not-motion`
- **Shown as:** A sparking copper spiral, tinted red-to-blue for its poles; the pot stays plain grey.
![Still coil under the pot](frames/induction-cooktop/f013_0-00-49.jpg)

### 4. Feed the coil household AC (60 Hz) [00:55]
- **Idea:** Mains power already reverses 60 times a second [01:00], making "a magnet that moves without going anywhere." [01:10]
- **Why it works:** The coil's poles swap back and forth, so the field changes.
- **But:** 60 Hz is audible; the stove hums. [01:13] - `side-effect`
- **So:** Push the frequency to tens of thousands of hertz. [01:19] - `push-parameter`
- **Shown as:** A sine wave over a wall socket, the coil's red/blue zones swapping, then the engineer clutching his ear.
![Coil poles flipping on mains AC](frames/induction-cooktop/f017_0-01-11.jpg)

### Final: High-frequency induction coil [01:25]
- **Idea:** The same motionless coil, driven at tens of kHz, silently shakes the pot tens of thousands of times per second.
- **Why it solves the chain:** No hand (1), no moving parts (2), a changing field without motion (3), a frequency above hearing (4).
- **Reveal:** After an oscilloscope shot and the engineer raising his arms, a black cooktop appears with "That's induction." [01:32]
![Finished induction cooktop](frames/induction-cooktop/f022_0-01-34.jpg)

## Scar map
Every part of the final design and the failure that forced it.
| Part of the final design | Exists because of | Rung |
|---|---|---|
| Electrical drive, no human motion | Someone had to keep shaking the magnet | 1 |
| Fixed flat coil, no moving parts | Motor noise, vibration, wear | 2 |
| Alternating current in the coil | A steady electromagnet heats nothing | 3 |
| High-frequency drive (tens of kHz) | 60 Hz hum | 4 |

## How the video tells it
- **Hook (first ~5 s):** The engineer holds a horseshoe magnet to his face: what if you shake a magnet "like crazy" over an iron plate?
- **Storyteller:** "Engineers" for the motor, then "a genius engineer" who "kept thinking". Connectives: So what if / But / So engineers / Then / But there was still a problem / Now.
- **Visual devices:** Desk-scale pot and motor; red atoms and red pot for heat; coil tinted by pole, colours swapping for AC; sine waves on a socket and an oscilloscope; engineer thinking (chin, eyes close-up), covering his ear, arms up.
- **Pacing:** 5 rungs in about 80 s after a 13 s hook, roughly 16 s per rung.

## Accuracy notes
- Heating comes mainly from eddy currents in the pot base (plus hysteresis loss in iron), not atomic friction; hence plain aluminium or copper pots fail on most cooktops.
- 60 Hz AC reverses 120 times per second, and much of the world uses 50 Hz. Real cooktops rectify mains and drive the coil with an inverter at about 20-100 kHz, which also improves efficiency, not only silence.
- The "genius engineer" is a teaching device: industrial induction heating long predates consumer cooktops (early 1970s).
