# Month 2 Capstone - Quick Start Guide
## 2-3 Hour Intensive Coding Session

**Goal:** Build a quantum teleportation simulator with realistic noise  
**Team:** 3 people, working together in real-time  
**Time:** 2-3 hours  

---

## Before You Start (5 minutes)

**Each person:**
1. Have Python 3.8+ installed
2. Have numpy, scipy, matplotlib ready: `pip install numpy scipy matplotlib`
3. Have Month 1's `operator_toolkit.py` available
4. Choose a code editor (VS Code, PyCharm, or Jupyter)

**As a team:**
1. Decide who will "drive" (type code while others navigate)
2. Set up screen sharing or use VS Code Live Share
3. Create a folder: `mkdir quantum-teleportation`

---

## Session Timeline

```
0:00-0:15  Setup & role assignment
0:15-0:45  Part 1: Density matrices (Person 1 leads)
0:45-1:30  Part 2: Quantum channels (Person 2 leads)
1:30-2:15  Part 3: Teleportation (Person 3 leads)  
2:15-2:45  Part 4: Analysis & plots (All together)
2:45-3:00  Wrap-up & documentation
```

---

## What You'll Build

**One Python file** (`teleportation_simulator.py`) containing:

1. **DensityMatrix class** - represents mixed quantum states
   - Methods: `purity()`, `entropy()`
   
2. **partial_trace() function** - traces out subsystems

3. **QuantumChannel class** - implements noise via Kraus operators
   - Channels: bit-flip, depolarizing, amplitude damping
   
4. **teleport_with_noise() function** - simulates noisy teleportation

5. **Analysis code** - generates 2 plots showing:
   - How purity/entropy evolve during amplitude damping
   - How teleportation fidelity degrades with noise

---

## Key Concepts (Read Together - 10 min)

### Density Matrices
Pure state: ρ = |ψ⟩⟨ψ|  (purity = 1, entropy = 0)  
Mixed state: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|  (purity < 1, entropy > 0)

### Kraus Operators  
Quantum channel: ε(ρ) = Σₖ Eₖ ρ Eₖ†  
Completeness: Σₖ Eₖ† Eₖ = I  

Example (bit-flip with probability p):
```
E₀ = √(1-p) [[1,0],[0,1]]  # No flip
E₁ = √p [[0,1],[1,0]]      # Flip occurs
```

### Quantum Teleportation
- Alice has state |ψ⟩ to send to Bob
- They share EPR pair: (|00⟩ + |11⟩)/√2
- Alice measures, sends 2 classical bits
- Bob applies correction, gets |ψ⟩
- **Key:** Fidelity depends on EPR pair quality!

---

## Deliverables

At the end of 3 hours, you should have:

✅ `teleportation_simulator.py` (working code)  
✅ `amplitude_damping.png` (plot)  
✅ `teleportation_fidelity.png` (plot)  
✅ `summary_report.txt` (findings)  
✅ Git repository with all files  

---

## Grading (100 points)

- Density matrix & partial trace: 30 points
- Quantum channels (Kraus operators): 25 points  
- Teleportation simulation: 20 points
- Analysis & figures: 15 points
- Summary report: 10 points

**Bonus (+10 each):**
- Full 8x8 teleportation (not approximation)
- Bloch sphere visualization
- Concurrence calculation

---

## Tips

1. **Test as you go** - don't write everything then debug
2. **Use print statements** - see what's happening
3. **Check shapes** - `print(variable.shape)` constantly
4. **Normalize vectors** - always divide by norm
5. **Ask each other** - three heads better than one!

---

## Roles (but work together!)

**Person 1:** Density matrices  
**Person 2:** Quantum channels  
**Person 3:** Teleportation  

Everyone contributes to debugging and analysis!

---

## Ready? Let's Go! 🚀

Open the full capstone document: `Month_2_Capstone_Challenge.md`

It has all the code templates and step-by-step instructions.

**Remember:** The goal is to **learn by doing**, not to have perfect code. If stuck for >10 minutes on something, simplify or move on!

---

**Good luck! Have fun with quantum mechanics! ⚛️**
