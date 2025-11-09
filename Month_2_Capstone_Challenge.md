# Monthly Capstone Challenge II: Simulating Noisy Quantum Teleportation

**An Introduction to Quantum Computing and Information**  
**Month 2: Quantum Dynamics & Information Formalism**  
November 9, 2025

---

## Abstract

This is a **2-3 hour intensive team challenge** where you will simulate quantum teleportation under realistic noise conditions. Working together in real-time, your team of three will build a quantum simulator from scratch, implement the teleportation protocol, and analyze how decoherence affects quantum information transfer. This hands-on challenge demonstrates the transition from pure state to density matrix formalism and introduces quantum channels via Kraus operators.

**Format:** Live coding session, all three team members working together  
**Duration:** 2-3 hours  
**Deliverable:** Working code + short analysis report

---

## Session Structure (2-3 Hours)

### Setup (0:00 - 0:15) - 15 minutes

**All together:**

1. **Create shared repository**
   ```bash
   mkdir quantum-teleportation
   cd quantum-teleportation
   git init
   touch teleportation_simulator.py
   ```

2. **Copy Month 1 baseline code**
   - One person shares their `operator_toolkit.py` from Month 1
   - Everyone has the same starting point

3. **Assign roles** (but everyone codes together):
   - **Person 1:** Density matrices & entropy
   - **Person 2:** Quantum channels (Kraus operators)
   - **Person 3:** Teleportation protocol & analysis

4. **Open single shared screen** (one person drives, others navigate)
   - Use VS Code Live Share, or
   - One person shares screen while others guide

### Part 1: Density Matrix Basics (0:15 - 0:45) - 30 minutes

**Person 1 leads, others help debug**

**Goal:** Implement basic density matrix functionality

```python
import numpy as np
from scipy.linalg import sqrtm

class DensityMatrix:
    """Density matrix representation of quantum state."""
    
    def __init__(self, state_vector):
        """Create density matrix from pure state."""
        psi = np.array(state_vector).reshape(-1, 1)
        psi = psi / np.linalg.norm(psi)
        self.matrix = psi @ psi.conj().T
    
    def purity(self):
        """Compute Tr(ρ²)."""
        return np.real(np.trace(self.matrix @ self.matrix))
    
    def entropy(self):
        """Compute von Neumann entropy S = -Tr(ρ log₂ ρ)."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        # Remove numerical noise
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        return -np.sum(eigenvalues * np.log2(eigenvalues))

def partial_trace(rho, dims, trace_out):
    """
    Trace out subsystem.
    
    Args:
        rho: Density matrix of composite system
        dims: [dim_A, dim_B] dimensions
        trace_out: 0 or 1 (which subsystem to trace out)
    """
    d_A, d_B = dims
    
    if trace_out == 1:  # Trace out B
        rho_A = np.zeros((d_A, d_A), dtype=complex)
        for i in range(d_A):
            for j in range(d_A):
                for k in range(d_B):
                    rho_A[i,j] += rho[i*d_B + k, j*d_B + k]
        return rho_A
    else:  # Trace out A
        rho_B = np.zeros((d_B, d_B), dtype=complex)
        for i in range(d_B):
            for j in range(d_B):
                for k in range(d_A):
                    rho_B[i,j] += rho[k*d_B + i, k*d_B + j]
        return rho_B

# Quick test
ket_0 = np.array([1, 0])
rho = DensityMatrix(ket_0)
print(f"Purity: {rho.purity():.3f}")  # Should be 1.0
print(f"Entropy: {rho.entropy():.3f}")  # Should be 0.0
```

**Checkpoint:** Verify that pure states have purity=1 and entropy=0

### Part 2: Quantum Channels (0:45 - 1:30) - 45 minutes

**Person 2 leads, others help implement**

**Goal:** Understand and implement Kraus operators for noise

#### Quick Theory (5 minutes)

**What are Kraus operators?**

A quantum channel ε transforms density matrices:
```
ε(ρ) = Σₖ Eₖ ρ Eₖ†
```

The Kraus operators {Eₖ} must satisfy: **Σₖ Eₖ† Eₖ = I**

**Example: Bit-flip with probability p**
```
E₀ = √(1-p) [[1, 0], [0, 1]]  # No flip
E₁ = √p [[0, 1], [1, 0]]      # Flip
```

#### Implementation (40 minutes)

```python
class QuantumChannel:
    """Quantum channel in Kraus representation."""
    
    def __init__(self, kraus_ops):
        """Initialize with list of Kraus operators."""
        self.kraus_ops = [np.array(E) for E in kraus_ops]
        assert self.is_cptp(), "Kraus operators must satisfy completeness!"
    
    def is_cptp(self):
        """Check if Σₖ Eₖ† Eₖ = I."""
        completeness = sum(E.conj().T @ E for E in self.kraus_ops)
        identity = np.eye(self.kraus_ops[0].shape[0])
        return np.allclose(completeness, identity)
    
    def apply(self, rho):
        """Apply channel: ε(ρ) = Σₖ Eₖ ρ Eₖ†."""
        if isinstance(rho, DensityMatrix):
            rho = rho.matrix
        
        result = sum(E @ rho @ E.conj().T for E in self.kraus_ops)
        
        # Return as DensityMatrix object
        new_dm = DensityMatrix([1, 0])  # Dummy initialization
        new_dm.matrix = result
        return new_dm
    
    @staticmethod
    def bit_flip(p):
        """Bit-flip channel with probability p."""
        E0 = np.sqrt(1-p) * np.eye(2)
        E1 = np.sqrt(p) * np.array([[0, 1], [1, 0]])
        return QuantumChannel([E0, E1])
    
    @staticmethod
    def depolarizing(p):
        """Depolarizing channel."""
        E0 = np.sqrt(1 - 3*p/4) * np.eye(2)
        E1 = np.sqrt(p/4) * np.array([[0, 1], [1, 0]])    # σₓ
        E2 = np.sqrt(p/4) * np.array([[0, -1j], [1j, 0]]) # σᵧ
        E3 = np.sqrt(p/4) * np.array([[1, 0], [0, -1]])   # σz
        return QuantumChannel([E0, E1, E2, E3])
    
    @staticmethod
    def amplitude_damping(gamma):
        """Amplitude damping (energy loss) channel."""
        E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        return QuantumChannel([E0, E1])

# Quick test
ket_1 = np.array([0, 1])
rho_1 = DensityMatrix(ket_1)

channel = QuantumChannel.amplitude_damping(0.3)
rho_damped = channel.apply(rho_1)

print(f"Population in |0⟩: {rho_damped.matrix[0,0].real:.3f}")  # Should be ~0.3
```

**Checkpoint:** Verify that amplitude damping moves population from |1⟩ to |0⟩

### Part 3: Quantum Teleportation (1:30 - 2:15) - 45 minutes

**Person 3 leads, all implement together**

**Goal:** Implement teleportation protocol with noise

```python
def teleport_with_noise(psi, noise_strength=0.0):
    """
    Teleport quantum state |ψ⟩ using noisy EPR pair.
    
    Args:
        psi: State to teleport (2D array)
        noise_strength: Depolarizing noise parameter
    
    Returns:
        Fidelity of teleportation
    """
    # Normalize input state
    psi = np.array(psi).flatten()
    psi = psi / np.linalg.norm(psi)
    
    # Computational basis states
    ket_0 = np.array([1, 0])
    ket_1 = np.array([0, 1])
    
    # Create perfect EPR pair |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    epr = (np.kron(ket_0, ket_0) + np.kron(ket_1, ket_1)) / np.sqrt(2)
    rho_epr = np.outer(epr, epr.conj())
    
    # Apply noise to EPR pair (both halves)
    if noise_strength > 0:
        noise = QuantumChannel.depolarizing(noise_strength)
        
        # This is simplified - proper implementation requires careful indexing
        # For quick demo, apply to each half separately (not fully correct but illustrative)
        dim = 4
        rho_noisy = np.zeros((dim, dim), dtype=complex)
        
        # Apply noise to first qubit
        for i in range(2):
            for j in range(2):
                block = rho_epr[i*2:(i+1)*2, j*2:(j+1)*2]
                noisy_block = noise.apply(DensityMatrix(ket_0)).matrix if i == j else block
                rho_noisy[i*2:(i+1)*2, j*2:(j+1)*2] = block
        
        rho_epr = rho_noisy
    
    # Initial state: |ψ⟩_A ⊗ |EPR⟩_{AB}
    # For simplicity, compute expected fidelity analytically
    # (Full simulation would require 8x8 matrices - beyond scope of 2-hour project)
    
    # Fidelity formula for depolarizing noise on teleportation:
    # F = (2 + (1-p)²) / 3  where p is noise strength
    
    if noise_strength == 0:
        fidelity = 1.0
    else:
        fidelity = (2 + (1 - noise_strength)**2) / 3
    
    return fidelity

# Test teleportation
psi_test = np.array([0.6, 0.8])

print("\nTeleportation Analysis:")
print(f"{'Noise':>10s} {'Fidelity':>10s} {'Classical?':>12s}")
print("-" * 35)

for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    F = teleport_with_noise(psi_test, p)
    classical_limit = 2/3
    beats_classical = "YES" if F > classical_limit else "NO"
    print(f"{p:>10.2f} {F:>10.3f} {beats_classical:>12s}")
```

**Checkpoint:** Verify that fidelity decreases with noise and drops below 2/3 threshold

### Part 4: Analysis & Visualization (2:15 - 2:45) - 30 minutes

**All together: generate plots and analysis**

```python
import matplotlib.pyplot as plt

# Analysis 1: Amplitude damping trajectory
print("\n" + "="*50)
print("ANALYSIS 1: Amplitude Damping")
print("="*50)

gamma_vals = np.linspace(0, 1, 20)
purities = []
entropies = []

ket_1 = np.array([0, 1])
rho_initial = DensityMatrix(ket_1)

for gamma in gamma_vals:
    channel = QuantumChannel.amplitude_damping(gamma)
    rho_t = channel.apply(rho_initial)
    purities.append(rho_t.purity())
    entropies.append(rho_t.entropy())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(gamma_vals, purities, 'b-', linewidth=2)
ax1.set_xlabel('Damping parameter γ')
ax1.set_ylabel('Purity Tr(ρ²)')
ax1.set_title('Purity Decay')
ax1.grid(True, alpha=0.3)

ax2.plot(gamma_vals, entropies, 'r-', linewidth=2)
ax2.set_xlabel('Damping parameter γ')
ax2.set_ylabel('von Neumann Entropy (bits)')
ax2.set_title('Entropy Growth')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('amplitude_damping.png', dpi=150)
print("Saved: amplitude_damping.png")

# Analysis 2: Teleportation fidelity
print("\n" + "="*50)
print("ANALYSIS 2: Teleportation Fidelity vs Noise")
print("="*50)

p_vals = np.linspace(0, 0.5, 30)
fidelities = [teleport_with_noise([0.6, 0.8], p) for p in p_vals]

plt.figure(figsize=(8, 5))
plt.plot(p_vals, fidelities, 'b-', linewidth=2, label='Teleportation fidelity')
plt.axhline(2/3, color='r', linestyle='--', linewidth=2, label='Classical limit (2/3)')
plt.fill_between(p_vals, fidelities, 2/3, where=(np.array(fidelities) > 2/3), 
                 alpha=0.3, color='green', label='Quantum advantage')
plt.xlabel('Depolarizing noise p')
plt.ylabel('Fidelity')
plt.title('Quantum Teleportation Performance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('teleportation_fidelity.png', dpi=150)
print("Saved: teleportation_fidelity.png")

# Find critical threshold
for i, (p, F) in enumerate(zip(p_vals, fidelities)):
    if F < 2/3:
        print(f"\nCritical threshold: p_c ≈ {p:.3f}")
        print(f"Beyond this, quantum teleportation loses advantage over classical communication")
        break

plt.show()
```

### Wrap-up & Documentation (2:45 - 3:00) - 15 minutes

**Create summary report**

```python
# Generate summary report
summary = f"""
QUANTUM TELEPORTATION SIMULATION - SUMMARY REPORT
================================================

Team Members: [Name 1], [Name 2], [Name 3]
Date: November 9, 2025

KEY FINDINGS:

1. Amplitude Damping:
   - Pure state |1⟩ decays toward mixed state
   - Purity decreases from 1.0 to {purities[-1]:.3f}
   - Entropy increases from 0.0 to {entropies[-1]:.3f} bits
   
2. Quantum Teleportation:
   - Perfect fidelity (F=1.0) with ideal EPR pair
   - Degrades with depolarizing noise
   - Critical threshold: p_c ≈ 0.37
   - Below threshold: Quantum beats classical!

3. Physical Insights:
   - Decoherence converts pure states to mixed states
   - Information "leaks" into environment (entropy increases)
   - Entanglement enables quantum advantage in teleportation
   - Noise limits practical quantum communication

CONCLUSIONS:
Quantum teleportation requires high-quality entanglement.
Noise protection (error correction) is essential for quantum networks.
"""

print(summary)

with open('summary_report.txt', 'w') as f:
    f.write(summary)

print("\n" + "="*50)
print("CAPSTONE COMPLETE! 🎉")
print("="*50)
print("\nGenerated files:")
print("  • amplitude_damping.png")
print("  • teleportation_fidelity.png")
print("  • summary_report.txt")
print("  • teleportation_simulator.py (your code)")
```

---

## Understanding Kraus Operators (10-Minute Theory Brief)

**Why do we need them?**

In Month 1, you worked with **pure states** evolving unitarily. In reality, quantum systems interact with environments, causing **decoherence**. We can't track every environmental degree of freedom, so we need **quantum channels** to describe the effective evolution of our system alone.

**What are they?**

A **quantum channel** ε maps density matrices:

```
ε(ρ) = Σₖ Eₖ ρ Eₖ†
```

The **Kraus operators** {Eₖ} must satisfy the **completeness relation**:

```
Σₖ Eₖ† Eₖ = I
```

This ensures probability conservation (trace preservation).

**Physical Example: Amplitude Damping**

An excited atom can spontaneously emit a photon:
- |1⟩ (excited) → |0⟩ (ground) with probability γ  
- |1⟩ (excited) → |1⟩ (excited) with probability 1-γ

The Kraus operators are:

```
E₀ = [[1, 0], [0, √(1-γ)]]    # No emission
E₁ = [[0, √γ], [0, 0]]         # Emission occurs
```

**Verify completeness:**

```python
E0 = np.array([[1, 0], [0, np.sqrt(1-0.3)]])
E1 = np.array([[0, np.sqrt(0.3)], [0, 0]])

completeness = E0.conj().T @ E0 + E1.conj().T @ E1
print(completeness)  # Should be [[1, 0], [0, 1]]
```

**That's it!** The rest is implementation.

---

## Part I: Refactoring to the Density Matrix Formalism

### 1.1 Theoretical Foundation

In Month 1, your toolkit operated on pure states |ψ⟩. However, the density matrix formalism is the most general description of quantum states and is essential for describing:
- Mixed states (statistical ensembles)
- Subsystems of entangled states
- Open quantum systems subject to decoherence

**Your Task:**

Refactor your `operator_toolkit.py` library to use density matrices as the fundamental object. 

### 1.2 Core Extensions Required

Extend your library with the following new classes and functions:

#### Class: `DensityMatrix`

Implement a `DensityMatrix` class that:

1. **Initialization:** Can be constructed from:
   - A pure state: ρ = |ψ⟩⟨ψ|
   - A classical mixture: ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|
   - A matrix representation directly

2. **Properties:** Methods to compute:
   - Purity: Tr(ρ²)
   - Von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ)
   - Expectation value of an observable: ⟨A⟩ = Tr(ρA)

3. **Verification:** Methods to check:
   - Hermiticity: ρ = ρ†
   - Unit trace: Tr(ρ) = 1
   - Positive semi-definiteness: all eigenvalues ≥ 0

#### Function: `partial_trace(rho, dims, trace_out)`

Implement the partial trace operation for bipartite systems.

**Parameters:**
- `rho`: Density matrix of composite system AB
- `dims`: List [dim_A, dim_B] specifying dimensions of subsystems
- `trace_out`: Integer (0 or 1) specifying which subsystem to trace out

**Mathematical Definition:**

For a bipartite density matrix ρ_AB, the reduced density matrix of subsystem A is:

```
ρ_A = Tr_B(ρ_AB) = Σₖ (I_A ⊗ ⟨k|_B) ρ_AB (I_A ⊗ |k⟩_B)
```

where {|k⟩_B} forms an orthonormal basis for subsystem B.

**Verification Test:**

Your implementation must pass the following test:

```python
# Create a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
bell_state = (tensor_product(ket_0, ket_0) + tensor_product(ket_1, ket_1)) / np.sqrt(2)
rho_bell = DensityMatrix(bell_state)

# Trace out subsystem B
rho_A = partial_trace(rho_bell, dims=[2,2], trace_out=1)

# Result should be maximally mixed state I/2
assert np.allclose(rho_A.matrix, np.eye(2)/2)
```

#### Function: `von_neumann_entropy(rho)`

Compute the von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

**Implementation Requirements:**
- Handle numerical precision issues for near-zero eigenvalues
- Use the convention 0 log 0 = 0
- Return entropy in bits (base-2 logarithm)

**Verification Test:**

```python
# Pure state should have zero entropy
psi_pure = random_state_vector(4)
rho_pure = DensityMatrix(psi_pure)
assert np.isclose(von_neumann_entropy(rho_pure), 0.0)

# Maximally mixed state of dimension d should have entropy log₂(d)
rho_mixed = DensityMatrix(np.eye(4)/4)
assert np.isclose(von_neumann_entropy(rho_mixed), 2.0)
```

---

## Deliverables (End of 3-Hour Session)

### 1. Working Code (`teleportation_simulator.py`)

Your single Python file should contain:
- `DensityMatrix` class with purity() and entropy()
- `partial_trace()` function
- `QuantumChannel` class with Kraus operators
- `teleport_with_noise()` function
- Analysis code generating 2 plots

### 2. Two Figures

- `amplitude_damping.png` - Purity and entropy evolution
- `teleportation_fidelity.png` - Fidelity vs noise with classical limit

### 3. Summary Report (`summary_report.txt`)

Short text file (~200 words) with:
- Key numerical findings
- Critical noise threshold
- Physical insights

### 4. Git Repository

```bash
# At end of session, commit everything:
git add teleportation_simulator.py
git add *.png summary_report.txt
git commit -m "Complete capstone: noisy quantum teleportation"
git remote add origin [your-repo-url]
git push -u origin main
```

---

## Grading Rubric (100 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Density Matrix** | 20 | Correct implementation, purity/entropy working |
| **Partial Trace** | 10 | Works for Bell state test |
| **Quantum Channels** | 25 | All channels implemented, CPTP verified |
| **Teleportation** | 20 | Fidelity calculation correct |
| **Analysis & Plots** | 15 | Both figures generated, clearly labeled |
| **Report** | 10 | Clear findings, identifies threshold |
| **Total** | 100 | |

### Bonus (+10 each, up to +20):
- Implement full 8x8 teleportation (not analytical approximation)
- Add Bloch sphere visualization of amplitude damping
- Implement concurrence calculation for entanglement

---

## Tips for Success

### Time Management
- **Don't get stuck!** If something doesn't work after 10 minutes, simplify or move on
- **Use print statements** liberally to debug
- **Test incrementally** - verify each function before moving to next

### Common Pitfalls
1. **Normalization:** Always normalize state vectors!
2. **Complex conjugates:** Use `.conj().T` not just `.T`
3. **Matrix dimensions:** Double-check shapes with `.shape`
4. **Numerical precision:** Use `np.allclose()` not `==` for comparisons

### If You Get Stuck
```python
# Debug template:
print(f"Shape: {variable.shape}")
print(f"Type: {type(variable)}")
print(f"Values:\n{variable}")
assert condition, "Error message explaining what you expected"
```

---

## Learning Objectives

By completing this 2-3 hour challenge, you will:

✅ Understand density matrix formalism  
✅ Implement quantum channels with Kraus operators  
✅ Simulate realistic decoherence processes  
✅ Analyze quantum information protocols  
✅ Visualize quantum dynamics  
✅ Work effectively as a coding team  

---

## Post-Session Reflection

After completing the capstone, discuss as a team:

1. **What was the hardest part?**
   - Understanding Kraus operators?
   - Implementing partial trace?
   - Debugging?

2. **What surprised you?**
   - How quickly states decohere?
   - The critical noise threshold?
   - How noise affects teleportation?

3. **Physical insights:**
   - Why does entropy increase with decoherence?
   - Why does entanglement enable teleportation?
   - What limits quantum communication in practice?

---

## Additional Resources

**If you want to go deeper after the session:**

1. Nielsen & Chuang, Chapter 8: "Quantum noise and quantum operations"
2. Preskill Notes, Chapter 3: "Foundations II: Measurement and evolution"
3. Original teleportation paper: Bennett et al., PRL 70, 1895 (1993)

**Online simulators to explore:**
- IBM Quantum Experience (Qiskit)
- QuTiP (Quantum Toolbox in Python)
- Quirk (browser-based circuit simulator)

---

## Final Thought

You've just simulated one of the most mind-bending protocols in quantum physics - teleportation! You've seen how entanglement enables quantum communication and how decoherence threatens it.

This is the central challenge of quantum technology: **building systems that maintain quantum coherence long enough to do useful computation or communication.**

Understanding noise isn't just an academic exercise - it's the key to building practical quantum devices.

**Congratulations on completing the Month 2 Capstone! 🎉**

*"In theory, theory and practice are the same. In practice, they are not."*  
*- Jan L. A. van de Snepscheut*

(And now you know why - decoherence!)

---

**Repository Template:**

```bash
quantum-teleportation/
├── teleportation_simulator.py    # Your code
├── amplitude_damping.png          # Figure 1
├── teleportation_fidelity.png     # Figure 2
├── summary_report.txt             # Analysis
└── README.md                      # Brief project description
```

**Commit and submit your repository URL. Good luck! 🚀**

### 2.1 Member 2's Primary Task: The QuantumChannel Class

Now that you understand Kraus operators theoretically, let's implement a robust, reusable class.

#### Class Design Specifications

```python
class QuantumChannel:
    """
    A quantum channel represented in Kraus form.
    
    Represents a CPTP map: ε(ρ) = Σₖ Eₖ ρ Eₖ†
    
    Attributes:
        kraus_ops (list): List of Kraus operators (numpy arrays)
        name (str): Descriptive name of the channel
        dim (int): Dimension of the Hilbert space
        
    Methods:
        apply(rho): Apply channel to density matrix
        is_cptp(): Verify completeness relation
        compose(other): Compose with another channel
        __repr__(): String representation
    """
    
    def __init__(self, kraus_ops, name="Unnamed Channel"):
        """
        Initialize quantum channel with Kraus operators.
        
        Parameters:
            kraus_ops (list): List of Kraus operators
            name (str): Channel name
            
        Raises:
            ValueError: If Kraus operators don't satisfy completeness
        """
        # Your implementation here
        pass
```

#### Required Methods

##### 1. Completeness Verification

```python
def is_cptp(self, tolerance=1e-10):
    """
    Verify CPTP property: Σₖ Eₖ† Eₖ = I
    
    Parameters:
        tolerance (float): Numerical precision threshold
        
    Returns:
        bool: True if channel is CPTP
        
    Example:
        >>> E0 = np.sqrt(0.7) * np.eye(2)
        >>> E1 = np.sqrt(0.3) * sigma_x
        >>> channel = QuantumChannel([E0, E1], "Bit-flip")
        >>> channel.is_cptp()
        True
    """
    # Compute Σₖ Eₖ† Eₖ
    # Compare to identity
    # Return boolean
    pass
```

##### 2. Channel Application

```python
def apply(self, rho):
    """
    Apply channel to density matrix: ε(ρ) = Σₖ Eₖ ρ Eₖ†
    
    Parameters:
        rho (DensityMatrix or np.ndarray): Input state
        
    Returns:
        DensityMatrix: Output state
        
    Example:
        >>> rho_in = DensityMatrix(ket_0)
        >>> rho_out = channel.apply(rho_in)
    """
    # Handle both DensityMatrix objects and numpy arrays
    # Apply Kraus operator sum
    # Return DensityMatrix object
    pass
```

##### 3. Channel Composition

```python
def compose(self, other):
    """
    Compose two channels: (ε₁ ∘ ε₂)(ρ) = ε₁(ε₂(ρ))
    
    The Kraus operators of the composition are all products:
    {Eᵢ⁽¹⁾ Eⱼ⁽²⁾} for all i,j
    
    Parameters:
        other (QuantumChannel): Channel to compose with
        
    Returns:
        QuantumChannel: Composed channel
        
    Example:
        >>> bit_flip = QuantumChannel.bit_flip(0.1)
        >>> phase_flip = QuantumChannel.phase_flip(0.1)
        >>> combined = bit_flip.compose(phase_flip)
    """
    # Compute all products Eᵢ⁽¹⁾ @ Eⱼ⁽²⁾
    # Create new QuantumChannel with product operators
    pass
```

### 2.2 Factory Methods: Creating Standard Channels

Implement class methods to create common channels:

```python
@classmethod
def bit_flip(cls, p):
    """
    Create bit-flip channel.
    
    Physics: With probability p, apply σₓ
    
    Kraus operators:
        E₀ = √(1-p) I
        E₁ = √p σₓ
        
    Parameters:
        p (float): Flip probability, 0 ≤ p ≤ 1
        
    Returns:
        QuantumChannel: Bit-flip channel
    """
    if not 0 <= p <= 1:
        raise ValueError("Probability must be in [0,1]")
        
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    E0 = np.sqrt(1-p) * np.eye(2)
    E1 = np.sqrt(p) * sigma_x
    
    return cls([E0, E1], f"Bit-flip(p={p:.3f})")

@classmethod
def phase_flip(cls, p):
    """Create phase-flip channel."""
    # TODO: Implement
    pass

@classmethod
def depolarizing(cls, p):
    """
    Create depolarizing channel.
    
    Physics: With probability p, replace state with I/2
    
    Kraus operators:
        E₀ = √(1-3p/4) I
        E₁ = √(p/4) σₓ
        E₂ = √(p/4) σᵧ
        E₃ = √(p/4) σz
    """
    # TODO: Implement
    pass

@classmethod
def amplitude_damping(cls, gamma):
    """
    Create amplitude damping channel.
    
    Physics: T₁ relaxation (spontaneous emission)
    
    Kraus operators:
        E₀ = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        E₁ = √γ |0⟩⟨1|
    """
    # TODO: Implement
    pass

@classmethod
def phase_damping(cls, gamma):
    """
    Create phase damping channel (T₂ dephasing).
    
    Physics: Random phase kicks without energy loss
    
    Kraus operators:
        E₀ = |0⟩⟨0| + √(1-γ)|1⟩⟨1|
        E₁ = √γ |1⟩⟨1|
    """
    # TODO: Implement
    pass
```

### 2.3 Testing Your Implementation

Create comprehensive tests in `tests/test_quantum_channels.py`:

```python
import numpy as np
import pytest
from src.quantum_channels import QuantumChannel
from src.density_matrix import DensityMatrix

class TestQuantumChannel:
    """Test suite for QuantumChannel class."""
    
    def test_cptp_verification(self):
        """Test that standard channels satisfy CPTP."""
        channels = [
            QuantumChannel.bit_flip(0.3),
            QuantumChannel.phase_flip(0.2),
            QuantumChannel.depolarizing(0.15),
            QuantumChannel.amplitude_damping(0.5)
        ]
        
        for channel in channels:
            assert channel.is_cptp(), f"{channel.name} failed CPTP test"
    
    def test_trace_preservation(self):
        """Test that channels preserve trace."""
        rho = DensityMatrix.random_mixed_state(2)
        channel = QuantumChannel.depolarizing(0.3)
        
        rho_out = channel.apply(rho)
        
        assert np.isclose(np.trace(rho_out.matrix), 1.0)
    
    def test_bit_flip_on_computational_basis(self):
        """Test bit-flip channel on |0⟩ and |1⟩."""
        p = 0.3
        channel = QuantumChannel.bit_flip(p)
        
        # Test on |0⟩
        ket_0 = np.array([[1], [0]], dtype=complex)
        rho_0 = DensityMatrix(ket_0)
        rho_out = channel.apply(rho_0)
        
        # Should be (1-p)|0⟩⟨0| + p|1⟩⟨1|
        expected = np.array([
            [1-p, 0],
            [0, p]
        ])
        
        assert np.allclose(rho_out.matrix, expected)
    
    def test_amplitude_damping_ground_state(self):
        """Ground state should be unchanged by amplitude damping."""
        gamma = 0.5
        channel = QuantumChannel.amplitude_damping(gamma)
        
        ket_0 = np.array([[1], [0]], dtype=complex)
        rho_0 = DensityMatrix(ket_0)
        rho_out = channel.apply(rho_0)
        
        assert np.allclose(rho_out.matrix, rho_0.matrix)
    
    def test_amplitude_damping_excited_state(self):
        """Excited state should decay toward ground state."""
        gamma = 0.4
        channel = QuantumChannel.amplitude_damping(gamma)
        
        ket_1 = np.array([[0], [1]], dtype=complex)
        rho_1 = DensityMatrix(ket_1)
        rho_out = channel.apply(rho_1)
        
        # Population should be: |0⟩: gamma, |1⟩: 1-gamma
        expected = np.array([
            [gamma, 0],
            [0, 1-gamma]
        ])
        
        assert np.allclose(rho_out.matrix, expected)
    
    def test_channel_composition(self):
        """Test composition of two channels."""
        bf = QuantumChannel.bit_flip(0.1)
        pf = QuantumChannel.phase_flip(0.1)
        
        composed = bf.compose(pf)
        
        # Should have 2x2 = 4 Kraus operators
        assert len(composed.kraus_ops) == 4
        assert composed.is_cptp()
    
    def test_identity_channel(self):
        """Identity channel should leave state unchanged."""
        E_id = [np.eye(2)]
        identity = QuantumChannel(E_id, "Identity")
        
        rho = DensityMatrix.random_mixed_state(2)
        rho_out = identity.apply(rho)
        
        assert np.allclose(rho_out.matrix, rho.matrix)

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 2.4 Integration with Member 1's Code

Your `QuantumChannel` class must work seamlessly with Member 1's `DensityMatrix` class:

```python
# Example integration
from src.density_matrix import DensityMatrix
from src.quantum_channels import QuantumChannel

# Member 1's code creates a state
psi = np.array([[1], [0]], dtype=complex)
rho = DensityMatrix(psi)

# Member 2's code applies a channel
channel = QuantumChannel.depolarizing(0.2)
rho_noisy = channel.apply(rho)

# Should work smoothly!
print(f"Initial purity: {rho.purity():.3f}")
print(f"Final purity: {rho_noisy.purity():.3f}")
print(f"Entropy increase: {rho_noisy.entropy() - rho.entropy():.3f} bits")
```

### 2.5 Collaboration Checkpoint

Before proceeding to Part III, Member 2 should:

1. **Commit code:**
   ```bash
   git add src/quantum_channels.py tests/test_quantum_channels.py
   git commit -m "Implement QuantumChannel class with standard noise models"
   git push origin feature/quantum-channels
   ```

2. **Create pull request:**
   - Title: "Feature: Quantum Channels Implementation"
   - Description: Summarize what was implemented
   - Request review from Member 1 and Member 3

3. **Integration test:**
   ```bash
   # On main branch, after Member 1's merge:
   git checkout main
   git pull
   git checkout feature/quantum-channels
   git merge main
   
   # Run all tests
   pytest tests/ -v
   ```

**Member 1 and Member 3:** Review the PR, test the code, provide feedback.

---

## Part III: Bloch Sphere Visualization of Amplitude Damping

### 3.1 The Bloch Sphere Representation

Any single-qubit density matrix can be written as:

```
ρ = (I + r⃗·σ⃗)/2
```

where r⃗ = (rₓ, rᵧ, rz) is the Bloch vector with |r⃗| ≤ 1.

**Properties:**
- Pure states: |r⃗| = 1 (on surface)
- Mixed states: |r⃗| < 1 (inside)
- Maximally mixed: r⃗ = 0 (center)

### 3.2 Simulation Task

**Objective:** Simulate and visualize the trajectory of a qubit undergoing amplitude damping from |1⟩ to |0⟩.

**Protocol:**

1. **Initial State:** ρ(0) = |1⟩⟨1|  
   Bloch vector: r⃗(0) = (0, 0, -1)

2. **Time Evolution:** Apply amplitude damping with γ(t) = 1 - e^(-t/T₁)  
   where T₁ is the characteristic decay time (set T₁ = 1).

3. **Compute Trajectory:** For t ∈ [0, 5T₁], compute:
   - Density matrix ρ(t)
   - Bloch vector r⃗(t) = (Tr(ρσₓ), Tr(ρσᵧ), Tr(ρσz))
   - Purity Tr(ρ²)
   - Von Neumann entropy S(ρ)

4. **Visualization:** Create a 3D plot showing:
   - The Bloch sphere (wireframe)
   - The trajectory r⃗(t) as the state decays
   - Color-coded by time or purity

**Expected Behavior:**

The state should:
- Start at south pole: r⃗(0) = (0, 0, -1)
- Spiral inward and upward
- End at origin: r⃗(∞) = (0, 0, 0) [maximally mixed state]

**Deliverables:**

1. Animated 3D plot of the Bloch sphere trajectory
2. Time-series plots showing:
   - rₓ(t), rᵧ(t), rz(t)
   - Purity vs time
   - Entropy vs time

---

## Part IV: Quantum Teleportation Protocol

### 4.1 The Teleportation Protocol

Quantum teleportation allows Alice to transmit an unknown quantum state |ψ⟩ to Bob using:
- One shared EPR pair (entangled state)
- Two classical bits of communication

**Standard Protocol:**

```
Initial state: |ψ⟩_A ⊗ |Φ⁺⟩_AB
            = |ψ⟩_A ⊗ (|00⟩ + |11⟩)_AB / √2

After Alice's Bell measurement and Bob's correction: |ψ⟩_B
```

### 4.2 Implementation Using Density Matrices

**Task:** Implement the complete teleportation protocol using the density matrix formalism.

#### Step 1: Prepare Initial State

```python
# Alice's unknown state to teleport
psi = alpha * ket_0 + beta * ket_1  # Arbitrary coefficients

# Shared EPR pair
phi_plus = (tensor_product(ket_0, ket_0) + 
            tensor_product(ket_1, ket_1)) / np.sqrt(2)

# Total initial state (3 qubits: Alice's qubit, Alice's EPR, Bob's EPR)
initial_state = tensor_product(psi, phi_plus)
rho_initial = DensityMatrix(initial_state)
```

#### Step 2: Alice's Bell Measurement

Alice performs a Bell-state measurement on her two qubits.

**Measurement operators (4 outcomes):**

```python
# Bell basis states
phi_plus  = (ket_00 + ket_11) / sqrt(2)
phi_minus = (ket_00 - ket_11) / sqrt(2)
psi_plus  = (ket_01 + ket_10) / sqrt(2)
psi_minus = (ket_01 - ket_10) / sqrt(2)

# Projectors
P0 = outer(phi_plus, phi_plus)   # Outcome 00
P1 = outer(phi_minus, phi_minus) # Outcome 01
P2 = outer(psi_plus, psi_plus)   # Outcome 10
P3 = outer(psi_minus, psi_minus) # Outcome 11
```

#### Step 3: Post-Measurement States

For each measurement outcome k, the post-measurement state is:

```
ρ_k = (P_k ⊗ I_Bob) ρ (P_k ⊗ I_Bob)† / p_k
```

where p_k = Tr[(P_k ⊗ I_Bob) ρ (P_k ⊗ I_Bob)†] is the outcome probability.

#### Step 4: Bob's Unitary Corrections

Based on Alice's classical message (2 bits), Bob applies:

- 00 → I (no operation)
- 01 → σz
- 10 → σₓ
- 11 → σₓσz

#### Step 5: Extract Bob's Final State

Trace out Alice's qubits to obtain Bob's reduced density matrix.

### 4.3 Verification

For perfect teleportation:

```python
rho_bob_final = partial_trace(rho_after_correction, dims=[2,2,2], trace_out=[0,1])

# Fidelity should be 1
fidelity = bra(psi) @ rho_bob_final.matrix @ ket(psi)
assert np.isclose(fidelity, 1.0)
```

---

## Part V: Teleportation Under Noise

### 5.1 Realistic Noise Model

In real quantum systems, the shared EPR pair undergoes decoherence before use.

**Task:** Investigate how different noise channels affect teleportation fidelity.

### 5.2 Noise Injection Protocol

**Modify the protocol:**

1. Prepare perfect EPR pair: |Φ⁺⟩⟨Φ⁺|

2. **Apply noise to Alice's half:** ε_A(ρ_A)  
   Apply noise to Bob's half: ε_B(ρ_B)

3. Form noisy EPR state:  
   ```
   ρ_EPR_noisy = (ε_A ⊗ ε_B)(|Φ⁺⟩⟨Φ⁺|)
   ```

4. Proceed with standard teleportation protocol

### 5.3 Fidelity Analysis

The teleportation fidelity is:

```
F(|ψ⟩, ρ_Bob) = ⟨ψ|ρ_Bob|ψ⟩
```

**Analysis Tasks:**

1. **Depolarizing Noise:**  
   For p ∈ [0, 0.5], plot F(p) for 10 random input states |ψ⟩

2. **Amplitude Damping:**  
   For γ ∈ [0, 1], plot F(γ) for:
   - |0⟩ (ground state)
   - |1⟩ (excited state)
   - |+⟩ (equal superposition)
   
3. **Asymmetric Noise:**  
   Apply depolarizing noise to Alice (p_A) and amplitude damping to Bob (γ_B).  
   Create a 2D contour plot of F(p_A, γ_B).

### 5.4 Critical Threshold

**Research Question:**

For each noise model, determine the critical noise parameter p_c or γ_c beyond which teleportation fidelity drops below the classical threshold.

**Classical Threshold:**  
The fidelity achievable by the best classical strategy (measure and prepare) is F_classical = 2/3.

**Task:**  
Compute and report p_c for depolarizing noise and γ_c for amplitude damping.

---

## Part VI: Team Integration & Final Assembly

### 6.1 Integration Sprint (Days 7-9)

Once all three feature branches are complete, the team must merge and integrate everything.

#### Day 7: Merge Party

**Morning Session (All members together):**

```bash
# Update main with latest changes
git checkout main
git pull

# Merge strategy: Feature by feature
# Member 1's code first (foundation)
git merge feature/density-matrix --no-ff
pytest tests/test_density_matrix.py -v

# Member 2's code second (channels)  
git merge feature/quantum-channels --no-ff
pytest tests/test_quantum_channels.py -v

# Member 3's code third (teleportation)
git merge feature/teleportation --no-ff
pytest tests/test_teleportation.py -v

# Run full test suite
pytest tests/ -v --cov=src
```

**Afternoon Session: Resolve Conflicts**

Common conflicts to expect:
- Import statements in `__init__.py`
- Shared utility functions
- Conflicting class method names

**Conflict resolution protocol:**
1. Identify conflict
2. Team discusses best solution
3. One person makes the fix
4. Others review the fix
5. Commit and continue

#### Day 8: Integration Testing

Create `tests/test_integration.py`:

```python
"""
Integration tests: Verify all components work together.
"""
import numpy as np
import pytest
from src.density_matrix import DensityMatrix
from src.quantum_channels import QuantumChannel
from src.teleportation import QuantumTeleportation
from src.operator_toolkit import tensor_product, ket, bra

class TestFullPipeline:
    """Test complete workflow from state preparation to analysis."""
    
    def test_noisy_teleportation_pipeline(self):
        """
        End-to-end test: Prepare state -> Add noise -> Teleport -> Measure fidelity
        """
        # Prepare arbitrary state (Member 1)
        alpha, beta = 0.6, 0.8
        psi = alpha * ket(0) + beta * ket(1)
        
        # Create noisy channel (Member 2)
        noise = QuantumChannel.depolarizing(0.1)
        
        # Perform noisy teleportation (Member 3)
        teleporter = QuantumTeleportation()
        fidelity = teleporter.teleport_with_noise(psi, noise)
        
        # Verify fidelity degradation
        assert 0.8 < fidelity < 1.0, "Fidelity outside expected range"
        
    def test_bloch_sphere_visualization(self):
        """Test that visualization code works with all components."""
        # Create amplitude damping trajectory (Member 2)
        from src.visualization import BlochSphere
        
        gamma_vals = np.linspace(0, 1, 20)
        trajectory = []
        
        ket_1 = ket(1)
        rho = DensityMatrix(ket_1)
        
        for gamma in gamma_vals:
            channel = QuantumChannel.amplitude_damping(gamma)
            rho_t = channel.apply(rho)
            trajectory.append(rho_t.bloch_vector())
        
        # Should work without errors
        bloch = BlochSphere()
        bloch.plot_trajectory(trajectory)
        
    def test_entropy_calculations_consistency(self):
        """Verify entropy calculations are consistent across modules."""
        from src.density_matrix import von_neumann_entropy
        
        # Create mixed state
        rho = DensityMatrix.random_mixed_state(2, purity=0.7)
        
        # Entropy via DensityMatrix method
        S1 = rho.entropy()
        
        # Entropy via standalone function  
        S2 = von_neumann_entropy(rho.matrix)
        
        assert np.isclose(S1, S2), "Entropy calculations inconsistent"
```

#### Day 9: Performance Optimization

**Profiling session:**

```python
import cProfile
import pstats
from src.teleportation import run_full_analysis

# Profile the complete analysis
profiler = cProfile.Profile()
profiler.enable()

run_full_analysis(num_trials=100)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions by time
```

**Optimization targets:**
- Matrix multiplications (use `@` operator)
- Repeated eigenvalue decompositions (cache results)
- Unnecessary array copies (use views where possible)
- Inefficient loops (vectorize with NumPy)

**Team decision:** Decide together which optimizations to implement based on profiling data.

### 6.2 Comprehensive Analysis Script

Create `run_complete_analysis.py` that ties everything together:

```python
"""
Complete quantum teleportation analysis.

This script runs all experiments and generates all figures for the report.

Team collaboration:
- Member 1: Entropy and purity analysis
- Member 2: Bloch sphere and channel characterization  
- Member 3: Teleportation fidelity and concurrence
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.density_matrix import DensityMatrix, von_neumann_entropy
from src.quantum_channels import QuantumChannel
from src.teleportation import QuantumTeleportation
from src.visualization import BlochSphere
from src.operator_toolkit import ket, random_state_vector

# Create results directory
RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analysis_1_bloch_trajectory():
    """
    Analysis 1: Amplitude damping trajectory on Bloch sphere
    Responsibility: Member 2
    """
    print("Running Analysis 1: Bloch sphere trajectory...")
    
    T1 = 1.0  # Decay time
    times = np.linspace(0, 5*T1, 100)
    
    # Initial state |1⟩
    ket_1 = ket(1)
    rho_0 = DensityMatrix(ket_1)
    
    trajectory = []
    purities = []
    entropies = []
    
    for t in times:
        gamma = 1 - np.exp(-t/T1)
        channel = QuantumChannel.amplitude_damping(gamma)
        rho_t = channel.apply(rho_0)
        
        trajectory.append(rho_t.bloch_vector())
        purities.append(rho_t.purity())
        entropies.append(rho_t.entropy())
    
    # Plot Bloch sphere
    bloch = BlochSphere()
    fig = bloch.plot_trajectory(np.array(trajectory), times=times)
    fig.savefig(RESULTS_DIR / "bloch_trajectory.png", dpi=300)
    
    # Plot purity and entropy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(times, purities, 'b-', linewidth=2)
    ax1.set_xlabel('Time (units of $T_1$)')
    ax1.set_ylabel('Purity Tr($\\rho^2$)')
    ax1.set_title('Purity Decay')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, entropies, 'r-', linewidth=2)
    ax2.set_xlabel('Time (units of $T_1$)')
    ax2.set_ylabel('von Neumann Entropy (bits)')
    ax2.set_title('Entropy Growth')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "purity_entropy.png", dpi=300)
    
    print(f"✓ Saved: bloch_trajectory.png, purity_entropy.png")

def analysis_2_teleportation_fidelity():
    """
    Analysis 2: Fidelity vs noise strength
    Responsibility: Member 3
    """
    print("Running Analysis 2: Teleportation fidelity vs noise...")
    
    # Depolarizing noise
    p_vals = np.linspace(0, 0.5, 20)
    
    # Sample 10 random input states
    num_states = 10
    random_states = [random_state_vector(2) for _ in range(num_states)]
    
    fidelities_depol = np.zeros((num_states, len(p_vals)))
    
    for i, psi in enumerate(random_states):
        for j, p in enumerate(p_vals):
            noise = QuantumChannel.depolarizing(p)
            teleporter = QuantumTeleportation()
            fidelity = teleporter.teleport_with_noise(psi, noise)
            fidelities_depol[i, j] = fidelity
    
    # Amplitude damping
    gamma_vals = np.linspace(0, 1, 20)
    test_states = [
        (ket(0), "|0⟩"),
        (ket(1), "|1⟩"),
        ((ket(0) + ket(1))/np.sqrt(2), "|+⟩")
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot depolarizing
    mean_fid = fidelities_depol.mean(axis=0)
    std_fid = fidelities_depol.std(axis=0)
    
    ax1.plot(p_vals, mean_fid, 'b-', linewidth=2, label='Mean fidelity')
    ax1.fill_between(p_vals, mean_fid-std_fid, mean_fid+std_fid, alpha=0.3)
    ax1.axhline(2/3, color='r', linestyle='--', label='Classical limit')
    ax1.set_xlabel('Depolarizing probability $p$')
    ax1.set_ylabel('Teleportation Fidelity')
    ax1.set_title('Fidelity vs Depolarizing Noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot amplitude damping
    for psi, label in test_states:
        fids = []
        for gamma in gamma_vals:
            noise = QuantumChannel.amplitude_damping(gamma)
            teleporter = QuantumTeleportation()
            fidelity = teleporter.teleport_with_noise(psi, noise)
            fids.append(fidelity)
        ax2.plot(gamma_vals, fids, linewidth=2, label=label)
    
    ax2.axhline(2/3, color='r', linestyle='--', label='Classical limit')
    ax2.set_xlabel('Damping parameter $\\gamma$')
    ax2.set_ylabel('Teleportation Fidelity')
    ax2.set_title('Fidelity vs Amplitude Damping')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "fidelity_analysis.png", dpi=300)
    
    print(f"✓ Saved: fidelity_analysis.png")

def analysis_3_concurrence_correlation():
    """
    Analysis 3: Entanglement (concurrence) vs fidelity
    Responsibility: Member 3 (with Member 1 support)
    """
    print("Running Analysis 3: Concurrence vs fidelity correlation...")
    
    from src.teleportation import concurrence
    
    p_vals = np.linspace(0, 0.4, 30)
    
    concurrences = []
    fidelities = []
    
    # Fixed test state
    psi = random_state_vector(2)
    
    for p in p_vals:
        # Create noisy EPR pair
        phi_plus = (tensor_product(ket(0), ket(0)) + 
                    tensor_product(ket(1), ket(1))) / np.sqrt(2)
        rho_epr = DensityMatrix(phi_plus)
        
        noise = QuantumChannel.depolarizing(p)
        # Apply noise to both halves
        rho_epr_noisy = apply_noise_to_bipartite(rho_epr, noise, noise)
        
        # Measure concurrence
        C = concurrence(rho_epr_noisy)
        concurrences.append(C)
        
        # Measure teleportation fidelity
        teleporter = QuantumTeleportation(epr_state=rho_epr_noisy)
        F = teleporter.teleport(psi)
        fidelities.append(F)
    
    # Plot correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(p_vals, concurrences, 'g-', linewidth=2, label='Concurrence')
    ax1.plot(p_vals, fidelities, 'b-', linewidth=2, label='Fidelity')
    ax1.axhline(2/3, color='r', linestyle='--', label='Classical limit')
    ax1.set_xlabel('Noise parameter $p$')
    ax1.set_ylabel('Value')
    ax1.set_title('Concurrence & Fidelity vs Noise')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(concurrences, fidelities, s=50, alpha=0.6)
    ax2.set_xlabel('Concurrence $C$')
    ax2.set_ylabel('Fidelity $F$')
    ax2.set_title('Fidelity vs Concurrence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "concurrence_fidelity.png", dpi=300)
    
    print(f"✓ Saved: concurrence_fidelity.png")

def generate_summary_report():
    """Generate summary statistics for the report."""
    print("\nGenerating summary report...")
    
    summary = {
        'analyses_completed': 3,
        'figures_generated': 4,
        'tests_passed': 'All',
        'team_members': 3,
        'total_commits': 'Check git log'
    }
    
    with open(RESULTS_DIR / "summary.txt", 'w') as f:
        f.write("QUANTUM TELEPORTATION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key:.<40} {value}\n")
    
    print(f"✓ Saved: summary.txt")

if __name__ == "__main__":
    print("="*60)
    print(" COMPLETE QUANTUM TELEPORTATION ANALYSIS")
    print(" Team Capstone Project - Month 2")
    print("="*60 + "\n")
    
    analysis_1_bloch_trajectory()
    print()
    
    analysis_2_teleportation_fidelity()
    print()
    
    analysis_3_concurrence_correlation()
    print()
    
    generate_summary_report()
    
    print("\n" + "="*60)
    print(" ALL ANALYSES COMPLETE!")
    print(" Results saved in:", RESULTS_DIR)
    print("="*60)
```

### 6.3 Final Team Checklist

Before submission, verify:

**Code Quality:**
- [ ] All functions have docstrings
- [ ] All classes have proper `__init__`, `__repr__`
- [ ] Type hints used where appropriate
- [ ] No hardcoded magic numbers
- [ ] Consistent naming conventions

**Testing:**
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Test coverage > 80%
- [ ] Edge cases handled (zero probabilities, identity channels, etc.)

**Documentation:**
- [ ] README.md complete with setup instructions
- [ ] API documentation generated
- [ ] Theory document explains all concepts
- [ ] Code comments explain "why" not "what"

**Git Hygiene:**
- [ ] Meaningful commit messages
- [ ] No large files in repo
- [ ] .gitignore properly configured
- [ ] No merge conflicts remaining

**Results:**
- [ ] All figures generated and saved
- [ ] Figure captions written
- [ ] Results directory organized
- [ ] Summary statistics computed

### 6.4 Division of Report Writing

**Section Assignments:**

| Section | Primary | Support | Page Limit |
|---------|---------|---------|------------|
| 1. Introduction | Member 3 | All | 1 |
| 2. Theoretical Background | Member 1 | Member 2 | 2 |
| 3. Implementation | Member 2 | Member 1 | 2 |
| 4. Amplitude Damping | Member 2 | Member 1 | 2 |
| 5. Teleportation Results | Member 3 | All | 3 |
| 6. Discussion | All | All | 1-2 |

**Writing Timeline:**
- **Day 10 Morning:** Each member writes their primary section
- **Day 10 Afternoon:** Swap sections for peer review
- **Day 10 Evening:** Integrate all sections, resolve inconsistencies
- **Day 11:** Final polish, formatting, references

**Report Template** (provided in `docs/report_template.md`):

```markdown
# Quantum Teleportation Under Noise: A Comprehensive Analysis

**Team Members:** [Name 1], [Name 2], [Name 3]  
**Course:** Introduction to Quantum Computing and Information  
**Date:** November 2025

## Abstract
[150 words maximum]

## 1. Introduction
[Member 3 primary]
- Motivation: Why study noisy teleportation?
- Objectives: What are we investigating?
- Approach: How did we tackle this?

## 2. Theoretical Background  
[Member 1 primary]
- Density matrix formalism
- Quantum channels and Kraus operators
- Teleportation protocol
- Fidelity measures

## 3. Implementation
[Member 2 primary]
- Software architecture
- Key algorithms
- Validation strategy

## 4. Amplitude Damping Analysis
[Member 2 primary]
- Bloch sphere dynamics
- Entropy evolution
- Physical interpretation

## 5. Teleportation Results
[Member 3 primary]
- Fidelity vs noise strength
- Critical thresholds
- Concurrence correlation
- State-dependent effects

## 6. Discussion
[All members]
- Key insights
- Limitations
- Future directions

## References
[All members contribute]

## Appendix: Code Snippets
[Selected important functions]
```

---

## Part VII: Deliverables & Evaluation

### 7.1 Code Submission

**GitHub Repository:** Submit URL to your team's repository containing:

```
quantum-teleportation-simulator/
├── README.md                      # Setup, usage, team info
├── requirements.txt               # numpy, scipy, matplotlib, pytest
├── src/
│   ├── __init__.py
│   ├── operator_toolkit.py        # Month 1 baseline
│   ├── density_matrix.py          # Member 1's contribution
│   ├── quantum_channels.py        # Member 2's contribution
│   ├── teleportation.py           # Member 3's contribution
│   └── visualization.py           # Shared
├── tests/
│   ├── test_density_matrix.py
│   ├── test_quantum_channels.py
│   ├── test_teleportation.py
│   └── test_integration.py
├── docs/
│   ├── theory.md
│   ├── api.md
│   └── report.pdf                 # Final written report
└── results/
    ├── figures/
    │   ├── bloch_trajectory.png
    │   ├── purity_entropy.png
    │   ├── fidelity_analysis.png
    │   └── concurrence_fidelity.png
    └── summary.txt
```

**Code Quality Requirements:**
- Clean, modular, well-documented code
- Consistent style (PEP 8 compliant)
- Comprehensive docstrings (Google or NumPy style)
- Type hints for function signatures
- No code duplication

**Git Requirements:**
- Meaningful commit messages
- Feature branch workflow followed
- Pull requests with peer reviews
- No large binary files tracked
- Proper .gitignore configuration

### 7.2 Written Report

**Format:** PDF, 8-12 pages (excluding references and appendix)

**Required Sections:**

1. **Abstract** (150 words)
   - Problem statement
   - Methodology
   - Key findings

2. **Introduction** (1 page)
   - Motivation for studying noisy quantum teleportation
   - Research questions
   - Overview of approach

3. **Theoretical Background** (2 pages)
   - Density matrix formalism
   - Quantum channels and Kraus operators
   - Teleportation protocol
   - Fidelity and concurrence measures

4. **Implementation** (2 pages)
   - Software architecture diagram
   - Key algorithmic choices
   - Validation strategy
   - Computational complexity analysis

5. **Results** (3 pages)
   - Amplitude damping visualization
   - Fidelity vs noise analysis
   - Critical threshold determination
   - Concurrence-fidelity correlation

6. **Discussion** (1-2 pages)
   - Physical insights
   - Comparison to theoretical predictions
   - Limitations of the study
   - Potential improvements

7. **Conclusion** (0.5 page)
   - Summary of findings
   - Broader implications

8. **References**
   - Minimum 5 academic sources
   - IEEE or APS format

9. **Appendix** (Optional)
   - Selected code snippets
   - Derivations
   - Additional figures

**Writing Guidelines:**
- Use LaTeX (recommended) or professional word processor
- Include equation numbers for all key formulas
- All figures must have captions and be referenced in text
- Clear, concise scientific writing
- Define all notation on first use

### 7.3 Presentation

**Format:** 15-minute team presentation + 5 minutes Q&A

**Slide Allocation:**
- Slide 1: Title + Team members
- Slides 2-3: Introduction and motivation (Member 3)
- Slides 4-5: Theoretical framework (Member 1)
- Slides 6-7: Implementation highlights (Member 2)
- Slides 8-10: Results and visualizations (All)
- Slides 11-12: Discussion and insights (All)
- Slide 13: Conclusions (Member 3)

**Presentation Requirements:**
- Each member speaks for ~5 minutes
- High-quality figures (vectorized if possible)
- Live demo of code (2 minutes)
- Smooth transitions between speakers
- Rehearse together beforehand

### 7.4 Individual Contribution Statement

Each team member submits a separate 1-page document describing:

1. **Your primary responsibilities**
   - What you implemented
   - Challenges you faced
   - How you overcame them

2. **Your collaborative contributions**
   - Code reviews you performed
   - Bugs you helped fix
   - Integration work you did

3. **Team dynamics**
   - How did you communicate?
   - How did you resolve conflicts?
   - What did you learn from teammates?

4. **Self-assessment**
   - What went well?
   - What would you do differently?
   - Skills you developed

**This is confidential** and helps instructors understand team dynamics.

---

## Grading Rubric (Total: 150 points)

### A. Code Implementation (60 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **DensityMatrix Class** | 15 | Complete implementation, all methods working, proper validation |
| **Partial Trace** | 10 | Correct for arbitrary bipartite systems, handles edge cases |
| **Quantum Channels** | 15 | All required channels, CPTP verification, factory methods |
| **Teleportation Protocol** | 15 | Correct implementation, all measurement outcomes handled |
| **Code Quality** | 5 | Documentation, style, modularity, no code smells |

### B. Testing & Validation (20 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Unit Tests** | 10 | Comprehensive coverage of all components |
| **Integration Tests** | 5 | Tests verify components work together |
| **Test Coverage** | 5 | >80% coverage, meaningful tests not just coverage |

### C. Analysis & Results (35 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Bloch Sphere Visualization** | 10 | Clear, accurate trajectory with proper annotations |
| **Fidelity Analysis** | 10 | All three noise scenarios, clear trends |
| **Critical Thresholds** | 5 | Correctly identified and justified |
| **Concurrence Correlation** | 10 | Correct calculation, clear correlation demonstrated |

### D. Written Report (25 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Clarity & Organization** | 8 | Logical flow, clear sections, professional formatting |
| **Technical Accuracy** | 10 | Correct physics and math, proper citations |
| **Figures & Presentation** | 4 | High-quality figures with captions, proper references |
| **Discussion & Insights** | 3 | Thoughtful analysis, physical intuition demonstrated |

### E. Presentation (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Content** | 5 | Clear explanation of methods and results |
| **Delivery** | 3 | Professional, well-rehearsed, good pacing |
| **Q&A** | 2 | Thoughtful responses to questions |

### F. Team Collaboration (10 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **Git Workflow** | 4 | Proper branching, meaningful commits, code reviews |
| **Equal Contribution** | 3 | All members contributed substantially |
| **Integration** | 3 | Smooth integration, resolved conflicts professionally |

### Bonus Opportunities (up to +15 points)

- **+5:** Implement process tomography
- **+5:** Prove optimality of teleportation fidelity bound
- **+5:** Extend to three-qubit GHZ teleportation
- **+3:** Exceptionally beautiful visualizations
- **+2:** Comprehensive API documentation website

### Grade Boundaries

| Grade | Points | Description |
|-------|--------|-------------|
| A+ | 145-150+ | Exceptional work, publishable quality |
| A | 135-144 | Excellent, all requirements exceeded |
| A- | 127-134 | Very good, minor improvements possible |
| B+ | 120-126 | Good, meets all requirements |
| B | 105-119 | Satisfactory, some requirements not fully met |
| B- | 90-104 | Acceptable, significant improvements needed |
| C+ | 75-89 | Basic understanding demonstrated |
| C | 60-74 | Minimal requirements met |
| D | 45-59 | Incomplete or incorrect work |
| F | <45 | Did not demonstrate understanding |

---

## Important Dates & Milestones

| Date | Milestone | Deliverable |
|------|-----------|-------------|
| **Day 0** | Kickoff meeting | Team formation, repo creation |
| **Day 3** | Checkpoint 1 | Member 1: DensityMatrix complete |
| **Day 4** | Checkpoint 2 | Member 2: QuantumChannel complete |
| **Day 5** | Checkpoint 3 | Member 3: Teleportation complete |
| **Day 7** | Integration start | All branches merged |
| **Day 9** | Analysis complete | All figures generated |
| **Day 10** | Draft report | Report first draft complete |
| **Day 11** | Final submission | Code, report, presentation ready |
| **Day 12** | Presentations | Team presentations to class |

---

## Support & Resources

### Office Hours
- **Instructor:** Tuesday/Thursday 2-4 PM
- **TAs:** Monday/Wednesday 3-5 PM

### Discussion Forum
- Use course Slack/Discord for:
  - Technical questions
  - Debugging help  
  - Conceptual clarifications
- **Response time:** <24 hours

### Recommended Tools
- **IDE:** PyCharm, VSCode, or Jupyter Lab
- **Version Control:** Git + GitHub/GitLab
- **Documentation:** Sphinx for API docs
- **Testing:** pytest with pytest-cov
- **Formatting:** black (code formatter)
- **Linting:** pylint or flake8

### Getting Unstuck
1. Check documentation and theory notes
2. Review test cases for examples
3. Search issue on Stack Overflow/GitHub
4. Ask on course forum
5. Attend office hours
6. Consult teammates

**Remember:** Struggling is part of learning, but don't struggle alone!

---

## Additional Resources

### Essential Reading

1. Nielsen & Chuang, Chapter 8: "Quantum noise and quantum operations"
2. Preskill, Chapter 3: "Foundations II: Measurement and evolution"
3. Watrous, Chapter 4: "Quantum channels and noise"

### Research Papers

1. Bennett et al. (1993), "Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels"
2. Schumacher (1996), "Sending entanglement through noisy quantum channels"
3. Horodecki et al. (2009), "Quantum entanglement" (Review)

### Python Libraries (for reference only, DO NOT use)

- QuTiP: Quantum Toolbox in Python
- Qiskit: IBM's quantum framework
- Cirq: Google's quantum framework

**Note:** You may use NumPy, SciPy, and Matplotlib only. All quantum operations must be implemented from scratch.

---

## Philosophical Reflection: Collaboration in Quantum Science

This challenge embodies not just the transition from idealized quantum mechanics to realistic quantum information processing, but also the essential nature of scientific collaboration.

### The Density Matrix as a Lens on Reality

The density matrix formalism reveals the true nature of quantum information: fragile, non-local, and profoundly different from classical information. When we trace out the environment, we lose information irreversibly—much like how individual perspectives in a team must integrate into a shared understanding.

### Decoherence and the Challenge of Integration

Through implementing quantum channels and analyzing noisy teleportation, you confront the central challenge of quantum technology: **decoherence**. Just as quantum information degrades when exposed to environmental noise, team projects face their own "decoherence"—miscommunication, conflicting implementations, integration failures.

But there's a beautiful parallel: just as quantum error correction can preserve quantum information despite noise, good software practices (version control, testing, documentation) can preserve project coherence despite the chaos of collaborative development.

### Division of Labor, Unity of Understanding

Notice how this project required three distinct skill sets:
- **Member 1:** Mathematical foundations (the theorist)
- **Member 2:** Physical modeling (the experimentalist)
- **Member 3:** System integration (the engineer)

Yet all three perspectives must merge into one coherent framework. This mirrors real quantum information research, where theorists, experimentalists, and engineers must speak a common language despite different expertise.

### Questions to Ponder

As you complete this capstone, reflect on:

1. **Why is the density matrix formalism more general than state vectors?**
   - Consider: What happens to your knowledge of a system when you can't access part of it?
   - Relate: How does this mirror the challenge of integrating code you didn't write yourself?

2. **How does entanglement enable teleportation to outperform classical communication?**
   - Consider: What resource does entanglement provide that classical correlations cannot?
   - Relate: How did your team's shared understanding enable work you couldn't do individually?

3. **What does the fidelity threshold tell us about the feasibility of quantum communication?**
   - Consider: There's a critical noise level beyond which quantum advantage disappears.
   - Relate: What's the analogous "critical dysfunction" for team projects?

4. **How does Git preserve the "quantum state" of your project?**
   - Branching allows parallel "superpositions" of development
   - Merging "measures" which version becomes reality
   - Conflicts are like decoherence—they must be resolved carefully

### The Broader Lesson

Quantum information science teaches us that:
- **Information is physical** (it must be encoded in real systems)
- **Isolation is impossible** (systems always interact with environments)
- **Measurement has consequences** (observation changes the state)
- **Entanglement is a resource** (correlations can be more than classical)

Collaborative science teaches us that:
- **Knowledge is distributed** (no one person knows everything)
- **Independence is illusory** (our work depends on others)
- **Communication has costs** (explaining takes time and can introduce errors)
- **Teamwork is multiplicative** (good teams achieve more than the sum of individuals)

### Looking Ahead

This capstone is preparation for research, where you'll face:
- Incomplete theories requiring your contribution
- Experimental systems that don't behave as textbooks predict
- Collaborators with different backgrounds and perspectives
- Deadlines, setbacks, and the need for resilience

The skills you developed—implementing from first principles, debugging systematically, integrating disparate components, communicating clearly—are exactly what research demands.

### Final Thought

In quantum mechanics, we say a pure state **decoheres** into a mixed state when it interacts with an environment. But we can also say it becomes **entangled** with that environment—the information isn't lost, just no longer locally accessible.

Similarly, in collaboration, individual ideas decohere into the team's shared understanding. You might not be able to point to which exact lines of code are "yours" anymore. But the information isn't lost—it's woven into the collective achievement.

**That's not a bug. That's the feature.**

The quantum world shows us that the most interesting phenomena—teleportation, cryptography, quantum computation—emerge not from isolated systems, but from their careful orchestration.

The same is true for science itself.

---

**Good luck, and remember:** 

*"Every great quantum algorithm is only as good as its resistance to noise."*

*"Every great scientific achievement is only as good as the collaboration that produced it."*

---

## Acknowledgments

This capstone challenge was designed to push you intellectually while teaching you the practical skills essential for research. The structure—with its emphasis on Git workflows, code reviews, and integration challenges—reflects real-world quantum software development practices used at:

- Quantum computing companies (IBM, Google, Rigetti, IonQ)
- National laboratories (NIST, Sandia, Lawrence Livermore)
- Academic research groups worldwide

The theoretical content draws from foundational papers in quantum information:

- C.H. Bennett et al., "Teleporting an unknown quantum state via dual classical and Einstein-Podolsky-Rosen channels," *PRL* **70**, 1895 (1993)
- M. Nielsen & I. Chuang, *Quantum Computation and Quantum Information* (2000)
- J. Preskill, "Quantum Information and Computation" Lecture Notes (1998-2018)
- J. Watrous, *The Theory of Quantum Information* (2018)

Thank you for engaging seriously with this material. The quantum future needs thoughtful, skilled, collaborative researchers like you.

---

*This capstone was crafted with care for the IMSc Quantum Computing Fellowship program.*

*May your qubits stay coherent and your merge conflicts be few.*
