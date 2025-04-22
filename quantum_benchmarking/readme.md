# ⚛️ Quantum Optimization on Classically Hard Problems

This folder accompanies both our [arXiv paper](https://arxiv.org/abs/XXXX.XXXXX) and a four-part Medium article series focused on using quantum algorithms to solve **classically hard combinatorial problems**. Each subfolder includes QUBO formulations, algorithm implementations, and benchmarking results.

> 🧠 These problems are NP-hard and serve as practical benchmarks to test the feasibility, scalability, and resource efficiency of quantum optimization techniques.

---

## 📰 Medium Article Series (2024)

| Part | Problem                           | Article Link                      | Folder      |
|------|-----------------------------------|-----------------------------------|-------------|
| 1️⃣   | Multi-Dimensional Knapsack (MDKP) | [Read Part 1](https://medium.com/...) | `MDKP/`     |
| 2️⃣   | Maximum Independent Set (MIS)     | [Read Part 2](https://medium.com/...) | `MIS/`      |
| 3️⃣   | Quadratic Assignment Problem (QAP)| [Read Part 3](https://medium.com/...) | `QAP/`      |
| 4️⃣   | Market Share Problem (MSP)        | [Read Part 4](https://medium.com/...) | `MSP/`      |

Each article walks through the theory, QUBO formulation, and quantum optimization results using Qiskit.

---

## 🧪 Algorithms Implemented

These quantum algorithms are used to benchmark solution feasibility, optimality gaps, and qubit efficiency:

- **VQE** (Variational Quantum Eigensolver)
- **CVaR-VQE** (Expectation truncation via Conditional Value at Risk)
- **QAOA** and **MA-QAOA** (Multi-Angle QAOA)
- **PCE** (Pauli Correlation Encoding, with multi-step classical refinement)

> Simulation is performed via Qiskit's Aer backend, using both default and matrix product state (MPS) methods.  
> Results are benchmarked against classical baselines (e.g., CPLEX).

---

## 🗂 Folder Structure

Each problem subfolder contains:

- `instance_*.json` — A sample QUBO-formatted instance
- `run_vqe.py` or `run_qaoa.ipynb` — Code for solving using one or more algorithms
- `plots/` — Bar charts and comparison plots used in the article
- `README.md` — Problem description + usage instructions

Example:
```bash
cd MDKP
python run_vqe.py --instance instance_01.json
```


## 📊 Evaluation Metrics

To evaluate and compare quantum algorithms fairly, we use the following metrics across all problem instances:

- **Feasibility**: Whether the returned solution satisfies all constraints of the original problem.
- **Optimality Gap (%)**: The percentage difference between the best-known classical solution and the quantum solution.
- **Relative Solution Quality (RSQ%)**: The ratio of the quantum solution’s objective value to the optimal one, expressed as a percentage.
- **Qubit Usage**: The number of qubits required for each method, highlighting resource efficiency (especially relevant for NISQ hardware).
- **Circuit Depth & Shots**: We also track circuit depth, number of shots, and classical optimization runtime to estimate real-world performance and cost.

---

## ⚡ Quickstart Guide

To get started, navigate into any problem folder (e.g., `MDKP/`, `MIS/`) and run the scripts or open the notebooks provided. You can experiment with:

- Different quantum algorithms (VQE, CVaR-VQE, QAOA, PCE)
- Modifying instance files
- Changing optimizer settings (e.g., COBYLA, SLSQP)
- Varying ansatz depth and layout

Each folder is standalone and includes a small, ready-to-run problem instance.

---

## 📚 Learn More

- **arXiv Paper**: [A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems](https://arxiv.org/abs/2503.12121)
- **Medium Series**: [Quantum Optimization on Classically Hard Problems](https://medium.com/@_MonitSharma)
- **Main Repository**: [medium-articles](https://github.com/MonitSharma/medium_articles)

Each article in the series walks through a real combinatorial problem, the quantum formulation, and insights from simulated runs—backed by code you can run yourself.

---

## 🧠 Contributions Welcome

This benchmarking framework is intended to grow. If you want to try your own algorithm, suggest improvements, or run experiments on actual hardware:

- Add your own QUBO instances or problem domains
- Improve the encoding techniques or classical preprocessing
- Share performance logs from quantum hardware
- Submit issues or PRs to extend the framework

---

## 📜 License

This project is released under the MIT License. You’re free to use, adapt, and extend it—with credit.

---

## 👋 Author

Created and maintained by [Monit Sharma](https://github.com/MonitSharma)  
Twitter: [@_MonitSharma](https://twitter.com/_MonitSharma)  
Medium: [@YourHandle](https://medium.com/@_MonitSharma)  

