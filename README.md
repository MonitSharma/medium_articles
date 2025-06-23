# 🧠 Medium Articles Code

Welcome to my open lab!  
This repository contains code, walkthroughs, and hands-on experiments from my **Medium articles**, ongoing research, and explorations in:

- ⚛️ Quantum Optimization on Classically Hard Problems
- 🔠 Transformers, LLMs & Attention Mechanisms
- 🤖 Reinforcement Learning and Decision Making

Each subfolder here corresponds to a topic I’ve written about or am actively researching. Every notebook is designed to be **educational, reproducible, and practically useful**.

📄 Find the theory + storytelling on Medium:  
[https://medium.com/@_monitsharma](https://medium.com/@_monitsharma)

---

## 📂 Folders

| Folder                  | Description                                                                                           |
|------------------------|-------------------------------------------------------------------------------------------------------|
| `quantum-benchmarking` | Code + notebooks from my arXiv paper & Medium series on quantum optimization and QUBO formulations   |
| `transformers-rl`      | Demos and notes on LLMs, attention, Transformers, PPO, Q-learning, and applications of deep RL        |
| `quantum-machine-learning` | Code and tutorials on QML implementations|

---

## ⚛️ Quantum Optimization Benchmarking

This series accompanies my arXiv paper:  
📄 **[A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems](https://arxiv.org/abs/2503.12121)**

And the Medium series:  
📰 **[Quantum Optimization on Classically Hard Problems](https://medium.com/@_MonitSharma)**

**Problems covered:**
- Multi-Dimensional Knapsack Problem (MDKP)
- Maximum Independent Set (MIS)
- Quadratic Assignment Problem (QAP)
- Market Share Problem (MSP)

**Algorithms implemented:**
- VQE and CVaR-VQE
- QAOA and MA-QAOA
- Pauli Correlation Encoding (PCE)

🧪 Code is modular and testable — run `run_vqe.py` or open `run_pce.ipynb` in each problem folder.

---

## 🔠 Transformers & RL (Coming Soon)

Hands-on walkthroughs for:
- LLM pretraining and masked self-attention
- Transformer architectures (GPT-style, encoder-decoder)
- PPO, A3C, Q-learning explained and implemented in PyTorch
- Use cases in games, dialogue systems, and time series

🧠 Each folder includes:
- Annotated Jupyter notebooks
- PyTorch-based examples
- Lightweight theory notes

---

## 🛠️ Installation

Clone this repo and install requirements (quantum section needs Qiskit, RL section uses PyTorch):

```bash
git clone https://github.com/MonitSharma/monit-learning-lab.git
cd monit-learning-lab
uv venv --python=python3.10
uv pip install -r requirements.txt
```


## 🤝 Contributing

This repository is meant to be a growing, community-friendly resource for learning and experimentation.  
If you find something useful, spot an issue, or want to build on top of what's here — contributions are very welcome!

### Ways to Contribute:
- ⭐ Star the repository to support the project
- 🐛 Report bugs or suggest improvements via issues
- 🧪 Add new experiments, notebooks, or problem instances
- 📖 Improve documentation and in-notebook explanations

Before submitting a PR, please make sure:
- Code is well-commented and clean
- Notebooks run top-to-bottom without errors
- You've tested changes on your end

Feel free to open an issue if you want to discuss ideas before implementing them!

---

## 👤 Author

Created and maintained by [Monit Sharma](https://github.com/MonitSharma)  

🐦 Twitter: [@_MonitSharma](https://twitter.com/_MonitSharma)  
✍️ Medium: [@MonitSharma](https://medium.com/@_monitsharma)

---

## 📜 License

This project is open-sourced under the [MIT License](LICENSE).  
Feel free to use, modify, and share — with credit.
