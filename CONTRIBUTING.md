# Contributing to gazeebo_drone

Thank you for your interest in contributing to the **GPS-Denied Visual Navigation System** (`gazeebo_drone`).

---

## 🛠️ Development Workflow

1. **Fork the Repository:** Create your own fork on GitHub.
2. **Clone & Install:**
   ```bash
   git clone https://github.com/your-username/gazeebo_drone.git
   cd gazeebo_drone
   pip install -e ".[dev]"
   ```
3. **Branching:** Create a feature or bugfix branch off `master`:
   ```bash
   git checkout -b feature/my-new-feature
   ```
4. **Run Unit Tests:**
   ```bash
   python3 -m pytest tests/ -v
   ```
5. **Linting & Formatting:** Ensure code conforms to Ruff and Mypy guidelines before opening a Pull Request.

---

## 🧪 Testing Guidelines

* Place unit tests in `tests/unit/` and integration tests in `tests/integration/`.
* Follow standard Pytest naming conventions (`test_*.py`).
* All PRs must pass the GitHub Actions CI test suite.
