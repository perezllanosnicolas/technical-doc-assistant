# Contributing to Technical Document Assistant

First off, thank you for considering contributing to this project! We aim to build a robust, local, and efficient RAG pipeline for technical documentation.

## Development Setup

1. **Fork and clone the repository** to your local machine.
2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies (including testing libraries):
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

## Wotkflow

1. Create a branch for your feature or bugfix: `git checkout -b feature/your-feature-name`.
2. Write your codeensuring it aligns with the existing architectural patterns (e.g., native LangChain abstractions, clear separation of retrieval and generation layers).
3. Write test for any new funcionality in the `tests/` directory. If you add a new retrieval strategy, ensure it is covered in `tests/test_retriever.py` with the appropriate Pydantic `Mock` objects.

## Testing Standards

All code changes must pass the automated test suite. Before submitting a pull request, run the test suite locally and ensure code coverage remains high (target: >90%).

    python -m pytest tests/ -v --cov=src --cov-report=term-missing

## Pull Request Process
1. Ensure your code passes all tests and does not introduce `LangChainDeprecationWarnings`.
2. Update the `README.md` if your changes add new environment variables, alter the pipeline architecture, or modify the benchmarking results.
3. Submit a Pull Request targeting the `main` branch with a clear description of the problem solved or the geature added. 

