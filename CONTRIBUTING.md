# Contributing to VisionKit

Thank you for your interest in contributing to VisionKit! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. **Create a branch** for your feature or bug fix:
   ```bash
   git checkout -b feature/my-new-feature
   ```
3. **Install** development dependencies:
   ```bash
   make dev
   ```

## Development Workflow

### Running Tests

```bash
make test
```

All tests use synthetic images generated with PIL — no external test fixtures needed.

### Linting and Formatting

```bash
make lint       # Check for issues
make format     # Auto-format code
make typecheck  # Run mypy
```

### Code Style

- We use **Ruff** for linting and formatting (configured in `pyproject.toml`).
- We use **mypy** in strict mode for type checking.
- Target Python version is **3.11+**.
- Line length limit is **100 characters**.

## Pull Request Process

1. Ensure all tests pass: `make test`
2. Ensure linting passes: `make lint`
3. Ensure type checking passes: `make typecheck`
4. Update documentation if you add or change public APIs.
5. Write descriptive commit messages.
6. Open a pull request against the `main` branch.

## Adding New Features

When adding a new image processing function:

1. Add the method to `src/visionkit/core.py` on the `VisionKit` class.
2. Add any helper utilities to `src/visionkit/utils.py`.
3. Write tests in `tests/test_core.py` using synthetic images.
4. Update the README with usage examples.

## Reporting Issues

- Use GitHub Issues to report bugs.
- Include your Python version, OS, and a minimal reproducible example.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
