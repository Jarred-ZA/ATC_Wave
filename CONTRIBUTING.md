# Contributing to ATC Radio Monitor

Thank you for your interest in contributing to the ATC Radio Monitor project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

1. A clear description of the enhancement
2. The motivation behind it
3. Any implementation ideas you may have

### Pull Requests

1. Fork the repository
2. Create a branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python -m unittest discover`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

## Testing

Run the tests with:

```bash
python -m unittest discover
```

Or with pytest:

```bash
pytest
```

## Style Guide

- Follow PEP 8 for Python code
- Include docstrings for all functions, classes, and modules
- Write clear commit messages

## Documentation

When adding new features, please update the relevant documentation:

- README.md for user-facing features
- Docstrings for API changes
- Comments for complex code sections

## Versioning

We use [Semantic Versioning](https://semver.org/). For version updates:

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality
- PATCH version for backwards-compatible bug fixes

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).