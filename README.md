# Volunteer

## Setup Instructions

### Prerequisites

This project uses [uv](https://docs.astral.sh/uv/) for Python package and environment management.

### Install uv

#### macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Project Setup

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/nbrengle/volunteer.git
cd volunteer
uv sync
```

2. Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Development

### Running Tests

Run all tests:
```bash
uv run pytest
```

### Code Quality

Run all code quality checks (linting, formatting, type checking, and tests):
```bash
uv run pre-commit run --all-files
```