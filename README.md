#HyperTuneAI – Distributed Compute Node Platform

## Prerequisites

- Python 3.7 or higher
- make (usually pre-installed on Linux/macOS, or via MinGW/WSL on Windows)

### Build the Environment

Create a virtual environment and install all dependencies:

```bash
make build
```

## Running the Application

Start the FastAPI development server with hot-reload enabled:

```bash
make run
```

The application will be available at `http://localhost:8000`. The server will automatically reload when you make changes to your code.

## Project Structure

```
.
├── app/
│   └── main.py          # FastAPI application entry point
├── requirements.txt     # Python dependencies
├── Makefile            # Build and run commands
└── README.md           # This file
```

## Available Commands

View all available commands:

```bash
make help
```

### Command Reference

- `make build` - Create virtual environment and install dependencies
- `make run` - Run the FastAPI application with auto-reload
- `make clean` - Remove the virtual environment

## Cleaning Up

To remove the virtual environment and start fresh:

```bash
make clean
```

You'll need to run `make build` again before running the application.
