# main.py
import flwr as fl
from server_app import app  # adjust path if needed

if __name__ == "__main__":
    fl.run_app(app=app)
