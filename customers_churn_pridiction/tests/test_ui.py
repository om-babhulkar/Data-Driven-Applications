import subprocess
import time

def test_app_runs_without_crash():
    """Ensure the Streamlit app runs successfully."""
    try:
        process = subprocess.Popen(
            ["streamlit", "run", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(5) 
        assert process.poll() is None, "Streamlit app crashed!"
    finally:
        process.terminate()
