"""
SignLang AI - Central Unified CLI Launcher
Execute any component with a single command or interactive menu.
"""

import sys
import subprocess
from pathlib import Path


def print_banner():
    print("""
============================================================
  🤟 SIGNLANG AI v2.0 - Unified Project Launcher
  Real-Time Sign Language Translator (Video to Text + Speech)
============================================================
1. Live Camera HUD Translator      (Cyberpunk OpenCV Glassmorphism)
2. Interactive Web Dashboard       (Streamlit Live Browser App)
3. Evaluate Model & Benchmarks     (Confusion Matrix & Latency)
4. Train Custom Model (GRU/LSTM)   (EarlyStopping & TFLite Export)
5. Prepare & Normalize Dataset     (Wrist-relative normalizer)
6. Record New Gesture Dataset      (Interactive Data Studio)
7. Run Pipeline Unit Tests         (Verify all components)
0. Exit
------------------------------------------------------------
""")


def main():
    root_dir = Path(__file__).resolve().parent
    py = sys.executable

    def run_cmd(args):
        subprocess.run([py] + args, cwd=str(root_dir))

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower().strip()
        if cmd in ("live", "hud", "realtime"):
            run_cmd(["-m", "src.realtime_detector"])
        elif cmd in ("web", "app", "streamlit"):
            run_cmd(["-m", "streamlit", "run", "app/streamlit_app.py"])
        elif cmd in ("eval", "evaluate", "metrics"):
            run_cmd(["-m", "src.evaluate"])
        elif cmd in ("train",):
            run_cmd(["-m", "src.train"] + sys.argv[2:])
        elif cmd in ("data", "prep", "dataset"):
            run_cmd(["-m", "src.dataset_builder"])
        elif cmd in ("record", "collect"):
            run_cmd(["-m", "src.data_collection"])
        elif cmd in ("test", "tests"):
            run_cmd(["tests/test_pipeline.py"])
        else:
            print(f"Unknown command: '{cmd}'. Choose from: live, web, eval, train, data, record, test.")
        return

    while True:
        print_banner()
        choice = input("Enter choice (0-7): ").strip()
        if choice == "1":
            run_cmd(["-m", "src.realtime_detector"])
        elif choice == "2":
            run_cmd(["-m", "streamlit", "run", "app/streamlit_app.py"])
        elif choice == "3":
            run_cmd(["-m", "src.evaluate"])
        elif choice == "4":
            run_cmd(["-m", "src.train"])
        elif choice == "5":
            run_cmd(["-m", "src.dataset_builder"])
        elif choice == "6":
            run_cmd(["-m", "src.data_collection"])
        elif choice == "7":
            run_cmd(["tests/test_pipeline.py"])
        elif choice == "0":
            print("Exiting SignLang AI Launcher. Goodbye!")
            break
        else:
            print("Invalid choice. Please select 0 through 7.")


if __name__ == "__main__":
    main()
