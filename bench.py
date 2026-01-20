import subprocess
import time
import os
import statistics

# Benchmark FENs
FENS = [
    ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Middlegame (Tactical)", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("Endgame (Pawn)", "8/8/8/k7/4P3/8/8/K7 w - - 0 1"),
    ("Endgame (Complex)", "4k3/8/3P4/8/8/8/6P1/4K3 w - - 0 1"),
    ("Deep Search (Tactical)", "2r3k1/1p1q1ppp/p1p1rb2/3p4/3P4/P1N1P2P/1P1Q1PP1/2R2RK1 w - - 0 1")
]

CPP_ENGINE_PATH = "cpp/fluxfish_cpp"
WORK_DIR = "cpp"
TRIALS_PER_FEN = 3
ITERATIONS = 100000

def run_bench():
    print("="*70)
    print(f"{'FEN Description':<30} | {'Nodes':<10} | {'NPS (Median)':<15}")
    print("-" * 70)
    
    overall_nps_list = []
    
    for label, fen in FENS:
        trial_nps = []
        
        for trial in range(TRIALS_PER_FEN):
            # Create a temporary FEN file for the engine to read
            with open("bench_temp.fen", "w") as f:
                f.write(fen)
                
            try:
                cmd = ["./fluxfish_cpp", "../bench_temp.fen", str(ITERATIONS), "10.0"]
                
                process = subprocess.Popen(
                    cmd,
                    cwd=WORK_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate()
                
                nps = 0
                nodes = 0
                for line in stdout.splitlines():
                    if "info string nodes" in line:
                        parts = line.split()
                        try:
                            # info string nodes 500000 nps 15234
                            nodes = int(parts[3])
                            nps = int(parts[5])
                        except:
                            pass
                
                if nps > 0:
                    trial_nps.append(nps)
                    
            except Exception as e:
                print(f"Error benchmarking {label} Trial {trial+1}: {e}")
            finally:
                if os.path.exists("bench_temp.fen"):
                    os.remove("bench_temp.fen")
        
        if trial_nps:
            median_nps = int(statistics.median(trial_nps))
            overall_nps_list.append(median_nps)
            print(f"{label:<30} | {ITERATIONS:<10} | {median_nps:<15}")
        else:
            print(f"{label:<30} | Error     | Error")
                
    if overall_nps_list:
        avg_nps = statistics.mean(overall_nps_list)
        print("-" * 70)
        print(f"{'AVERAGE MEDIAN NPS':<30} | {'':<10} | {int(avg_nps):<15}")
    print("="*70)

if __name__ == "__main__":
    if not os.path.exists(CPP_ENGINE_PATH):
        print(f"Error: Engine binary not found at {CPP_ENGINE_PATH}. Run 'make' in cpp/ first.")
    else:
        run_bench()
