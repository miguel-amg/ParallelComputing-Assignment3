import re
import matplotlib.pyplot as plt

def parse_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    threads = []
    real_times = []

    for i in range(len(lines)):
        if "Running with OMP_NUM_THREADS=" in lines[i]:
            threads.append(int(lines[i].split('=')[-1].strip()))
        if "real" in lines[i]:
            time_parts = re.findall(r'\d+\.?\d*', lines[i])
            real_seconds = int(time_parts[0]) * 60 + float(time_parts[1])
            real_times.append(real_seconds)
    
    return threads, real_times

# Parse both files
threads1, real_times1 = parse_file('resultado1.txt')
threads2, real_times2 = parse_file('resultado2.txt')

# Combine threads and real times into a single dataset
threads = threads1 + threads2
real_times = real_times1 + real_times2

# Sort combined data by thread count (optional for better graph visualization)
combined_data = sorted(zip(threads, real_times))
threads, real_times = zip(*combined_data)

# Calculate speedup relative to the first sequential runtime
sequential_time = real_times[0]
speedups = [sequential_time / t for t in real_times]

# Plot the combined results
plt.figure(figsize=(10, 6))
plt.plot(threads, speedups, marker='o', label='Resultados')
plt.xlabel('Numero de threads')
plt.ylabel('Speedup')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure and display
plt.savefig("combined_speedup_comparison.png")
plt.show()
