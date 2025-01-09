import matplotlib.pyplot as plt

def plot_scalability(sizes, times):
    # Criar o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, marker='o', linestyle='-', color='b', label='Tempo de Execução')

    # Adicionar valores concretos (labels) abaixo de cada ponto
    for i in range(len(sizes)):
        plt.text(sizes[i], times[i], f'{sizes[i]}', ha='center', va='bottom', fontsize=9, color='red')

    # Títulos e etiquetas
    plt.title('Escalabilidade - Tempo vs Tamanho do Problema')
    plt.xlabel('Tamanho do Problema (size)')
    plt.ylabel('Tempo de Execução (s)')
    plt.grid(True)
    plt.legend()

    # Mostrar o gráfico
    plt.tight_layout()
    plt.show()

# Exemplo de uso - Substitui com os teus valores
sizes = [16, 32, 50, 64, 84, 100, 128, 168, 200, 256]  # Exemplos de sizes
times = [4.260, 4.087, 3.972, 5.626, 8.097, 4.293, 17.453, 38.073, 64.416, 141.736]  # Exemplos de tempos

plot_scalability(sizes, times)

