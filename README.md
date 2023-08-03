import random
import numpy as np
import tensorflow as tf

# Dicionário para mapear ação para índice
acao_para_indice = {'pedra': 0, 'papel': 1, 'tesoura': 2}
indice_para_acao = {0: 'pedra', 1: 'papel', 2: 'tesoura'}

# Configurações do jogo
num_acoes = 3
num_episodios = 10000
taxa_aprendizado = 0.1
gamma = 0.9
exploracao_probabilidade_inicial = 1.0
exploracao_probabilidade_final = 0.01
exploracao_decay = 0.999
tamanho_replay_buffer = 1000
tamanho_lote = 32

# Função para escolher ação exploratória ou greedy
def escolher_acao(exploracao_probabilidade, valores_acoes):
    if random.uniform(0, 1) < exploracao_probabilidade:
        return random.randint(0, num_acoes - 1)
    else:
        return np.argmax(valores_acoes)

# Criação do modelo Q-Learning usando TensorFlow
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(num_acoes,), activation='relu'),
    tf.keras.layers.Dense(num_acoes)
])

modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado),
               loss=tf.keras.losses.MeanSquaredError())

# Inicialização do replay buffer
replay_buffer = []

# Treinamento do agente
for episodio in range(num_episodios):
    estado = np.zeros((1, num_acoes))
    estado[0][random.randint(0, num_acoes - 1)] = 1  # Inicialização aleatória do estado

    for t in range(100):  # Número máximo de passos por episódio (evitar loops infinitos)
        exploracao_probabilidade = max(exploracao_probabilidade_final,
                                       exploracao_probabilidade_inicial * (exploracao_decay ** episodio))

        # Escolher ação com base na estratégia de exploração vs. exploração
        acao = escolher_acao(exploracao_probabilidade, modelo.predict(estado)[0])

        # Realizar a ação e obter a recompensa
        proximo_estado = np.zeros((1, num_acoes))
        proximo_estado[0][random.randint(0, num_acoes - 1)] = 1  # Inicialização aleatória do próximo estado
        recompensa = 0

        # Calcular a recompensa com base nas regras do jogo
        if (acao == acao_para_indice['pedra'] and np.argmax(proximo_estado) == acao_para_indice['tesoura']) or \
                (acao == acao_para_indice['tesoura'] and np.argmax(proximo_estado) == acao_para_indice['papel']) or \
                (acao == acao_para_indice['papel'] and np.argmax(proximo_estado) == acao_para_indice['pedra']):
            recompensa = 1
        elif (acao == acao_para_indice['tesoura'] and np.argmax(proximo_estado) == acao_para_indice['pedra']) or \
                (acao == acao_para_indice['papel'] and np.argmax(proximo_estado) == acao_para_indice['tesoura']) or \
                (acao == acao_para_indice['pedra'] and np.argmax(proximo_estado) == acao_para_indice['papel']):
            recompensa = -1

        # Adicionar a transição ao replay buffer
        replay_buffer.append((estado, acao, recompensa, proximo_estado))

        # Limitar o tamanho do replay buffer
        if len(replay_buffer) > tamanho_replay_buffer:
            replay_buffer.pop(0)

        # Amostrar lotes aleatórios do replay buffer
        indices_amostra = np.random.choice(len(replay_buffer), tamanho_lote, replace=False)
        estados_lote, acoes_lote, recompensas_lote, proximos_estados_lote = zip(*[replay_buffer[i] for i in indices_amostra])

        # Calcular os valores Q usando a equação do Q-Learning
        valores_Q = modelo.predict_on_batch(np.array(estados_lote))
        valores_Q_proximos_estados = modelo.predict_on_batch(np.array(proximos_estados_lote))
        valores_Q_alvo = np.copy(valores_Q)
        for i in range(tamanho_lote):
            valores_Q_alvo[i][acoes_lote[i]] = recompensas_lote[i] + gamma * np.amax(valores_Q_proximos_estados[i])

        # Treinar o modelo usando os valores Q alvo
        modelo.train_on_batch(np.array(estados_lote), valores_Q_alvo)

        # Atualizar o estado atual
        estado = proximo_estado

# Função para permitir ao usuário jogar contra o agente treinado
def jogar_contra_agente_treinado():
    print("Bem-vindo ao Jogo Pedra, Papel e Tesoura contra o Agente Treinado!")
    print("Escolha sua jogada (pedra, papel ou tesoura) ou digite 'sair' para encerrar o jogo.")

    while True:
        escolha_usuario = input("Sua jogada: ").lower()

        if escolha_usuario == 'sair':
            break

        if escolha_usuario not in acao_para_indice:
            print("Opção inválida. Escolha novamente.")
            continue

        acao_usuario = acao_para_indice[escolha_usuario]
        acao_agente = escolher_acao(0, modelo.predict(np.zeros((1, num_acoes)))[0])
        resultado = ""
        if acao_usuario == acao_agente:
            resultado = "Empate!"
        elif (acao_usuario == acao_para_indice['pedra'] and acao_agente == acao_para_indice['tesoura']) or \
                (acao_usuario == acao_para_indice['tesoura'] and acao_agente == acao_para_indice['papel']) or \
                (acao_usuario == acao_para_indice['papel'] and acao_agente == acao_para_indice['pedra']):
            resultado = "Você ganhou!"
        else:
            resultado = "Você perdeu!"

        print(f"Você escolheu: {escolha_usuario}")
        print(f"Agente escolheu: {indice_para_acao[acao_agente]}")
        print(resultado)

if __name__ == "__main__":
    jogar_contra_agente_treinado()
