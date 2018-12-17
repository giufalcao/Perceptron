#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random

class Perceptron:    
	def __init__(redeNeural, x, y, taxa_aprendizado=0.1, epocas=1000, bias=1):
		redeNeural.amostras = amostras
		redeNeural.saidas = saidas
		redeNeural.taxa_aprendizado = taxa_aprendizado
		redeNeural.epocas = epocas
		redeNeural.bias = bias
		redeNeural.n_amostras = len(amostras)
		redeNeural.n_entradas = len(amostras[0]) 
		redeNeural.pesos = []
        
        
	def treinar(redeNeural):
        
        # Inserir o valor do limiar na posição "0" para cada amostra da lista "amostras"
		# Ex.: [[0.72, 0.82], ...] vira [[1, 0.72, 0.82], ...]
		for amostra in redeNeural.amostras:
			amostra.insert(0, redeNeural.bias)

		# Gerar valores randômicos entre 0 e 1 (pesos) conforme o número de atributos
		for i in range(redeNeural.n_entradas):
			redeNeural.pesos.append(random.random())
		# Inserir o valor do bias na posição "0" do vetor de pesos
		redeNeural.pesos.insert(0, 0)
		
		# Inicializar contador de épocas
		n_epocas = 0

		while True:
			# Inicializar variável erro
			# (quando terminar loop e erro continuar False, é pq não tem mais diferença entre valor calculado e desejado)
			erro = False

			# Para cada amostra...
			for i in range(redeNeural.n_amostras):
				# Inicializar potencial de ativação
				u = 0
				# Para cada atributo...
				for j in range(redeNeural.n_entradas + 1):
					# Multiplicar amostra e seu peso e também somar com o potencial que já tinha
					u += redeNeural.pesos[j] * redeNeural.amostras[i][j] 
				# Obter a saída da rede considerando g a função sinal
				y = redeNeural.sinal(u)

				# Verificar se a saída da rede é diferente da saída desejada
				if y != redeNeural.saidas[i]:
					# Calcular o erro
					erro_aux = redeNeural.saidas[i] - y
					# Fazer o ajuste dos pesos para cada elemento da amostra
					for j in range(redeNeural.n_entradas + 1):
						redeNeural.pesos[j] = redeNeural.pesos[j] + redeNeural.taxa_aprendizado * erro_aux * redeNeural.amostras[i][j]
					# Atualizar variável erro, já que erro é diferente de zero (existe)
					erro = True

			# Atualizar contador de épocas
			n_epocas += 1

			# Critérios de parada do loop: erro inexistente ou o número de épocas ultrapassar limite pré-estabelecido
			if not erro or (n_epocas > redeNeural.epocas):
				break

	## Testes para "novas" amostras
	def teste(redeNeural, amostra):
		# Inserir o valor do bias na posição "0" para cada amostra da lista "amostras"
		amostra.insert(0, redeNeural.bias)
		# Inicializar potencial de ativação
		u = 0
		# Para cada atributo...
		for i in range(redeNeural.n_entradas + 1):
			# Multiplicar amostra e seu peso e também somar com o potencial que já tinha
			u += redeNeural.pesos[i] * amostra[i]
		# Obter a saída da rede considerando g a função sinal
		y = redeNeural.sinal(u)
		print('Classe: %d' % y)

	## Função sinal
	def sinal(redeNeural, u):
		if u >= 0:
			return 1
		return -1

# Amostras (entrada e saída) para treinamento
amostras = [[0.72, 0.82],   [0.91, -0.69],
			[0.46, 0.80],   [0.03, 0.93],
			[0.12, 0.25],   [0.96, 0.47],
			[0.8, -0.75],   [0.46, 0.98],
			[0.66, 0.24],   [0.72, -0.15],
			[0.35, 0.01],   [-0.16, 0.84],
			[-0.04, 0.68],  [-0.11, 0.1],
			[0.31, -0.96],  [0.0, -0.26],
			[-0.43, -0.65], [0.57, -0.97],
			[-0.47, -0.03], [-0.72, -0.64],
			[-0.57, 0.15],  [-0.25, -0.43],
			[0.47, -0.88],  [-0.12, -0.9],
			[-0.58, 0.62],  [-0.48, 0.05],
			[-0.79, -0.92], [-0.42, -0.09],
			[-0.76, 0.65],  [-0.77, -0.76]]
 
saidas = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

# Chamar classe e fazer treinamento
rede = Perceptron(amostras, saidas)
rede.treinar()

# Entrando com amostra para teste
rede.teste([0.45, -0.90])
rede.teste([-0.35, 0.50])
rede.teste([-0.50, -0.50])
rede.teste([0.75, 0.90])
rede.teste([0.45, 0.20])

#sys.exit("fim de teste")