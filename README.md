# VSS-Env

## Introdução
O **VSS-Env** é um framework desenvolvido para estudar algoritmos e técnicas de *Reinforcement Learning* aplicados à categoria IEEE VSSS (*Very Small Size Soccer*). Ele fornece uma interface com o simulador FIRASim para treinar e avaliar os agentes inteligentes no contexto do futebol de robôs.

## Pré-requisitos
É necessário que tenha instalado o simulador FIRASim.
https://github.com/VSSSLeague/FIRASim

## Instalação
Se você deseja instalar a versão de desenvolvimento diretamente do GitHub:
```bash
git clone https://github.com/DaviRosimES/VSS-Env.git
```

## Uso Básico
Abaixo está um exemplo simples de como utilizar o VSS-Env:
```python
import gymnasium as gym
import VSS-Env

env = gym.make("Stricker-v0")
state = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    env.render()

env.close()
```

## Contribuição
Se deseja contribuir com o desenvolvimento do VSS-Env:
1. Faça um *fork* do repositório.
2. Clone o projeto: `git clone https://github.com/DaviRosimES/VSS-Env.git`
3. Crie uma *branch* para suas modificações: `git checkout -b minha-feature`
4. Submeta um *pull request*!

## Contato
Caso tenha dúvidas ou sugestões, entre em contato pelo e-mail: `davi.rosim@ges.inatel.br`.

