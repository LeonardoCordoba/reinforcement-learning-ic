# -*- coding: utf-8 -*-
"""tp2_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gbr8-2q9foSJHYiy9GF-Mz4ZlbJPBvY7
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# %matplotlib inline

# Esta función es necesaria para ejecutar plotly en Collaboratory, en Jupyter notebook no
def enable_plotly_in_cell():
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)

class BernoulliBanditEnv(object): 
    # Clase que define el environment donde el reward es 0 o 1 dependiendo de una probabilidad p.

    def __init__(self, num_arms=10, p=None):
        self.num_arms = num_arms
        self.actions = np.arange(num_arms)     # define set of actions

        if len(p)==1:
            self.p = np.random.beta(0.5, 0.5, size=num_arms)
        elif len(p) == num_arms:
            self.p = p
        else:
            raise Exception('Number of probabilities ({}) does not correspond to number of arms ({}).'.format(len(p), num_arms))
        self.best_action = np.argmax(self.p)   # La mejor accion dado el enviroenment

    def reward(self, action):
        return np.random.binomial(1, p=self.p[action])

class Agent(object):
    # Clase que define al agente. Cada agente cuenta con una regla de decisión y una regla de aprendizaje.
    
    def __init__(self, learning_rule, decision_rule, param=None):
        self.decision_rule = decision_rule
        self.learning_rule = learning_rule

        if decision_rule == "epsilon-greedy":
            self.epsilon = param["epsilon"]
        
        if decision_rule == "UCB":
            self.UCB_param = param["UCB_param"]
        
        if decision_rule =='gradient':
            # inicializar este agente con param={'arms':k, 'learning_rate': 0.01}
            self.arms = param['arms']
            self.learning_rate = param['learning_rate']
            try:
                self.baseline = param['baseline']
            except KeyError:
                self.baseline = True
    
    def environment(self, env, init_q):
        # inicializa el environment
        self.env = env                                  
        self.k = env.num_arms                           
        self.actions = np.arange(self.k)                
        self.act_count = np.zeros(self.k)               
        self.iteration = 0     
        if self.learning_rule == "BayesianBetaPrior":
            self.alpha = np.random.uniform(size=self.k)
            self.beta = np.random.uniform(size=self.k)
        if len(init_q) == self.k:
            self.q_estimate = init_q
        else:
            raise Exception('Number of initial values ({}) does not correspond to number of arms ({}).'.format(len(init_q), self.k))

        if self.decision_rule =='gradient':
            # inicializar este agente con param={'arms':k, 'learning_rate': 0.01}
            self.H_pref = np.zeros(self.arms)
            self.probabilities = np.array([1/self.arms] * self.arms)
            
    def learn(self, a, r):
        # dada una acción y una recompenza, actualiza la value function.
        if self.learning_rule == "averaging":
            self.q_estimate[a] += 1/self.act_count[a] * (r - self.q_estimate[a])
            
        if self.learning_rule == "BayesianBetaPrior":
            self.alpha[a] += r
            self.beta[a] += 1 - r 
            
    def act(self):
        # realiza una acción.
        self.iteration += 1
        if self.decision_rule == "greedy":
            # Elijo de manera greedy, y si hay empate...
            tied_max = np.argwhere(self.q_estimate == np.max(self.q_estimate)).flatten().tolist()
            # ...tiro una moneda
            selected_action = [np.random.choice(tied_max) if len(tied_max)>1 else tied_max[0]][0]

        if self.decision_rule == "epsilon-greedy":
            # tiro una moneda con peso epsilon
            #try:
            #  sample = np.random.choice([0, 1], p=[self.epsilon, 1-self.epsilon])
            #except ValueError:
            #  print(self.epsilon)
            #  print(1-self.epsilon)
            sample = np.random.binomial(1, p=1-self.epsilon)
            # Si sale cara...
            if sample == 0:
                # elijo una accion al azar
                selected_action = np.random.choice(self.q_estimate)
                selected_action = int(selected_action)
            # Si no...
            else:
                # Elijo de manera greedy
                tied_max = np.argwhere(self.q_estimate == np.max(self.q_estimate)).flatten().tolist()
                selected_action = [np.random.choice(tied_max) if len(tied_max)>1 else tied_max[0]][0]
                selected_action = int(selected_action)
        
        if self.decision_rule == "UCB":
            # Inicio una accion vacia
            selected_action = None
            # recorro las acciones
            for a in self.actions:
                # Si una accion no la probe nunca
                if self.act_count[a] == 0:
                    # Elijo esa accion y freno
                    selected_action = a
                    break
            # Si ya probe todas las acciones
            if selected_action is None:
                # Empleo la formula para generar un q modificado
                modified_q_estimate = [self.q_estimate[a] + self.UCB_param*np.sqrt(np.log(self.iteration)/self.act_count[a]) for a in self.actions]
                # Elijo de manera greedy segun ese q modificado
                tied_max = np.argwhere(modified_q_estimate == np.max(modified_q_estimate)).flatten().tolist()
                selected_action = [np.random.choice(tied_max) if len(tied_max)>1 else tied_max[0]][0]

        if self.decision_rule == "gradient":
            # sampleo la acción a realizar (el [1][0] es por el formato que devuelve nonzero)
            ##selected_action = np.nonzero(np.random.multinomial(1, self.probabilities, size=1))[1][0]
            selected_action = np.argmax(np.random.multinomial(1, self.probabilities, size=1)[0])
            
            # calculo el reward
            r = self.env.reward(selected_action) 
            # actualizo la preferencia H según la acción que tomé como
            for a in range(self.k):
                # defino si utilizo el baseline o no
                if self.baseline:
                    r_estim = self.q_estimate[a]
                else:
                    r_estim = 0
                self.H_pref[a] += self.learning_rate * (r - r_estim) * ((selected_action == a)-self.probabilities[a])    
            # actualizo las probas de los brazos según Gibbs(Boltzmann)
            suma = sum(map(np.exp, self.H_pref))
            for a in range(self.k):
              self.probabilities[a] = np.exp(self.H_pref[a])/ suma   
        self.act_count[selected_action] += 1
        return selected_action

def simulateBandits(agents, narms, initp=None, initq=None, repetitions=1000, N=100):
    # función que realiza las simulaciones de los agentes. Se define el número de repeticiones que seran
    #  promediadas y el número de pasos N. agents es una lista de agentes.
    
    rewards = np.zeros((len(agents), repetitions, N))
    bestarm = np.zeros((len(agents), repetitions, N))
    for i, agent in enumerate(agents):
        for j in np.arange(repetitions):
            environment = BernoulliBanditEnv(num_arms=narms, p=initp)
            agent.environment(environment, initq if not(initq == None) else np.zeros(narms))
            for n in np.arange(N):
                a = agent.act()
                r = environment.reward(a)
                agent.learn(a, r)
                rewards[i, j, n] = r
                bestarm[i, j, n] = 1 if a == environment.best_action else 0
    
    return np.squeeze(np.mean(rewards, axis=1)), np.squeeze(np.mean(bestarm, axis=1))

def plot_name(agent):
    pref = agent.learning_rule 
    rule = agent.decision_rule
    if pref == "averaging":
        pref = "av" 
    if rule == "epsilon-greedy":
        name = pref + "_" + agent.decision_rule + "-e=" + str(agent.epsilon)
    elif rule == "gradient":
        if agent.baseline:
            name = pref + "_" + agent.decision_rule +"_wbsl_alpha=" + str(agent.learning_rate)
        else:
            name = pref + "_" + agent.decision_rule +"_nobsl_alpha=" + str(agent.learning_rate)
    else:
        name = pref + "_" + agent.decision_rule
    return name
    
def plot_results(agents, actions, rewards):
    rew_trace_ls = []
    cum_rew_trace_ls = []
    actions_trace_ls = []
    
    for i, j in enumerate(agents):
        agents_i = agents[i]
        actions_i = actions[i,:]
        rewards_i = rewards[i,:]
        iterations = [i for i in range(agents_i.iteration)]
     
        reward_trace = go.Scatter(
            x = iterations,
            y = rewards_i,
            mode = 'lines',
            name = plot_name(agents_i)
        )
        
        rew_trace_ls.append(reward_trace)
        
        cum_reward_trace = go.Scatter(
            x = iterations,
            y = np.cumsum(rewards_i),
            mode = 'lines',
            name = plot_name(agents_i)
        )
        
        cum_rew_trace_ls.append(cum_reward_trace)
        
        actions_trace = go.Scatter(
            x = iterations,
            y = actions_i*100,
            mode = 'lines',
            name = plot_name(agents_i)
        )
        
        actions_trace_ls.append(actions_trace)
        
        layouts = [go.Layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text='Iteración',
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text=t,
                font=dict(
                    family='Courier New, monospace',
                    size=18,
                    color='#7f7f7f'
                )
            )
        )
    ) for t in ["Reward", "Reward acumulado", "Accion correcta (%)"]]
    
    data = [rew_trace_ls, cum_rew_trace_ls, actions_trace_ls]
    
    figures = [go.Figure(data=data[i], layout=layouts[i]) for i in range(3)]

    return figures

"""# Ejercicios

1) Completar pertinentemente el código donde diga "COMPLETAR".

2) Realizar simulaciones con un bandit de 2 brazos (P = [0.4, 0.8]) para cada una de las reglas de decisión y graficar la recompensa promedio, la recompensa acumulada y el porcentaje de veces que fue elegido el mejor brazo en función de los pasos. Interprete los resultados.

3) Realizar simulaciones con un bandit de 10 brazos (P = [0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2]) para cada una de las reglas de decisión y graficar la recompensa promedio, la recompensa acumulada y el porcentaje de veces que fue elegido el mejor brazo en función de los pasos. Interprete los resultados.

4) Estudie la dependencia del hiperparametro epsilon en la regla de decisión epsilon-greedy.

## Ejercicio 2
"""

p = [0.4, 0.8]

env = BernoulliBanditEnv(num_arms=2, p=p)

agent_greedy = Agent(learning_rule="averaging", decision_rule="greedy")
agent_eg = Agent(learning_rule="averaging", decision_rule="epsilon-greedy", param={"epsilon":0.2})
agent_ucb = Agent(learning_rule="averaging", decision_rule="UCB", param={"UCB_param":1.1})
agent_gradient = Agent(learning_rule="averaging", decision_rule="gradient", param={"arms":2, "learning_rate": 0.4})

agents = [agent_greedy, agent_eg, agent_ucb, agent_gradient]
reward, bestarm = simulateBandits(agents, 2, p)

figures = plot_results(agents, bestarm, reward)

enable_plotly_in_cell()
iplot(figures[0])

enable_plotly_in_cell()
iplot(figures[1])

enable_plotly_in_cell()
iplot(figures[2])

"""### Conclusiones
Al observar los resultados obtenidos, lo primero que podemos notar es que la mejor estrategia en el sentido del reward acumulado es la regla de actualización del gradiente y UCB con muy poca diferencia. En los primeros pasos ganan UCB pero con más iteraciones es alcanzada por el gradiente, a pesar de que la regla del gradiente susceptible a los parámetros del learning rate y de la utilización o no de un baseline (ver apéndice). UCB es más ruidosa al principio para encontrar el brazo óptimo y estabilizandose luego. Por el lado del gradiente podemos observar la similitud con el método del descenso por el gradiente  no sólo en el comportamiento del reward promedio, más "suave" que el resto, sino también observando el comportamiento del porcentaje de elección donde cada vez elije mejor. La estrategia greedy es relativamente competitiva y buena en los primeros pasos, siendo estable ly ogrando mejores resultados incluso que una estrategia epsilon greedy con probabilidad 0.2 de elección al azar a largo plazo, un parámetro que estudiaremos más adelante. Esta última estrategia no logra encontrar nunca el mejor brazo ya que la elección del brazo óptimo no supera el 50% a lo largo de las iteraciones y no se observa una mejora en estas elecciones.

## Ejercicio 3
"""

p = [0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.8, 0.7, 0.5]
env = BernoulliBanditEnv(num_arms=10, p=p)

agent_greedy_10 = Agent(learning_rule="averaging", decision_rule="greedy")
agent_eg_10 = Agent(learning_rule="averaging", decision_rule="epsilon-greedy", param={"epsilon":0.2})
agent_ucb_10 = Agent(learning_rule="averaging", decision_rule="UCB", param={"UCB_param":1.1})
agent_gradient_10 = Agent(learning_rule="averaging", decision_rule="gradient", param={"arms":10, "learning_rate": 0.9})
agent_gradient_sbsl = Agent(learning_rule="averaging", decision_rule="gradient", param={"arms":10, "learning_rate": 0.9,
                            "baseline": False})

agents_10 = [agent_greedy_10, agent_eg_10, agent_ucb_10, agent_gradient_10, agent_gradient_sbsl]
reward_10, bestarm_10 = simulateBandits(agents_10, 10, p)

figures_10 = plot_results(agents_10, bestarm_10, reward_10)

enable_plotly_in_cell()
iplot(figures_10[0])

enable_plotly_in_cell()
iplot(figures_10[1])

enable_plotly_in_cell()
iplot(figures_10[2])

"""#### Conclusiones

Se realizaron diferentes pruebas y a partir de los resultados obtenidos se puede concluir lo siguiente:

    - El algoritmo greedy obtiene buenos resultados al comienzo y rápidamente converge a una solución subóptima. De esta forma, cuantas más iteraciones se ejecutan, se alcanza un resultado relativamente peor. 
    
    - El algoritmo epsilon-greedy con el epsilon elegido da los peores resultados, ya que explora mucho y no logra converger a una buena solución.
    
    - El algoritmo UCB explora muchísimo al comienzo y oscila mucho en sus recompensas. A partir de la iteración 70, aproximadamente, tiende a crecer en su porcentaje de acción correctamente elegida de manera más o menos suave. El reward acumulado que alcanza es maś alto que el greedy pero se ubica por debajo del reward acumuladoi del algoritmo gradient, en sus dos variantes.
    
    - Se probaron dos variantes del algoritmo gradient, con baseline y sin baseline. El caso con baseline (en violeta) obtiene resultados algo mejores al comienzo, mientras que sin baseline se alcanzan mejores resultados a partir de la iteración 60 aproximadamente. Se realizaron distintas pruebas, variando el learning rate y se obtuvieron mejores resultados con un learning rate suficientemente grande (en este problema 0.9 parece ser un buen valor). Esto indica que para obtener buenos resultados se necesita tener un learning rate que permita salir de óptimos locales. Este algoritmo (en sus dos variantes) obtuvo el mejor reward acumulado y alcanzó el mejor porcentaje de acción correcta al final de las iteraciones.

## Ejercicio 4
"""

p = [0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2]
env = BernoulliBanditEnv(num_arms=10, p=p)
agent_eg_10 = lambda x: Agent(learning_rule="averaging", decision_rule="epsilon-greedy", param={"epsilon":x})
agents_eps = [agent_eg_10(x) for x in [0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9]]

reward_eps, bestarm_eps = simulateBandits(agents_eps, 10, p)

figures_eps = plot_results(agents_eps, bestarm_eps, reward_eps)
enable_plotly_in_cell()
iplot(figures_eps[0])

enable_plotly_in_cell()
iplot(figures_eps[1])

enable_plotly_in_cell()
iplot(figures_eps[2])

"""#### Conclusiones
En este caso podemos apreciar que en los casos en los cuales el epsilon es muy grande el bandit explora mucho y el algoritmo no lograr converger a una buena solución estable, ya que con una alta probabilidad está eligiendo una solución al azar. 
Mientras un epsilon más chico pero mayor a 0 se mantiene la posibilidad de explorar e ir convergiendo a una mejor solución, siempre manteniendo la penalidad de, una vez encontrada la mejor solución, tener que seguir explorando.
Finalmente, si epsilon es igual a 0 tenemos el algoritmo greedy, el cual prueba todas las opciones al menos una vez y se queda con la mejor. En este caso, si elije o no la solución óptima va a depender de cuál es el primer reward que obtiene para cada acción, y luego va a elegir la acción que mejor reward le dio.

# Pruebas Adicionales

Decidimos realizar algunas pruebas extra para entender el comportamiento del baseline y del learning rate en el caso de la actualizacion del gradiente y comparar sus rendimientos.

### Analisis del alfa y del baseline en la actualización gradient

#### Con baseline
"""

p = [0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2]
env = BernoulliBanditEnv(num_arms=10, p=p)
agent_grad_10 = lambda x: Agent(learning_rule="averaging", decision_rule="gradient", 
                                param={"arms":10, "learning_rate":x})
agents_grad = [agent_grad_10(x) for x in [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]]

reward_grad, bestarm_grad = simulateBandits(agents_grad, 10, p)

figures_grad = plot_results(agents_grad, bestarm_grad, reward_grad)
enable_plotly_in_cell()
iplot(figures_grad[0])

enable_plotly_in_cell()
iplot(figures_grad[2])

"""#### Sin baseline"""

agent_grad_10_sb = lambda x: Agent(learning_rule="averaging", decision_rule="gradient", 
                                param={"arms":10, "learning_rate":x, "baseline": False})
agents_grad_sb = [agent_grad_10_sb(x) for x in [0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9]]

reward_grad_sb, bestarm_grad_sb = simulateBandits(agents_grad_sb, 10, p)

figures_grad_sb = plot_results(agents_grad_sb, bestarm_grad_sb, reward_grad_sb)
enable_plotly_in_cell()
iplot(figures_grad_sb[0])

enable_plotly_in_cell()
iplot(figures_grad_sb[2])

"""#### Conclusión
Sin baseline los mejores resultados alcanzados tiende a estar un poco por debajo con respecto a incluir el baseline. Además, un learning rate mayor tiende a dar mejores resultados, en este problema 0.9 parece ser un buen valor.
"""