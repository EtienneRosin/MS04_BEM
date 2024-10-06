Ce projet consiste à la réalisation d'un solveur BEM (Boundary Element Method) rapide dans le cas de la diffraction d'un onde plane en 2D par un disque de rayon $a$ centré en 0. Le comportment de l'onde diffracté est décrit par :


$$
\begin{cases}
\Delta u^+ + k^2u^+ = 0, & \text{dans } \Omega^+ := \Omega \setminus \Omega^- \\
                u^+ = u^\text{inc}, &  \text{sur } \Gamma
\end{cases}
$$
où $\Omega^-$ est le disque, $\Gamma$ sa frontière, $u^\text{inc} = e^{-i \boldsymbol{k}\cdot \boldsymbol{x}}$ l'onde plane incidente.




Dans l'état actuel des choses, pour lancer notre calcul numérique de la représentation intégrale, il faut :
- se mettre dans un répertoire dans lequel on veut installer le projet
- cloner le dépo git dans ce répertoire : 
~~~bash
git clone https://github.com/EtienneRosin/MS04_TP.git
~~~

- ouvrir un terminal
- se placer dans le répertoire `MS04_TP/`
- Installer le package en mode développement
~~~bash
pip install -e .
~~~
- ouvrir le fichier `helmoltz_2d/disc_case/integral_representation_with_known_solution.py`
- l'exécuter