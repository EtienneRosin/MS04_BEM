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




Find $p \in H^{-1/2}(\Gamma)$ s.t. :
$$\displaystyle \int_\Gamma G(\vec{x}, \vec{y})p(\vec{y}) d \Gamma(\vec{y}) = u^{\text{inc}}(\vec{x})$$
                
Which can be expressed as matricial system :
$$A\vec{p} = \vec{b}$$

With : 
$\displaystyle b_i = - \int_{\Gamma_i} u^{\text{inc}}(\vec{y}) d \Gamma(\vec{y})$, $p_i = p(\vec{y}_i)$ and 

$\displaystyle A_{ij} = \int_{\Gamma_j}\int_{\Gamma_i}G(\vec{x}, \vec{y})d \Gamma(\vec{x})d \Gamma(\vec{y})$