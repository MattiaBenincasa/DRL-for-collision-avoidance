# 🤖 rl-lidar-robot

Mapless mobile robot navigation using 2D LIDAR — three approaches: **DWA** (classical), **DDQN** (paper-style), **Curriculum Learning** (multi-level progressive training).

---

## 🇬🇧 English

### 📁 Project structure

```
├── launch.sh                     # Build + roslaunch (usage: ./launch.sh DWA.launch)
├── execute.sh                    # Quick-start: training.launch (legacy)
├── src/
│   ├── storm.urdf                # Robot URDF description (LIDAR, dimensions)
│   ├── control_layer/
│   │   └── scripts/
│   │       ├── DDQNAgent.py      # 🧠 Double DQN (52→300→300→11, γ=0.99, buffer 30k)
│   │       ├── DWA.py            # 📐 Dynamic Window Approach local planner
│   │       └── MinimalController.py  # 🔧 Open-loop steering test
│   ├── platform_layer/
│   │   └── scripts/
│   │       ├── CommandManager.py  # ⚙️ Action → /cmd_vel (v=0.3, ω=-0.8..+0.8)
│   │       ├── LaserSensor.py     # 📡 /scan subscriber
│   │       ├── StatusReporter.py  # 📊 State vector (50 LIDAR + 2 goal info)
│   │       └── SimPositionTracker.py  # 📍 Robot position helper
│   ├── simulation_layer/
│   │   ├── launch/
│   │   │   ├── DWA.launch             # DWA simulation on map_1
│   │   │   ├── training_paper.launch  # DDQN training on map_2
│   │   │   ├── training_curr.launch   # Curriculum DDQN on map_curr
│   │   │   ├── testing_paper.launch   # Model evaluation on map_3
│   │   │   └── training.launch        # (Legacy — references training.py)
│   │   ├── scripts/
│   │   │   ├── training_paper.py      # Standard DDQN (3000 ep, simple reward)
│   │   │   ├── training_curr.py       # Curriculum DDQN (4 levels, shaped reward)
│   │   │   ├── testing_paper.py       # Model evaluation (300s on map_3, greedy)
│   │   │   ├── run_DWA.py             # DWA entry-point (waypoints + goal)
│   │   │   ├── CurriculumManager.py   # Curriculum logic (window, thresholds, min ep)
│   │   │   ├── SimulationManager.py   # Respawn, pause/resume, goal checking
│   │   │   ├── ParseWorld.py          # SDF parser → spawn/target Zone
│   │   │   ├── SaveMetrics.py         # CSV logger with metadata
│   │   │   └── Pose2D.py              # 2D pose dataclass
│   │   └── worlds/
│   │       ├── map_1.world        # 🟢 DWA map (corridor, 1 spawn + target)
│   │       ├── map_2.world        # 🟡 Paper training map (cylindrical obstacles)
│   │       ├── map_3.world        # 🔵 Paper testing map
│   │       └── map_curr.world     # 🟠 Curriculum map (4 progressive spawns)
│   └── models/                    # 💾 Trained models (.keras/.h5 + CSV logs)
└── utils/
    └── dlrl_analysis.py           # 📊 Metrics analysis + PDF report generator
```

### 🛠️ Prerequisites

- **ROS Noetic** (or Melodic)
- **Gazebo** (desktop-full install)
- Python 3, `tensorflow`, `numpy`

### ⚡ Quick setup

```bash
cd /home/ros/Desktop/rl-lidar-robot
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip3 install tensorflow numpy
catkin_make
source devel/setup.bash
```

Or use `./launch.sh <file.launch>` which handles build + roslaunch automatically.

### 🎯 Simulations

#### 1️⃣ DWA — Dynamic Window Approach (`DWA.launch`)

Classical local planner on **map_1** with 10 fixed waypoints.

```bash
./launch.sh DWA.launch                 # With GUI
./launch.sh 'DWA.launch gui:=false headless:=true'  # Headless
```

| Parameter | Value |
|-----------|-------|
| Linear speed | 0 – 0.5 m/s |
| Angular speed | ±0.8 rad/s |
| Prediction horizon | 4.0 s (dt=0.2s) |
| Robot radius | 0.3 m |
| `to_goal_cost_gain` | 5.0 |
| `speed_cost_gain` | 5.0 |
| `obstacle_cost_gain` | 0.5 |

- **State:** 50 LIDAR rays → obstacle map
- **Control:** Continuous (v, ω) sampled in dynamic window
- **Cost:** obstacle + to_goal + speed (hand-tuned)
- **Waypoints:** 10 → final goal (~6m path)
- **Recovery:** Rotate in place when all trajectories blocked

#### 2️⃣ DDQN — Paper-style (`training_paper.launch`)

End-to-end Double DQN training on **map_2** with simple collision-based reward.

```bash
./launch.sh training_paper.launch
```

| Parameter | Value |
|-----------|-------|
| **State** | 50 LIDAR readings (normalized /5.0) |
| **Actions** | 11 discrete, ω = -0.8 + 0.16·k, v = 0.3 m/s |
| **Reward** | +5/step, -1000 on collision |
| **Network** | 50 → Dense(300, ReLU) → Dense(300, ReLU) → 11 |
| **Training** | 3000 episodes, ε 1.0→0.05 (decay 0.995) |
| **Output** | `models/ddqn_model.keras` |

The episode ends only on collision or step limit (800 steps). No curriculum, no reward shaping. Baseline aligned with Feng et al. (Robotics 2021).

#### 3️⃣ Curriculum Learning (`training_curr.launch`)

Progressive DDQN training with 4 difficulty levels on **map_curr**.

```bash
./launch.sh training_curr.launch
```

| Parameter | Value |
|-----------|-------|
| **State** | 52 (50 LIDAR /5.0 + 2 goal info: distance/10 + angle) |
| **Actions** | 11 discrete, ω = -0.8 + 0.16·k, v = 0.3 m/s |
| **Reward** | Target +500, collision -200, timeout -50, progress 50·Δdist, movement bonus +5, time -0.05/step |
| **Network** | 52 → Dense(300, ReLU) → Dense(300, ReLU) → 11 |
| **Output** | `models/ddqn_curr_model.h5` |

**Curriculum levels:**

| Level | Spawn | Distance to target | Step limit | Min episodes | Max consecutive fails |
|-------|-------|-------------------|------------|--------------|----------------------|
| 1 | spawn_1 (near target) | ~2 m | 300 | 500 | 10 |
| 2 | spawn_2 (medium) | ~4 m | 500 | 1000 | 15 |
| 3 | spawn_3 (far, corridors) | ~6 m | 750 | 1500 | 20 |
| 4 | spawn_4 (farthest) | ~9 m | 1000 | 0 | 25 |

The curriculum prevents premature advancement: the agent must accumulate a minimum number of episodes at each level before its success rate is evaluated for promotion.

### 🧪 Model evaluation

Evaluate a trained model on **map_3** (300 seconds, ε=0, respawn on collision):

```bash
# Edit model_path in testing_paper.py (line 26) before launching
./launch.sh testing_paper.launch
```

Report example:

```
Model              m3.keras
Duration           300.1s
Total steps        2988
Collisions         0
Collision steps    0 (0.0%)
Collision-free     2988 (100.0%)
```

### 📊 Results analysis

```bash
python3 utils/dlrl_analysis.py     # Generates PDF with training curves
```

### 🧠 DDQN hyperparameters

| Parameter | training_paper | training_curr |
|-----------|---------------|---------------|
| γ (discount factor) | 0.99 | 0.99 |
| ε start | 1.0 | 1.0 |
| ε min | 0.05 | 0.05 |
| ε decay | 0.995 | 0.995 |
| Batch size | 64 | 64 |
| Replay buffer | 30,000 | 30,000 |
| Target network update | every 1000 steps | every 1000 steps |
| Learning rate | 0.001 | 0.001 |
| Architecture | 300 → 300 | 300 → 300 |
| Optimizer | Adam | Adam |

### 🗺️ Map structure

World files define structured maps using the convention `map_N`. Each map consists of three models:

| Model | Purpose |
|-------|---------|
| `map_N` | Walls and obstacles (static collision geometry) |
| `map_N_spawn` | Spawn zones (green rectangles) |
| `map_N_target` | Target zones (blue/red rectangles) |

`ParseWorld` automatically pairs spawn and target zones by matching their numeric indices (e.g., `spawn_1` → `target_1`). Zones support `<stepLimit>` metadata tags for per-map time limits.

---

## 🇮🇹 Italiano

Navigazione robot mobile senza mappa usando LIDAR 2D — tre approcci: **DWA** (classico), **DDQN** (paper), **Curriculum Learning** (multi-livello progressivo).

### 📁 Struttura del progetto

```
├── launch.sh                     # Build + roslaunch (uso: ./launch.sh DWA.launch)
├── execute.sh                    # Avvio rapido: training.launch (legacy)
├── src/
│   ├── storm.urdf                # Descrizione URDF del robot (LIDAR, dimensioni)
│   ├── control_layer/
│   │   └── scripts/
│   │       ├── DDQNAgent.py      # 🧠 Doppia DQN (52→300→300→11, γ=0.99, buffer 30k)
│   │       ├── DWA.py            # 📐 Dynamic Window Approach (pianificatore locale)
│   │       └── MinimalController.py  # 🔧 Test sterzo aperto (velocità fissa)
│   ├── platform_layer/
│   │   └── scripts/
│   │       ├── CommandManager.py  # ⚙️ Mappa azione → /cmd_vel (v=0.3, ω=-0.8..+0.8)
│   │       ├── LaserSensor.py     # 📡 Sottoscrittore /scan
│   │       ├── StatusReporter.py  # 📊 Vettore stato (50 LIDAR + 2 goal info)
│   │       └── SimPositionTracker.py  # 📍 Helper posizione robot
│   ├── simulation_layer/
│   │   ├── launch/
│   │   │   ├── DWA.launch             # Simulazione DWA su map_1
│   │   │   ├── training_paper.launch  # Addestramento DDQN su map_2
│   │   │   ├── training_curr.launch   # Addestramento curriculum su map_curr
│   │   │   ├── testing_paper.launch   # Valutazione modello su map_3
│   │   │   └── training.launch        # (Legacy — riferisce training.py inesistente)
│   │   ├── scripts/
│   │   │   ├── training_paper.py      # DDQN standard (3000 ep, reward semplice)
│   │   │   ├── training_curr.py       # DDQN curriculare (4 livelli, reward shaping)
│   │   │   ├── testing_paper.py       # Valutazione modello (300s su map_3, greedy)
│   │   │   ├── run_DWA.py             # Entry-point DWA (waypoint + goal)
│   │   │   ├── CurriculumManager.py   # Logica curriculum (finestra, soglie, min ep)
│   │   │   ├── SimulationManager.py   # Respawn, pausa/ripristino, verifica goal
│   │   │   ├── ParseWorld.py          # Parsing SDF → Zone spawn/target
│   │   │   ├── SaveMetrics.py         # Logger CSV con metadati
│   │   │   └── Pose2D.py              # Dataclass posa 2D
│   │   └── worlds/
│   │       ├── map_1.world        # 🟢 Mappa DWA (corridoio, 1 spawn + target)
│   │       ├── map_2.world        # 🟡 Mappa training paper (ostacoli cilindrici)
│   │       ├── map_3.world        # 🔵 Mappa testing paper
│   │       └── map_curr.world     # 🟠 Mappa curriculum (4 spawn progressivi)
│   └── models/                    # 💾 Modelli addestrati (.keras/.h5 + CSV log)
└── utils/
    └── dlrl_analysis.py           # 📊 Analisi metriche + generazione PDF
```

### 🛠️ Prerequisiti

- **ROS Noetic** (o Melodic)
- **Gazebo** (installazione desktop-full)
- Python 3, `tensorflow`, `numpy`

### ⚡ Setup rapido

```bash
cd /home/ros/Desktop/rl-lidar-robot
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src -r -y
pip3 install tensorflow numpy
catkin_make
source devel/setup.bash
```

Oppure usa `./launch.sh <file.launch>` che gestisce build + roslaunch automaticamente.

### 🎯 Simulazioni

#### 1️⃣ DWA — Dynamic Window Approach (`DWA.launch`)

Pianificatore locale classico su **map_1** con 10 waypoint fissi.

```bash
./launch.sh DWA.launch                 # Con GUI
./launch.sh 'DWA.launch gui:=false headless:=true'  # Headless
```

| Parametro | Valore |
|-----------|--------|
| Velocità lineare | 0 – 0.5 m/s |
| Velocità angolare | ±0.8 rad/s |
| Orizzonte predizione | 4.0 s (dt=0.2s) |
| Raggio robot | 0.3 m |
| `to_goal_cost_gain` | 5.0 |
| `speed_cost_gain` | 5.0 |
| `obstacle_cost_gain` | 0.5 |

- **Stato:** 50 raggi LIDAR → mappa ostacoli
- **Controllo:** (v, ω) continuo, campionato nella finestra dinamica
- **Costo:** obstacle + to_goal + speed (tarati manualmente)
- **Waypoint:** 10 waypoint → goal finale (~6m)
- **Recovery:** Rotazione sul posto se tutte le traiettorie sono bloccate

#### 2️⃣ DDQN — Paper (`training_paper.launch`)

Addestramento Double DQN end-to-end su **map_2** con reward basato solo su collisione.

```bash
./launch.sh training_paper.launch
```

| Parametro | Valore |
|-----------|--------|
| **Stato** | 50 letture LIDAR (normalizzate /5.0) |
| **Azioni** | 11 discrete, ω = -0.8 + 0.16·k, v = 0.3 m/s |
| **Reward** | +5/passo, -1000 in collisione |
| **Rete** | 50 → Dense(300, ReLU) → Dense(300, ReLU) → 11 |
| **Training** | 3000 episodi, ε 1.0→0.05 (decadimento 0.995) |
| **Output** | `models/ddqn_model.keras` |

L'episodio termina solo per collisione o limite passi (800). Nessun curriculum, nessun reward shaping. Baseline allineato a Feng et al. (Robotics 2021).

#### 3️⃣ Curriculum Learning (`training_curr.launch`)

Addestramento DDQN progressivo con 4 livelli di difficoltà su **map_curr**.

```bash
./launch.sh training_curr.launch
```

| Parametro | Valore |
|-----------|--------|
| **Stato** | 52 (50 LIDAR /5.0 + 2 goal info: distanza/10 + angolo) |
| **Azioni** | 11 discrete, ω = -0.8 + 0.16·k, v = 0.3 m/s |
| **Reward** | Target +500, collisione -200, timeout -50, progresso 50·Δdistanza, bonus movimento +5, tempo -0.05/passo |
| **Rete** | 52 → Dense(300, ReLU) → Dense(300, ReLU) → 11 |
| **Output** | `models/ddqn_curr_model.h5` |

**Livelli curriculum:**

| Livello | Spawn | Distanza dal target | Step limit | Min episodi | Fallimenti max consecutivi |
|---------|-------|--------------------|------------|-------------|---------------------------|
| 1 | spawn_1 (vicino) | ~2 m | 300 | 500 | 10 |
| 2 | spawn_2 (medio) | ~4 m | 500 | 1000 | 15 |
| 3 | spawn_3 (lontano, corridoi) | ~6 m | 750 | 1500 | 20 |
| 4 | spawn_4 (massima distanza) | ~9 m | 1000 | 0 | 25 |

Il curriculum previene l'avanzamento prematuro: l'agente deve accumulare un numero minimo di episodi a ogni livello prima che il tasso di successo venga valutato per la promozione.

### 🧪 Valutazione del modello

Valuta un modello addestrato su **map_3** (300 secondi, ε=0, respawn in caso di collisione):

```bash
# Modifica model_path in testing_paper.py (riga 26) prima di lanciare
./launch.sh testing_paper.launch
```

Esempio di report:

```
Model              m3.keras
Duration           300.1s
Total steps        2988
Collisions         0
Collision steps    0 (0.0%)
Collision-free     2988 (100.0%)
```

### 📊 Analisi dei risultati

```bash
python3 utils/dlrl_analysis.py     # Genera PDF con curve di training
```

### 🧠 Iperparametri DDQN

| Parametro | training_paper | training_curr |
|-----------|---------------|---------------|
| γ (fattore di sconto) | 0.99 | 0.99 |
| ε iniziale | 1.0 | 1.0 |
| ε minimo | 0.05 | 0.05 |
| ε decay | 0.995 | 0.995 |
| Batch size | 64 | 64 |
| Replay buffer | 30.000 | 30.000 |
| Aggiornamento target network | ogni 1000 passi | ogni 1000 passi |
| Learning rate | 0.001 | 0.001 |
| Architettura | 300 → 300 | 300 → 300 |
| Ottimizzatore | Adam | Adam |

### 🗺️ Struttura delle mappe

I file world definiscono mappe strutturate usando la convenzione `map_N`. Ogni mappa è composta da tre modelli:

| Modello | Scopo |
|---------|-------|
| `map_N` | Muri e ostacoli (geometria statica di collisione) |
| `map_N_spawn` | Zone di spawn (rettangoli verdi) |
| `map_N_target` | Zone target (rettangoli blu/rossi) |

`ParseWorld` abbina automaticamente zone spawn e target confrontando i loro indici numerici (es. `spawn_1` → `target_1`). Le zone supportano tag `<stepLimit>` per limiti di tempo personalizzati per mappa.

---

📝 Progetto accademico — Academic project.
