# RL_Course_DHBW

# Projekt: Robuste Roboter-Steuerung mit Deep Reinforcement Learning

Dieses Projekt implementiert einen Soft Actor-Critic (SAC) Agenten, um simulierte Roboter in den [Gymnasium](https://gymnasium.farama.org/) (ehemals OpenAI Gym) Umgebungen zu steuern.

**Trainingsfokus:** Training des **HalfCheetah-v4** Roboters, um eine robuste Laufbewegung zu erlernen. 

## Projektstruktur

- `HalfCheetah_Training.ipynb`: Das Haupt-Jupyter-Notebook für das Training des HalfCheetah-Agenten, das den gesamten Prozess von der Umgebungseinrichtung bis zur Auswertung enthält.
- `requirements.txt`: Liste der notwendigen Python-Pakete.


## Setup

1.  **Voraussetzungen:** Stellen Sie sicher, dass Sie Python 3.8+ installiert haben. Für optimale Leistung auf Apple M1 Macs wird eine PyTorch-Installation mit MPS-Unterstützung empfohlen.

2.  **Paketinstallation:**
    Erstellen Sie eine `requirements.txt` Datei mit den folgenden Inhalten:

    ```
    gymnasium[mujoco]
    stable-baselines3[extra]
    torch # oder torch-nightly für die neueste MPS-Unterstützung
    tensorboard
    shimmy # Nur wenn dm_control Umgebungen verwendet werden sollen
    dm_control # Nur wenn dm_control Umgebungen verwendet werden sollen
    ```

    Installieren Sie die benötigten Pakete:
    ```bash
    pip install -r requirements.txt
    ```

    **Hinweis zu `mujoco`:** Die Installation von `mujoco` kann zusätzliche Schritte erfordern, insbesondere das Einrichten der MuJoCo-Lizenz und das Herunterladen der Binärdateien. Folgen Sie den Anweisungen in der [Gymnasium-Dokumentation für MuJoCo](https://gymnasium.farama.org/tutorials/installation_guides/mujoco_installation/).

3.  **Ausführung:**
    Öffnen und führen Sie das Jupyter Notebook `HalfCheetah_SAC_Training.ipynb` aus, um mit dem Training des SAC-Agenten für HalfCheetah zu beginnen.
