# RL_Course_DHBW

# Projekt: Robuste Roboter-Steuerung mit Deep Reinforcement Learning

Dieses Projekt implementiert einen Soft Actor-Critic (SAC) Agenten, um simulierte Roboter in den [Gymnasium](https://gymnasium.farama.org/) (ehemals OpenAI Gym) Umgebungen zu steuern.

**Trainingsfokus:** Training des **HalfCheetah-v4** Roboters, um eine robuste Laufbewegung zu erlernen. 

## Projektstruktur

- `HalfCheetah_Training.ipynb`: Das Haupt-Jupyter-Notebook für das Training des HalfCheetah-Agenten, das den gesamten Prozess von der Umgebungseinrichtung bis zur Auswertung enthält.
- `requirements.txt`: Liste der notwendigen Python-Pakete.
- `trained_sac_agent`: Verzeichnis zum Speichern der trainierten HalfCheetah-Modelle.

## Setup

1.  **Voraussetzungen:** Stellen Sie sicher, dass Sie Python 3.8+ installiert haben. Für optimale Leistung auf Apple M1 Macs wird eine PyTorch-Installation mit MPS-Unterstützung empfohlen.

2.  **Paketinstallation:**

    Installieren Sie die benötigten Pakete:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ausführung:**
    Öffnen und führen Sie das Jupyter Notebook `HalfCheetah_SAC_Training.ipynb` aus, um mit dem Training des SAC-Agenten für HalfCheetah zu beginnen.
