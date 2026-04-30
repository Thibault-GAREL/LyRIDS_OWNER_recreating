"""Classe abstraite commune à tous les trainers (MD, ET, NER)."""
import abc


class BaseTrainer(abc.ABC):
    """Contrat commun : chaque trainer concret doit savoir charger ses données,
    s'entraîner, s'évaluer, et sauver/charger ses poids.
    """

    @abc.abstractmethod
    def load_data(self, training: bool = True) -> None:
        """Charge le(s) dataset(s). Si training=False, on saute le train set."""

    @abc.abstractmethod
    def train(self) -> None:
        """Entraîne le modèle sur le train set."""

    @abc.abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Évalue le modèle sur le test set. Retourne un dict de métriques."""

    @abc.abstractmethod
    def save_model(self, folder: str) -> None:
        """Sauve les poids dans le dossier."""

    @abc.abstractmethod
    def load_model(self, folder: str) -> None:
        """Charge les poids depuis le dossier."""
