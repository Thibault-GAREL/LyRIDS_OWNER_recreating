"""Classe de base pour tous les preprocesseurs de datasets NER.

Définit l'interface (le contrat) que chaque preprocesseur concret doit
implémenter, pour qu'on puisse les utiliser de manière interchangeable.
"""
import abc


class BasePreprocessor(abc.ABC):
    """Interface commune aux preprocesseurs de datasets NER.

    Une classe enfant est obligée d'implémenter `preprocess_and_save`,
    sinon Python refuse de l'instancier.
    """

    @abc.abstractmethod
    def preprocess_and_save(self, output_folder: str) -> None:
        """Lit le dataset source, le convertit au format OWNER, et le sauve.

        Le dossier de sortie doit contenir au moins train.json et test.json
        (et dev.json si applicable).

        Args:
            output_folder: dossier où écrire les fichiers JSON.
        """
