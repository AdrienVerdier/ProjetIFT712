from clint.textui import puts, indent, colored
from pyfiglet import Figlet
import os

class Console:
    """
    Cette classe nous fournit des méthodes de bases afin 
    de pouvoir afficher du texte dans notre console
    """

    # Variable de configuration de l'affichage du texte
    INDENT = 4

    COULEUR_ENTETE = None
    COULEUR_SECTION = None
    COULEUR_TITRE = None
    COULEUR_TEXTE = None
    COULEUR_WARNING = "green"
    COULEUR_ERREUR = "red"
    SYMBOLE = True

    @staticmethod
    def sortie(texte, couleur=None):
        """Affiche un texte dans stdout

        Args:
            texte (String): Texte à afficher
            couleur (String, optional): Couleur du texte. Defaults to `Console.COULEUR_TEXTE`.
        """
        if couleur is None:
            couleur = Console.COULEUR_TEXTE

        valeur_indentation = Console.INDENT

        a_afficher = Console.couleur(texte, couleur)
        with indent(valeur_indentation):
            puts(a_afficher)

    @staticmethod
    def titre(texte, couleur=None):
        """Affiche un texte comme un titre dans stdout

        Args:
            texte (String): Texte à afficher
            couleur (String, optional): Couleur du texte. Defaults to `Console.COULEUR_TITRE`.
        """
        if couleur is None:
            couleur = Console.COULEUR_TITRE

        a_afficher = Console.couleur("{} {}".format(">", texte), couleur)
        puts(a_afficher)

    @staticmethod
    def section(texte, delimiter='=', longueur=60, couleur=None):
        """Affiche un texte comme une section dans stdout

        Args:
            texte (String): Texte à afficher
            delimiter (String, optional): Caractère utilisé comme delimiter. Defaults to '='.
            longueur (int, optional): Longueur de l'en-tête de la section. Defaults to 60.
            couleur ([type], optional): Couleur du texte. Defaults to `Console.COULEUR_SECTION`.
        """
        if couleur is None:
            couleur = Console.COULEUR_SECTION

        if longueur > os.get_terminal_size()[0]:
            longueur = os.get_terminal_size()[0]

        delimitation = Console.couleur(delimiter * longueur, couleur)
        puts(delimitation)

        liste_texte = texte.splitlines()
        for ligne in ligne_texte:
            ligne_a_ecrire = line.strip()
            a_afficher = Console.couleur(ligne_a_ecrire, couleur)
            valeur_indentation = int((longueur - len(ligne_a_ecrire)) / 2)
            with indent(valeur_indentation):
                puts(a_afficher)
        puts(delimitation)