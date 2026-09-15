"""Labelled requests and the tool each one should reach for.

Both languages, because the assistant answers in whichever it is spoken to and
tool descriptions are written in English - which is precisely where naive
matching falls down.

Keep these phrased the way someone would actually say them out loud. A case
that quotes the tool's own words measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Case:
    utterance: str
    expected_tool: str
    #: Arguments the model should extract, where they are unambiguous.
    expected_arguments: Dict[str, str] = field(default_factory=dict)
    language: str = "fr"
    #: Turns that came before, for follow-ups that name nothing themselves.
    context: Optional[List[str]] = None


#: Tools a plausible MCP setup would add on top of the built-in ones, used to
#: push the tool count past the selection threshold.
EXTRA_TOOLS = [
    ("git__commit", "Record staged changes in the repository with a message"),
    ("git__status", "Show which files have been modified in the repository"),
    ("git__log", "Show recent commits in the repository"),
    ("git__branch", "Create or list branches in the repository"),
    ("git__diff", "Show the difference between commits in the repository"),
    ("notes__list_notes", "List the titles of every note"),
    ("notes__read_note", "Read one of the user's notes by title"),
    ("notes__write_note", "Create or update a note"),
    ("notes__delete_note", "Delete a note permanently"),
    ("calendar__list_events", "List the events on the user's calendar for a day"),
    ("calendar__create_event", "Add an event to the user's calendar"),
    ("calendar__delete_event", "Remove an event from the user's calendar"),
    ("mail__search", "Search the user's email messages"),
    ("mail__read", "Read one email message"),
    ("mail__send", "Send an email message to someone"),
    ("sqlite__query", "Run a read-only SQL query against a database"),
    ("sqlite__tables", "List the tables in a database"),
    ("weather__forecast", "Get the weather forecast for a place"),
]


CASES: List[Case] = [
    # -- files, French ---------------------------------------------------- #
    Case("crée un fichier de courses dans mes documents", "fs__write"),
    Case("écris bonjour dans le fichier test point txt", "fs__write"),
    Case("ajoute du lait à ma liste de courses", "fs__write"),
    Case("lis-moi le fichier notes point txt", "fs__read"),
    Case("qu'est-ce qu'il y a dans mon dossier documents", "fs__list"),
    Case("supprime le fichier temporaire", "fs__delete"),
    Case("supprime le répertoire des vieux logs", "fs__delete"),
    Case("cherche le mot TODO dans mes fichiers", "fs__search"),
    Case("renomme ce fichier en archive point txt", "fs__move"),
    Case("copie ce fichier dans le dossier de sauvegarde", "fs__copy"),
    Case("crée un dossier pour le projet", "fs__mkdir"),
    Case("quelle est la taille de ce fichier", "fs__stat"),
    Case("remplace bonjour par salut dans ce fichier", "fs__edit"),
    # -- files, English --------------------------------------------------- #
    Case("create a shopping list file in my documents", "fs__write", language="en"),
    Case("read me the notes file", "fs__read", language="en"),
    Case("what is in my documents folder", "fs__list", language="en"),
    Case("delete the temporary file", "fs__delete", language="en"),
    Case("find the word TODO in my files", "fs__search", language="en"),
    Case("rename this file to archive dot txt", "fs__move", language="en"),
    # -- web -------------------------------------------------------------- #
    Case("qu'est-ce qu'il y a sur le site example point com", "web__fetch"),
    Case("va voir la page d'accueil de python point org", "web__fetch"),
    Case("what does the example dot com page say", "web__fetch", language="en"),
    # -- applications ------------------------------------------------------ #
    Case("ouvre firefox", "app__launch"),
    Case("lance mon éditeur de code", "app__launch"),
    Case("open firefox please", "app__launch", language="en"),
    # -- MCP: notes -------------------------------------------------------- #
    Case("lis ma note sur les courses", "notes__read_note"),
    Case("quelles notes est-ce que j'ai", "notes__list_notes"),
    Case("efface ma note sur le plombier", "notes__delete_note"),
    # -- MCP: git ---------------------------------------------------------- #
    Case("quels fichiers ai-je modifiés dans le dépôt", "git__status"),
    Case("montre-moi les derniers commits", "git__log"),
    Case("crée une branche pour cette fonctionnalité", "git__branch"),
    # -- MCP: mail and calendar -------------------------------------------- #
    Case("envoie un mail à Marie", "mail__send"),
    Case("cherche les mails de la banque", "mail__search"),
    Case("qu'est-ce que j'ai au calendrier demain", "calendar__list_events"),
    Case("ajoute un rendez-vous chez le dentiste jeudi", "calendar__create_event"),
    # -- follow-ups, which name nothing on their own ----------------------- #
    Case(
        "et supprime-le",
        "fs__delete",
        context=["lis-moi le fichier brouillon point txt"],
    ),
    Case(
        "maintenant envoie-le",
        "mail__send",
        context=["écris un mail de remerciement à Paul"],
    ),
]
