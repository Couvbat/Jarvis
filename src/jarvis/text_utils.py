"""Small text helpers shared by command matching and tool selection."""

from __future__ import annotations

import re
import unicodedata

#: Words that carry no signal for either matching commands or picking tools.
#: Both languages, because the assistant answers in whichever it is spoken to.
STOPWORDS = frozenset("""
a an and are as at be by can could do does for from get give go had has have how
i if in into is it its me my of on or please que should so that the their them
then there these they this to up us was we were what when where which who will
with would you your

a ai as au aux avec ce ces dans de des du elle en est et eux il je la le les leur
lui ma mais me meme mes moi mon ne nos notre nous on ou par pas peux pour qu que
qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y
""".split())


def strip_accents(text: str) -> str:
    """Drop combining marks, keeping case."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalise(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse whitespace.

    Whisper's accents and punctuation vary between runs, so anything matched
    against a transcript is compared on this form.
    """
    unaccented = strip_accents(text.lower())
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in unaccented)
    return " ".join(cleaned.split())


def tokenise(text: str, keep_stopwords: bool = False) -> list[str]:
    """Split text into comparable terms.

    Identifiers are split on case and separators too, so ``fs__read_file``
    yields ``fs``, ``read`` and ``file`` - otherwise a tool name would only
    ever match itself.
    """
    # Accents come off first: "Crée" must split as one word, not "Cr" + "e".
    pieces = re.split(r"[^0-9A-Za-z]+", strip_accents(text))
    terms: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        for part in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+", piece):
            term = part.lower()
            if len(term) < 2:
                continue
            if not keep_stopwords and term in STOPWORDS:
                continue
            terms.append(term)
    return terms


#: Synonyms mapped to the English vocabulary tool descriptions are written
#: in - mostly French, but English ones too where the natural word to say is
#: not the word a description happens to use ("open" vs "launch"). Tool
#: selection compares what the user said against tool names and descriptions,
#: and those are English while the assistant answers in whichever language it
#: is spoken to - so without this, a French request matches almost nothing and
#: selection degrades to registration order.
#:
#: This is a stopgap with a known ceiling: it only covers vocabulary written
#: down here, and a third-party MCP server can use any words it likes. The
#: replacement is a multilingual embedding backend, which costs a deep learning
#: stack and should be adopted on evidence from tests/eval rather than
#: on principle.
TERM_ALIASES = {
    # files and directories
    "fichier": ("file",), "fichiers": ("file",),
    "dossier": ("directory", "folder"), "dossiers": ("directory", "folder"),
    "repertoire": ("directory", "folder"), "repertoires": ("directory", "folder"),
    "chemin": ("path",),
    "lire": ("read",), "lis": ("read",), "lit": ("read",), "lecture": ("read",),
    "ecrire": ("write",), "ecris": ("write",), "ecrit": ("write",),
    "creer": ("create", "write"), "cree": ("create", "write"),
    "nouveau": ("create", "new"), "nouvelle": ("create", "new"),
    "supprimer": ("delete", "remove"), "supprime": ("delete", "remove"),
    "effacer": ("delete", "remove"), "efface": ("delete", "remove"),
    "modifier": ("edit", "write"), "modifie": ("edit", "write"),
    "changer": ("edit", "replace"), "change": ("edit", "replace"),
    "remplacer": ("replace", "edit"), "remplace": ("replace", "edit"),
    "ajouter": ("append", "add"), "ajoute": ("append", "add"),
    "deplacer": ("move",), "deplace": ("move",),
    "renommer": ("move", "rename"), "renomme": ("move", "rename"),
    "copier": ("copy",), "copie": ("copy",),
    "lister": ("list",), "liste": ("list",), "affiche": ("list", "show"),
    "chercher": ("search", "find"), "cherche": ("search", "find"),
    "trouver": ("search", "find"), "trouve": ("search", "find"),
    "rechercher": ("search",), "recherche": ("search",),
    "taille": ("size",), "contenu": ("content",),
    # web
    "site": ("web", "page", "url"), "page": ("page", "web"),
    "internet": ("web",), "web": ("web",), "lien": ("url", "link"),
    "telecharger": ("fetch", "download"), "recupere": ("fetch",),
    # applications
    "ouvrir": ("launch", "open"), "ouvre": ("launch", "open"),
    "lancer": ("launch", "run"), "lance": ("launch", "run"),
    "demarrer": ("launch", "start"), "demarre": ("launch", "start"),
    "application": ("application",), "commande": ("command",),
    "executer": ("run", "command"), "execute": ("run", "command"),
    # common MCP domains
    "depot": ("repository", "git"), "commit": ("commit",),
    "branche": ("branch",), "modifications": ("changes", "modified"),
    "courriel": ("mail", "email"), "mail": ("mail", "email"),
    "message": ("message", "mail"), "envoyer": ("send",), "envoie": ("send",),
    "calendrier": ("calendar",), "agenda": ("calendar",),
    "evenement": ("event",), "evenements": ("event",),
    "rendez": ("event", "meeting"), "reunion": ("meeting", "event"),
    "note": ("note",), "notes": ("note",),
    "base": ("database",), "donnees": ("data", "database"),
    "requete": ("query",),
    "demain": ("tomorrow",), "aujourd": ("today",), "hui": ("today",),
    # English, where the natural spoken word is not the one in the description
    "open": ("launch", "open"), "start": ("launch", "run"),
    "run": ("launch", "run", "command"),
    "remove": ("delete", "remove"), "erase": ("delete",),
    "make": ("create",), "new": ("create",),
    "rename": ("move", "rename"), "show": ("list", "show"),
    "folder": ("directory", "folder"),
    "website": ("web", "page"),
    "email": ("mail", "email"), "repo": ("repository",),
}


def expand(terms: list[str]) -> list[str]:
    """Add the English equivalents of any French terms, keeping the originals."""
    expanded = list(terms)
    for term in terms:
        expanded.extend(TERM_ALIASES.get(term, ()))
    return expanded
