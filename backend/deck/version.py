"""Steam Deck edition version: shown in Settings, printed on start and drawn
on the preloader (bottom right). Bump it for a Deck feature change."""
DECK_VERSION = "deck-1.10"


def label():
    """As drawn on the preloader: "v deck.1.10"."""
    return "v " + DECK_VERSION.replace("-", ".")
