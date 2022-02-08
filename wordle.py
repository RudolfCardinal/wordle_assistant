#!/usr/bin/env python
"""
Wordle solver.

- By Rudolf Cardinal <rudolf@pobox.com>, from 2022-02-08.
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
from collections import Counter
from enum import Enum
from functools import total_ordering
import logging
import os
import re
from typing import Any, Dict, List, Set

from colors import color  # pip install ansicolors
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORDLIST = os.path.join(THIS_DIR, "five_letter_words.txt")
DEFAULT_OS_DICT = "/usr/share/dict/words"

WORDLEN = 5
WORD_REGEX = re.compile(rf"^[A-Z]{{{WORDLEN}}}$", re.IGNORECASE)
CHAR_ABSENT = "_"
CHAR_PRESENT = "O"
CHAR_CORRECT = "X"
FEEDBACK_REGEX = re.compile(
    rf"^[{CHAR_ABSENT}{CHAR_PRESENT}{CHAR_CORRECT}]{{{WORDLEN}}}$",
    re.IGNORECASE
)

COLOUR_FG = "white"
COLOUR_BG_ABSENT = "black"
COLOUR_BG_PRESENT_WRONG_LOCATION = "orange"
COLOUR_BG_PRESENT_RIGHT_LOCATION = "green"
STYLE = "bold"

N_GUESSES = 6

DEFAULT_SHOW_THRESHOLD = 100
DEFAULT_ADVICE_TOP_N = 10


# =============================================================================
# Enums
# =============================================================================

class CharFeedback(Enum):
    """
    Classes of Wordle feedback on each character.
    """
    ABSENT = 1
    PRESENT_WRONG_LOCATION = 2
    PRESENT_RIGHT_LOCATION = 3


# =============================================================================
# Helper functions
# =============================================================================

def coloured_char(c: str, feedback: CharFeedback) -> str:
    """
    Returns a string with ANSI codes to colour the character according to the
    feedback (and then reset afterwards).
    """
    if feedback == CharFeedback.ABSENT:
        bg = COLOUR_BG_ABSENT
    elif feedback == CharFeedback.PRESENT_WRONG_LOCATION:
        bg = COLOUR_BG_PRESENT_WRONG_LOCATION
    elif feedback == CharFeedback.PRESENT_RIGHT_LOCATION:
        bg = COLOUR_BG_PRESENT_RIGHT_LOCATION
    else:
        raise AssertionError("bug in coloured_char")
    return color(c, fg=COLOUR_FG, bg=bg, style=STYLE)


def prettylist(words: List[Any]) -> str:
    """
    Formats a wordlist.
    """
    return ", ".join(str(x) for x in words)


# =============================================================================
# Creating a wordlist
# =============================================================================

def make_wordlist(from_filename: str,
                  to_filename: str) -> None:
    n_read = 0
    n_written = 0
    seen = set()  # type: Set[str]
    with open(from_filename, "rt") as f, open(to_filename, "wt") as t:
        for line in f:
            n_read += 1
            word = line.strip()
            if WORD_REGEX.match(word):
                uppercase_word = word.upper()
                if uppercase_word not in seen:
                    t.write(uppercase_word + "\n")
                    seen.add(uppercase_word)
                    n_written += 1
    log.info(f"Read {n_read} words from {from_filename}")
    log.info(f"Wrote {n_written} ({n_letters}-letter) words to {to_filename}")


# =============================================================================
# Solving
# =============================================================================

# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------

class Clue:
    """
    Represents a word and its feedback.
    """
    def __init__(self, word: str, feedback: List[CharFeedback]):
        assert len(word) == WORDLEN
        assert len(feedback) == WORDLEN
        self.word = word.upper()
        self.feedback = feedback
        self.char_feedback_pairs = tuple(zip(self.word, self.feedback))

    @classmethod
    def read_from_user(cls) -> "Clue":
        """
        Read a word from the user, and positional feedback, and return our
        structure.
        """
        prefix1 = "-" * 57  # for alignment
        prefix2 = "." * 58
        word = ""
        while not WORD_REGEX.match(word):
            word = input(f"{prefix1}> Enter the five-letter word: ")
            word = word.strip().upper()
        feedback_str = ""
        while not FEEDBACK_REGEX.match(feedback_str):
            feedback_str = input(
                f"Enter the feedback ({CHAR_ABSENT!r} absent, "
                f"{CHAR_PRESENT!r} present but wrong location, "
                f"{CHAR_CORRECT!r} correct location): "
            )
            feedback_str = feedback_str.strip().upper()
        feedback = []  # type: List[CharFeedback]
        for f_char in feedback_str:
            if f_char == CHAR_ABSENT:
                f = CharFeedback.ABSENT
            elif f_char == CHAR_PRESENT:
                f = CharFeedback.PRESENT_WRONG_LOCATION
            elif f_char == CHAR_CORRECT:
                f = CharFeedback.PRESENT_RIGHT_LOCATION
            else:
                raise AssertionError("bug in read_from_user")
            feedback.append(f)
        clue = cls(word, feedback)
        print(f"{prefix2} You have entered this clue: {clue.as_string()}")
        return clue

    def as_string(self) -> str:
        """
        Coloured string representation.
        """
        return "".join(
            coloured_char(c, f)
            for c, f in self.char_feedback_pairs
        )

    def compatible(self, candidate_word: str) -> bool:
        """
        Is the feedback compatible with another (candidate) word?

        The other word is assumed to be in upper case and of the correct
        length.
        """
        for c_pos, c_char in enumerate(candidate_word):
            for f_pos, (f_char, feedback) in enumerate(self.char_feedback_pairs):  # noqa
                if f_char == c_char:
                    if feedback == CharFeedback.ABSENT:
                        # We have information that this character is absent
                        # from the target word. Therefore, the candidate word
                        # is wrong.
                        return False
                    elif (feedback == CharFeedback.PRESENT_WRONG_LOCATION
                            and c_pos == f_pos):
                        # We know that this character is not in this position.
                        return False
                elif (feedback == CharFeedback.PRESENT_RIGHT_LOCATION
                        and c_pos == f_pos):
                    # We know what character should be here... and it isn't.
                    return False
        return True


# -----------------------------------------------------------------------------
# Solver function
# -----------------------------------------------------------------------------

def read_words(wordlist_filename: str) -> List[str]:
    """
    Read all words from our pre-filtered wordlist.
    """
    words = []  # type: List[str]
    with open(wordlist_filename) as f:
        for line in f:
            word = line.strip()
            words.append(word)
    return sorted(words)


def get_letter_frequencies(words: List[str]) -> Dict[str, float]:
    """
    For a list of words, return a dictionary that maps each letter to its
    relative frequency (including 0.0 if absent). The relative frequencies will
    sum to (approximately) 1.

    We don't get information about >1 position per clue, so I think we should
    do this as frequency of "letters within words", not "letters", e.g. that
    the word THREE contributes only one E.
    """
    freq = {
        # We set a score of 0.0 for anything we don't encounter, and we also
        # fix the dictionary order so it displays nicely.
        chr(letter_ascii_code): 0.0
        for letter_ascii_code in range(ord('A'), ord('Z') + 1)
    }
    letter_counts = Counter()
    for word in words:
        # Count each letter only once per word, by using set(possible_word)
        # rather than possible_word as the iterable to update the counter.
        letter_counts.update(set(word))
    # For Python 3.10+, we could do:
    #   total = letter_counts.total()
    # but instead:
    total = sum(v for v in letter_counts.values())
    for letter, n in letter_counts.items():
        freq[letter] = n / total
    # log.debug(f"Frequency sum: {sum(v for v in freq.values())}")
    return freq

    # *** Not yet done: if we know there's an "E", this continues to give
    #     points for guessing E.
    # *** Not yet done: formal "reduction of possibility space" measure.


@total_ordering
class WordScore:
    def __init__(self,
                 word: str,
                 possible_words: List[str],
                 letter_frequencies_in_possible_words: Dict[str, float],
                 clues: List[Clue],
                 n_guesses_left: int):
        self.word = word
        self.possible_words = possible_words
        self.letter_freq = letter_frequencies_in_possible_words
        self.clues = clues
        self.n_guesses_left = n_guesses_left

    def __str__(self) -> str:
        return f"{self.word} ({self.score})"

    def __eq__(self, other: "WordScore") -> bool:
        return self.score == other.score

    def __lt__(self, other: "WordScore") -> bool:
        return self.score < other.score

    # -------------------------------------------------------------------------
    # A key aspect: scoring candidate guesses to provide advice.
    # -------------------------------------------------------------------------

    @property
    def score(self) -> float:
        """
        Key "thinking" algorithm. Returns a score for this word: how good would
        it be to use this as the next guess?
        """
        s = 0.0
        if self.n_guesses_left <= 1 and self.word not in self.possible_words:
            # If we have only one guess left, we must make a stab at the word
            # itself.
            return s
        for letter in set(self.word):
            s += self.letter_freq[letter]
        return s


def show_advice(all_words: List[str],
                possible_words: List[str],
                clues: List[Clue],
                n_guesses_left: int,
                top_n: int = 5) -> None:
    """
    Show advice to the user: what word should be guessed next?
    """
    letter_freq = get_letter_frequencies(possible_words)
    log.info(f"Letter frequencies in remaining possible words: {letter_freq}")

    # Any word may be a candidate for a guess, not just the possibilities --
    # for example, if we know 4/5 letters in the correct positions early on, we
    # might be better off with a guess that has lots of options for that final
    # letter, rather than guessing them sequentially in a single position.
    # Therefore, we generate a score for every word in all_words.
    options = [
        WordScore(
            word=w,
            possible_words=possible_words,
            letter_frequencies_in_possible_words=letter_freq,
            clues=clues,
            n_guesses_left=n_guesses_left
        )
        for w in all_words
    ]
    options.sort(reverse=True)  # from high to low scores

    # The thinking is done. Now we just need to present them nicely.
    top_n_options_str = prettylist(options[:top_n])
    log.info(f"Top {top_n} suggestions: {top_n_options_str}")
    best_score = options[0].score
    # Find the equal best suggestion(s)
    top_words = []  # type: List[str]
    for o in options:
        if o.score < best_score:
            break
        top_words.append(o.word)
    log.info(f"Best suggestion(s): {prettylist(top_words)}")


def solve(wordlist_filename: str,
          show_threshold: int = DEFAULT_SHOW_THRESHOLD,
          advice_top_n: int = DEFAULT_ADVICE_TOP_N) -> None:
    """
    Solve in a basic way using user guesses.
    """
    log.info("Wordle Assistant. By Rudolf Cardinal <rudolf@pobox.com>.")
    all_words = read_words(wordlist_filename)
    possibilities = all_words
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while guesses_remaining > 0:
        # Show the current situation
        this_guess = N_GUESSES - guesses_remaining + 1
        log.info(f"This is guess {this_guess}. "
                 f"Guesses remaining: {guesses_remaining}")
        log.info(f"Number of possible words: {len(possibilities)}")
        if len(possibilities) <= show_threshold:
            log.info(f"Possibilities: {prettylist(possibilities)}")
        else:
            log.info(f"Not yet showing possibilities (>{show_threshold})")

        # Provide advice
        show_advice(
            all_words=all_words,
            possible_words=possibilities,
            clues=clues,
            n_guesses_left=guesses_remaining,
            top_n=advice_top_n
        )

        # Read the results of a guess
        clue = Clue.read_from_user()
        clues.append(clue)

        # Filter. Have we solved, or failed?
        possibilities = [w for w in possibilities if clue.compatible(w)]
        if len(possibilities) == 1:
            log.info(f"Word is: {possibilities[0]}")
            return
        elif len(possibilities) == 0:
            log.error("Word is not in our dictionary!")
            return
        guesses_remaining -= 1
    log.info(F"Out of guesses! Remaining possibilities were: {possibilities}")


# =============================================================================
# Command-line entry piont
# =============================================================================

def main() -> None:
    # -------------------------------------------------------------------------
    # Arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        "Wordle solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--wordlist_filename", default=DEFAULT_WORDLIST,
        help=f"File containing all {WORDLEN}-letter words in upper case"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Be verbose"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cmd_make = "make_wordlist"
    parser_make = subparsers.add_parser(
        cmd_make,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_make.add_argument(
        "--source_dict", default=DEFAULT_OS_DICT,
        help="File of all dictionary words."
    )

    cmd_solve = "solve"
    parser_solve = subparsers.add_parser(
        cmd_solve,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_solve.add_argument(
        "--show_threshold", type=int, default=DEFAULT_SHOW_THRESHOLD,
        help="Show all possibilities when there are this many or fewer left"
    )
    parser_solve.add_argument(
        "--advice_top_n", type=int, default=DEFAULT_ADVICE_TOP_N,
        help="When showing advice, show this many top candidates"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    # -------------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------------
    if args.command == cmd_make:
        make_wordlist(args.source_dict, args.wordlist_filename)
    elif args.command == cmd_solve:
        solve(
            wordlist_filename=args.wordlist_filename,
            show_threshold=args.show_threshold,
            advice_top_n=args.advice_top_n
        )
    else:
        raise AssertionError("argument-parsing bug")


if __name__ == '__main__':
    main()
