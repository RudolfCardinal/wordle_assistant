#!/usr/bin/env python
"""
Wordle assistant.

By Rudolf Cardinal <rudolf@pobox.com>, from 2022-02-08.

Wordle is:

- https://www.powerlanguage.co.uk/wordle/
- https://www.nytimes.com/games/wordle/index.html

Run self-tests with:

.. code-block:: bash

    pip install pytest
    pytest wordle.py

Test algorithms with:

.. code-block:: bash

    ./wordle.py test_performance --algorithm UnknownLetterExplorerMindingGuessCount
    ./wordle.py test_performance --algorithm UnknownLetterExplorerAmongstPossible
    ./wordle.py test_performance --algorithm PositionalExplorerAmongstPossible
    ./wordle.py test_performance --algorithm PositionalExplorerMindingGuessCount
    ./wordle.py test_performance --algorithm Eliminator
    ./wordle.py test_performance --algorithm EliminatorAmongstPossible

Then run ``compare_algorithms.R``. Specimen output:

.. code-block:: none

                                    algorithm min median     mean max prop_success
    1:   UnknownLetterExplorerAmongstPossible   1      4 4.254699  13    0.9458086
    2: UnknownLetterExplorerMindingGuessCount   1      4 4.233531   9    0.9872989
    3:      PositionalExplorerAmongstPossible   1      4 4.213378  13    0.9476715
    4:    PositionalExplorerMindingGuessCount   1      4 4.213378  13    0.9476715
    5:                             Eliminator   1      4 3.822523   7    0.9996613
    6:              EliminatorAmongstPossible   1      4 4.091448  12    0.9591871

For others' (better) work, see

- https://sonorouschocolate.com/notes/index.php?title=The_best_strategies_for_Wordle
- https://towardsdatascience.com/finding-the-best-wordle-opener-with-machine-learning-ce81331c5759
- https://www.youtube.com/watch?v=v68zYyaEmEA
  ... particularly this one.
- https://kotaku.com/wordle-starting-word-math-science-bot-algorithm-crane-p-1848496404

Not used:

- The Wordle code, or any information about its permitted guesses/answers; we
  just use the standard Linux dictionary (or another, if you prefer). This
  differs from most approaches I've seen. Wordle apparently uses a long guess
  list (~13k words) and a short possible answer list (~2.5k words); the Linux
  dictionary contains ~6k five-letter words. Wordle doesn't accept all of those
  as guesses, though!

- Information about word frequencies in English. Some code developed to use
  this, but I think this is irrelevant -- I don't think Wordle is reflecting
  English word frequency, just picking from a list (i.e. with a flat
  probability).

"""  # noqa

# =============================================================================
# Imports
# =============================================================================

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
from enum import Enum
from functools import reduce, total_ordering
# from itertools import product
import logging
from multiprocessing import cpu_count
from operator import or_
import os
import re
from statistics import median, mean
from timeit import default_timer as timer
from typing import (
    Any, Dict, Generator, Iterable, List, Optional, Sequence, Set, Tuple, Type,
    Union
)
import unittest

from colors import color  # pip install ansicolors
from cardinal_pythonlib.lists import chunks
from cardinal_pythonlib.logs import (
    configure_logger_for_colour,
    main_only_quicksetup_rootlogger,
)
from cardinal_pythonlib.maths_py import round_sf
import numpy as np
import ray

rootlog = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OS_DICT = "/usr/share/dict/words"
DEFAULT_WORDLIST = os.path.join(THIS_DIR, "five_letter_words.txt")
DEFAULT_FREQLIST = os.path.join(THIS_DIR, "five_letter_word_frequencies.txt")

# Defining the game
WORDLEN = 5
N_GUESSES = 6
# ALL_LETTERS = tuple(
#     chr(_letter_ascii_code)
#     for _letter_ascii_code in range(ord('A'), ord('Z') + 1)
# )

# Regular expressions to read from files or the user
WORD_REGEX = re.compile(rf"^[A-Z]{{{WORDLEN}}}$", re.IGNORECASE)
CHAR_ABSENT_OR_REDUNDANT = "_"
CHAR_PRESENT_WRONG_LOC = "-"
CHAR_CORRECT = "="
_FEEDBACK_REGEX_STR = (
    rf"^[\{CHAR_ABSENT_OR_REDUNDANT}"
    rf"\{CHAR_PRESENT_WRONG_LOC}"
    rf"\{CHAR_CORRECT}]{{{WORDLEN}}}$"
)
FEEDBACK_REGEX = re.compile(_FEEDBACK_REGEX_STR, re.IGNORECASE)

# Colours and styles for displaying guesses, via the ansicolors package
COLOUR_ABSENT_REDUNDANT = dict(fg="white", bg="black", style="bold")
COLOUR_PRESENT_WRONG_LOCATION = dict(fg="white", bg="yellow", style="bold")
COLOUR_PRESENT_RIGHT_LOCATION = dict(fg="white", bg="green", style="bold")

# Types
WORDSCORE_TYPE = Union[None, float, int, Iterable[Union[int, float]]]

# Defaults
DEFAULT_SHOW_THRESHOLD = 100
DEFAULT_ADVICE_TOP_N = 10
DEFAULT_NPROC = cpu_count()
DEFAULT_SIG_FIGURES = 3


# =============================================================================
# Enums
# =============================================================================

class CharFeedback(Enum):
    """
    Possible types of feedback about each character.
    """
    ABSENT_OR_REDUNDANT = 1
    PRESENT_WRONG_LOCATION = 2
    PRESENT_RIGHT_LOCATION = 3

    @property
    def plain_str(self) -> str:
        """
        Plain string representation.
        """
        if self == CharFeedback.ABSENT_OR_REDUNDANT:
            return CHAR_ABSENT_OR_REDUNDANT
        elif self == CharFeedback.PRESENT_WRONG_LOCATION:
            return CHAR_PRESENT_WRONG_LOC
        elif self == CharFeedback.PRESENT_RIGHT_LOCATION:
            return CHAR_CORRECT
        else:
            raise AssertionError("bug")


# =============================================================================
# Helper functions
# =============================================================================

# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

def colourful_char(x: str, feedback: CharFeedback) -> str:
    """
    Returns a string with ANSI codes to colour the character according to the
    feedback (and then reset afterwards).
    """
    if feedback == CharFeedback.ABSENT_OR_REDUNDANT:
        colour_params = COLOUR_ABSENT_REDUNDANT
    elif feedback == CharFeedback.PRESENT_WRONG_LOCATION:
        colour_params = COLOUR_PRESENT_WRONG_LOCATION
    elif feedback == CharFeedback.PRESENT_RIGHT_LOCATION:
        colour_params = COLOUR_PRESENT_RIGHT_LOCATION
    else:
        raise AssertionError("bug")
    return color(x, **colour_params)


def prettylist(words: Iterable[Any]) -> str:
    """
    Formats a wordlist.
    """
    return ", ".join(str(x) for x in words)


def convert_sf(x: WORDSCORE_TYPE,
               sig_fig: int = DEFAULT_SIG_FIGURES) -> WORDSCORE_TYPE:
    """
    Formats things to a certain number of significant figures.
    """
    if x is None or isinstance(x, int):
        return x
    if isinstance(x, float):
        return round_sf(x, sig_fig)
    results = []
    for y in x:
        if isinstance(y, float):
            results.append(round_sf(y, sig_fig))
        else:
            results.append(y)
    return results


# -----------------------------------------------------------------------------
# Reading word lists
# -----------------------------------------------------------------------------

def make_wordlist(from_filename: str,
                  to_filename: str) -> None:
    """
    Reads a dictionary file and creates a list of 5-letter words.
    """
    rootlog.info(f"Reading from {from_filename}")
    rootlog.info(f"Writing to {to_filename}")
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
    rootlog.info(f"Read {n_read} words from {from_filename}")
    rootlog.info(f"Wrote {n_written} ({WORDLEN}-letter) words to {to_filename}")


def make_np_array_words(words: List[str]) -> np.array:
    """
    Converts to an appropriate Numpy array type.
    """
    return np.array(words, dtype=f"U{WORDLEN}")


def read_words(wordlist_filename: str,
               max_n: int = None) -> np.array:
    """
    Read all words, with accompanying frequencies if known, from our
    pre-filtered wordlist.
    """
    words = []  # type: List[str]
    with open(wordlist_filename) as f:
        n_read = 0
        for line in f:
            word = line.strip()
            words.append(word)
            n_read += 1
            if max_n is not None and n_read >= max_n:
                rootlog.warning(f"Reading only {n_read} words")
                break
    return make_np_array_words(sorted(words))


_ = '''
def make_frequency_list(url: str,
                        to_filename: str,
                        encoding: str = "utf-8") -> None:
    """
    Reads a URL and writes a word/frequency pair file.
    """
    log.info(f"Reading from {url}")
    log.info(f"Writing to {to_filename}")
    n_read = 0
    n_written = 0
    with urlopen(url) as f, open(to_filename, "wt") as t:
        contents = f.read().decode(encoding)
        for line in contents.split("\n"):
            elements = line.split(" ")
            if len(elements) != 2:
                continue
            n_read += 1
            word, frequency = elements
            if not WORD_REGEX.match(word):
                continue
            word = word.upper()
            t.write(f"{word} {frequency}\n")
            n_written += 1
    log.info(f"Read {n_read} words from {url}")
    log.info(f"Wrote {n_written} ({WORDLEN}-letter) words to {to_filename}")


def read_words_frequencies(wordlist_filename: str,
                           frequencylist_filename: str = None,
                           default_frequency: int = 1,
                           max_n: int = None) -> Dict[str, int]:
    """
    Read all words, with accompanying frequencies if known, from our
    pre-filtered wordlist.
    """
    words = read_words(wordlist_filename, max_n)
    freqdict = {}  # type: Dict[str, int]
    if frequencylist_filename:
        with open(frequencylist_filename) as f:
            for line in f:
                elements = line.strip().split(" ")
                if len(elements) != 2:
                    continue
                word, frequency = elements
                freqdict[word] = frequency
    d = {}  # type: Dict[str, int]
    for w in sorted(words):
        d[w] = freqdict.get(w, default_frequency)
    return d


def get_letter_frequencies_positional(words: Set[str]) \
        -> List[Dict[str, float]]:
    """
    For a set of words, return a list of length WORDLEN, each element of which
    is a dictionary mapping each letter to its relative frequency.
    """
    freqlist = []  # type: List[Dict[str, float]]
    for pos in range(WORDLEN):
        freq = {letter: 0.0 for letter in ALL_LETTERS}
        letter_counts = Counter()
        for word in words:
            letter_counts.update(word[pos])
        total = sum(v for v in letter_counts.values())
        for letter, n in letter_counts.items():
            freq[letter] = n / total
        freqlist.append(freq)
    return freqlist

'''


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

@contextmanager
def time_section(name: str,
                 loglevel: int = logging.DEBUG) -> Generator[None, None, None]:
    start = timer()
    try:
        yield
    finally:
        end = timer()
        rootlog.log(loglevel, f"{name} took {end - start} s")


# =============================================================================
# Solving
# =============================================================================

# -----------------------------------------------------------------------------
# Clue
# -----------------------------------------------------------------------------

class Clue:
    """
    Represents a word and its feedback.
    """
    # _ALL_FEEDBACK_CHARS = (
    #     CharFeedback.ABSENT_OR_REDUNDANT,
    #     CharFeedback.PRESENT_WRONG_LOCATION,
    #     CharFeedback.PRESENT_RIGHT_LOCATION,
    # )
    # ALL_FEEDBACK_COMBINATIONS = tuple(
    #     product(*([_ALL_FEEDBACK_CHARS] * WORDLEN))
    # )

    def __init__(self, word: str, feedback: Sequence[CharFeedback]):
        """
        Args:

            word: the word being guessed
            feedback: character-by-character feedback
        """
        assert len(word) == WORDLEN
        assert len(feedback) == WORDLEN
        self.clue_word = word.upper()
        self.feedback = tuple(feedback)
        self.char_feedback_pairs = tuple(zip(self.clue_word, self.feedback))
        present_right_loc = set(
            c
            for c, f in self.char_feedback_pairs
            if f == CharFeedback.PRESENT_RIGHT_LOCATION
        )
        present_wrong_loc = set(
            c
            for c, f in self.char_feedback_pairs
            if f == CharFeedback.PRESENT_WRONG_LOCATION
        )
        self.present = present_right_loc.union(present_wrong_loc)
        self.absent = set(
            c
            for c, f in self.char_feedback_pairs
            if f == CharFeedback.ABSENT_OR_REDUNDANT
            and c not in self.present
        )

    # -------------------------------------------------------------------------
    # Creation
    # -------------------------------------------------------------------------

    @classmethod
    def read_from_user(cls) -> "Clue":
        """
        Read a word from the user, and positional feedback from the user
        likewise (from the online game), and return our structure.
        """
        prefix1 = "-" * 57  # for presentational alignment
        prefix2 = "." * 58
        # We allow mis-entry of the feedback string to take you back to word
        # entry.
        word = ""
        feedback_str = ""
        while not (WORD_REGEX.match(word)
                   and FEEDBACK_REGEX.match(feedback_str)):
            word = ""
            while not WORD_REGEX.match(word):
                word = input(
                    f"{prefix1}> Enter the five-letter word: "
                ).strip().upper()
            feedback_str = input(
                f"Enter the feedback ({CHAR_ABSENT_OR_REDUNDANT!r} absent, "
                f"{CHAR_PRESENT_WRONG_LOC!r} present but wrong location, "
                f"{CHAR_CORRECT!r} correct location): "
            ).strip().upper()
        clue = cls.get_from_strings(word, feedback_str)
        print(f"{prefix2} You have entered this clue: {clue}")
        return clue

    @classmethod
    def feedback_from_str(cls, feedback_str: str) -> List[CharFeedback]:
        """
        Create coded feedback from a string.
        """
        assert FEEDBACK_REGEX.match(feedback_str)
        feedback = []  # type: List[CharFeedback]
        for f_char in feedback_str:
            if f_char == CHAR_ABSENT_OR_REDUNDANT:
                f = CharFeedback.ABSENT_OR_REDUNDANT
            elif f_char == CHAR_PRESENT_WRONG_LOC:
                f = CharFeedback.PRESENT_WRONG_LOCATION
            elif f_char == CHAR_CORRECT:
                f = CharFeedback.PRESENT_RIGHT_LOCATION
            else:
                raise AssertionError("bug in read_from_user")
            feedback.append(f)
        return feedback

    @classmethod
    def get_from_strings(cls, guess: str, feedback_str: str) -> "Clue":
        """
        Use our internal string format to create a clue object.
        """
        assert WORD_REGEX.match(guess)
        feedback = cls.feedback_from_str(feedback_str)
        return cls(guess, feedback)

    # -------------------------------------------------------------------------
    # Thinking one way: clue from guess/target.
    # -------------------------------------------------------------------------

    @classmethod
    def get_from_known_word(cls, guess: str, target: str) -> "Clue":
        """
        For automatic testing, and some algorithms: given that we know the
        word (or given that we hypothesise that this word is the target),
        return the clue for a given guess.

        There is some subtlety here. I think Wordle must have some concept of
        "available to guess", and prioritize this (1) guess in correct
        location, (2) guess in incorrect location, sequenced from the start.
        For example, if you have a word with two Es and you guess one in the
        correct location, the other is marked as "absent[/redundant]", not "in
        the wrong place". Similarly, if you have a word with an E in and you
        guess a word with two Es in the wrong place, only the first gets marked
        "wrong place", and the second gets marked "absent/redundant".

        The self-testing framework checks against some proven results.
        """
        feedback = []  # type: List[CharFeedback]
        letters_available = list(target)  # splits string into letters
        # Prioritize correct locations.
        for g_pos, g_char in enumerate(guess):
            if target[g_pos] == g_char:
                letters_available.remove(g_char)
        # Then work in sequence.
        for g_pos, g_char in enumerate(guess):
            if target[g_pos] == g_char:
                f = CharFeedback.PRESENT_RIGHT_LOCATION
            elif g_char in target and g_char in letters_available:
                f = CharFeedback.PRESENT_WRONG_LOCATION
                letters_available.remove(g_char)
            else:
                f = CharFeedback.ABSENT_OR_REDUNDANT
            feedback.append(f)
        return cls(guess, feedback)

    # -------------------------------------------------------------------------
    # Thinking the other way: checking a guess against a clue
    # -------------------------------------------------------------------------

    def compatible(self, guess: str) -> bool:
        """
        Key thinking function: uses the information from a clue to eliminate
        possibilities.
        """
        guess_available_pos = set(range(WORDLEN))
        # Check "known correct position" letters match
        for pos in range(WORDLEN):
            if self.feedback[pos] == CharFeedback.PRESENT_RIGHT_LOCATION:
                # The clue contains the correct letter at this location.
                # If the guess doesn't match here, it's wrong.
                if guess[pos] != self.clue_word[pos]:
                    return False
                guess_available_pos.remove(pos)
        # Next, "present but wrong location" ones.
        for cluepos in range(WORDLEN):
            if self.feedback[cluepos] == CharFeedback.PRESENT_WRONG_LOCATION:
                # The clue contains a letter here that must be present in the
                # guess, IN ADDITION to any identical letter that has already
                # been matched for position. And it must NOT be at this
                # location itself.
                cluechar = self.clue_word[cluepos]
                # (1) Not at the same location.
                if guess[cluepos] == cluechar:
                    # Exact match; that's wrong.
                    return False
                # (2) Somewhere else (but not a correct/known location).
                found = False
                for guesspos in guess_available_pos:
                    if guess[guesspos] == cluechar:
                        found = True
                        guess_available_pos.remove(guesspos)
                        break
                if not found:
                    # Nope, it's not here.
                    return False
        # Finally, any others are absent.
        for cluepos in range(WORDLEN):
            if self.feedback[cluepos] == CharFeedback.ABSENT_OR_REDUNDANT:
                cluechar = self.clue_word[cluepos]
                for guesspos in guess_available_pos:
                    if guess[guesspos] == cluechar:
                        # Present, but shouldn't be
                        return False
        # OK! It's a match
        return True

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    def __eq__(self, other: "Clue") -> bool:
        """
        Equality check.
        """
        return (
            self.clue_word == other.clue_word
            and self.feedback == other.feedback
        )

    # -------------------------------------------------------------------------
    # Displays and string representations
    # -------------------------------------------------------------------------

    @property
    def colourful_str(self) -> str:
        """
        Colourful string representation.
        """
        return "".join(
            colourful_char(c, f)
            for c, f in self.char_feedback_pairs
        )

    @property
    def plain_str(self) -> str:
        """
        Plain string representation.
        """
        return f"{self.clue_word}/{self.feedback_str}"

    @classmethod
    def feedback_str_from_coded(cls, feedback: Sequence[CharFeedback]) -> str:
        """
        Pretty version of feedback.
        """
        return "".join(f.plain_str for f in feedback)

    @property
    def feedback_str(self) -> str:
        """
        Feedback in our plain string format.
        """
        return self.feedback_str_from_coded(self.feedback)

    def __str__(self) -> str:
        """
        String representation. The colourful one leaves a colour residue for
        logs.
        """
        return self.plain_str

    # -------------------------------------------------------------------------
    # Checking letters and guesses
    # -------------------------------------------------------------------------

    def letter_known_absent_or_redundant(self, letter: str) -> bool:
        """
        Is the specified letter known to be absent from the target word, or
        redundant in addition to the same letter in the right place, from
        this clue?
        """
        for c, f in self.char_feedback_pairs:
            if c == letter and f == CharFeedback.ABSENT_OR_REDUNDANT:
                return True
        return False

    def letter_known_present(self, letter: str) -> bool:
        """
        Is the specified letter known to be present in the target word, from
        this clue?
        """
        present_codes = (CharFeedback.PRESENT_WRONG_LOCATION,
                         CharFeedback.PRESENT_RIGHT_LOCATION)
        for c, f in self.char_feedback_pairs:
            if c == letter and f in present_codes:
                return True
        return False

    def letter_known_location(self, letter: str) -> bool:
        """
        Is the specified letter known to be present at a known location in the
        target word, from this clue?
        """
        for c, f in self.char_feedback_pairs:
            if c == letter and f == CharFeedback.PRESENT_RIGHT_LOCATION:
                return True
        return False

    def correct(self) -> bool:
        """
        Was the guess correct?
        """
        return all(
            c == CharFeedback.PRESENT_RIGHT_LOCATION for c in self.feedback
        )

    def is_location_known(self, pos: int) -> bool:
        """
        Do we know the letter at this location? (Zero-based index.)
        """
        return self.feedback[pos] == CharFeedback.PRESENT_RIGHT_LOCATION


# -----------------------------------------------------------------------------
# ClueGroup
# -----------------------------------------------------------------------------

class ClueGroup:
    """
    Represents multiple clues.
    """
    def __init__(self, clues: List[Clue]) -> None:
        self.clues = clues
        self.present = reduce(or_, (c.present for c in clues), set())
        self.absent = reduce(or_, (c.absent for c in clues), set())

    def __str__(self) -> str:
        """
        Summary of our calculated details.
        """
        p = "".join(sorted(self.present)) or "?"
        a = "".join(sorted(self.absent)) or "?"
        c = ", ".join(str(cl) for cl in self.clues) or "?"
        return f"Clues: {c}. Target must contain {p}; must not contain {a}."

    def compatible(self, guess: str) -> bool:
        """
        Is a guess compatible with a bunch of clues?
        """
        # Basic checks quickly:
        g_letters = set(guess)  # splits string into letters
        if len(g_letters.intersection(self.present)) != len(self.present):
            # Letters that we know should be present are missing.
            return False
        if g_letters.intersection(self.absent):
            # Letters that we know should be absent are in this candidate.
            return False
        # In more detail:
        return all(c.compatible(guess) for c in self.clues)

    @property
    def known_positions(self) -> List[int]:
        """
        Character positions whose letter we know.
        """
        known = []  # type: List[int]
        for pos in range(WORDLEN):
            for clue in self.clues:
                if clue.is_location_known(pos):
                    known.append(pos)
                    break
        return known

    @property
    def unknown_positions(self) -> List[int]:
        """
        Character positions whose letter we know.
        """
        unknown = []  # type: List[int]
        for pos in range(WORDLEN):
            for clue in self.clues:
                if clue.is_location_known(pos):
                    break
            unknown.append(pos)
        return unknown


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------

class State:
    """
    Represents summary information about where we stand.
    """
    def __init__(self,
                 all_words: np.array,
                 clues: List[Clue],
                 guesses_remaining: int,
                 show_threshold: int = DEFAULT_SHOW_THRESHOLD,
                 sig_fig: Optional[int] = DEFAULT_SIG_FIGURES,
                 previous_state: Optional["State"] = None) -> None:
        """
        Args:
            all_words: all words in the game
            clues: clues so far
            guesses_remaining: number of guesses remaining
        """
        # Supplied
        self.all_words = all_words
        self.cluegroup = ClueGroup(clues)
        self.guesses_remaining = guesses_remaining
        self.show_threshold = show_threshold
        self.sig_fig = sig_fig

        # Derived.
        self.this_guess = N_GUESSES - guesses_remaining + 1
        # noinspection PyUnresolvedReferences
        source = previous_state.possible_words if previous_state else all_words
        self.possible_words = set(
            w
            for w in source
            if self.cluegroup.compatible(w)
        )
        self._letter_counter = None  # type: Optional[Counter]
        self._letter_counters_by_pos = None  # type: Optional[List[Counter]]

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def _format_scoredict(self, d: Dict[str, float]) -> Dict[str, float]:
        return {
            k: convert_sf(v, self.sig_fig)
            for k, v in d.items()
        }

    def __str__(self) -> str:
        """
        String representation.
        """
        cg = self.cluegroup
        return "\n".join([
            (
                f"- This is guess {self.this_guess}. "
                f"Guesses remaining: {self.guesses_remaining}."
            ),
            f"- {cg}",
            (
                f"- Number of possible words: {self.n_possible}. " +
                (
                    f"Possibilities: {self.pretty_possibilities}"
                    if self.n_possible <= self.show_threshold
                    else f"Not yet showing possibilities "
                         f"(>{self.show_threshold})."
                )
            ),
            f"- Guesses so far: {self.pretty_clues}",
            # ... last -- messes up log colour
        ])

    @property
    def pretty_clues(self) -> str:
        """
        Pretty string version of the clues so far.
        """
        return prettylist(clue.colourful_str for clue in self.cluegroup.clues)

    @property
    def pretty_possibilities(self) -> str:
        """
        Pretty string version of the possibilities left.
        """
        return prettylist(sorted(self.possible_words))

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    @property
    def n_possible(self) -> int:
        """
        Number of possibilities left.
        """
        return len(self.possible_words)

    @property
    def first_possible(self) -> Optional[str]:
        """
        First possibility (in random order -- meaningful only if there is
        just one possibility).
        """
        if self.n_possible == 0:
            return None
        return next(iter(self.possible_words))

    def is_possible(self, guess: str) -> bool:
        """
        Is this guess one of the remaining possibilities?
        """
        return guess in self.possible_words

    # -------------------------------------------------------------------------
    # Letter frequency
    # -------------------------------------------------------------------------

    def letter_count(self, letter: str) -> int:
        """
        Returns the number of remaining possible words in which this letter
        appears.

        We don't get information about >1 position per clue, so I think we
        should do this as frequency of "letters within words", not "letters",
        e.g. that the word THREE contributes only one E.
        """
        if self._letter_counter is None:
            counter = Counter()
            for word in self.possible_words:
                # Count each letter only once per word, by using
                # set(possible_word) rather than possible_word as the iterable
                # to update the counter.
                counter.update(set(word))
            self._letter_counter = counter
        return self._letter_counter[letter]

    def letter_count_unknown(self, letter: str) -> int:
        """
        As for letter_count(), but restricted to letters whose presence we're
        unsure of.
        """
        if letter in self.cluegroup.present:
            return 0
        return self.letter_count(letter)

    def sum_letter_count_unknown_for_word(self, guess: str) -> int:
        """
        Returns the sum of our "unknown" letter frequencies for unique letters
        in the candidate word. Used by some algorithms.
        """
        return sum(
            self.letter_count_unknown(letter)
            for letter in set(guess)
        )

    def letter_count_positional(self, letter: str, pos: int) -> int:
        """
        Returns the number of remaining possible words in which this letter
        appears at this position.
        """
        if self._letter_counters_by_pos is None:
            counters = [Counter() for _ in range(WORDLEN)]
            for word in self.possible_words:
                for p in range(WORDLEN):
                    counters[p].update(word[p])
            self._letter_counters_by_pos = counters
        return self._letter_counters_by_pos[pos][letter]

    # def letter_count_positional_unknown(self, letter: str, pos: int) -> int:
    #     """
    #     As for letter_count_positional(), but for unknown letters.
    #     """
    #     if letter in self.cluegroup.present:
    #         return 0
    #     return self.letter_count_positional(letter, pos)

    def sum_letter_count_positional_unknown_for_word(self, guess: str) -> int:
        """
        Returns the sum of our "unknown"-position letter frequencies for unique
        letters in the candidate word. Used by some algorithms.
        """
        counts = [0] * WORDLEN
        for pos in self.cluegroup.unknown_positions:
            counts[pos] = self.letter_count_positional(guess[pos], pos)
        # But don't score a single letter twice:
        countdict = {}  # type: Dict[str, int]
        for pos, letter in enumerate(guess):
            c = counts[pos]
            if letter in countdict:
                countdict[letter] = max(countdict[letter], c)
            else:
                countdict[letter] = c
        return sum(countdict.values())

    # -------------------------------------------------------------------------
    # A bit more sophisticated in terms of information provided
    # -------------------------------------------------------------------------

    def probability_of_feedback(self, guess: str, debug: bool = False) \
            -> Dict[Tuple[CharFeedback, ...], float]:
        """
        Returns a dictionary whose keys are the feedback options possible for
        this guess (given the possibilities that remain), and whose values are
        the corresponding feedback probabilities (assuming that all possible
        words are equiprobable).
        """
        counter = Counter(
            Clue.get_from_known_word(guess, target).feedback
            for target in self.possible_words
        )
        d = defaultdict(float)  # default value 0.0
        # Python 3.10 has Counter.total(), but before then:
        total = sum(counter.values())
        for feedback, count in counter.items():
            d[feedback] = count / total
        if debug:
            pretty_d = {
                Clue.feedback_str_from_coded(feedback): convert_sf(p)
                for feedback, p in d.items()
            }
            rootlog.debug(f"probability_of_feedback({guess!r}): {pretty_d}")
        return d

    def n_eliminated_by_clue(self, clue: Clue, debug: bool = False) -> int:
        """
        Returns the number of remaining possibilities that would be eliminated
        by a specific clue (= guess + feedback).
        """
        n_before = len(self.possible_words)
        n_after = sum(
            1 for w in self.possible_words
            if clue.compatible(w)
        )
        n_eliminated = n_before - n_after
        if debug:
            rootlog.debug(f"Clue {clue} would eliminate "
                          f"{n_eliminated} of {n_before}, leaving {n_after}")
        return n_eliminated

    def weighted_n_eliminated(self, guess: str) -> float:
        """
        Returns the number of current possibilities likely to be eliminated by
        a guess, averaged over the possible types of feedback the guess might
        elicit (weighted for the probability of that type of feedback). Assumes
        all possible words are equiprobable.

        For example, PEEVE is a poor guess (low weighted number of words
        eliminated) compared to PETAL; ABETS is even better. WHIZZ is an
        example of a particularly bad first guess.

        This is currently slow (about 1.1 s per guess with the initial 5905
        words).
        """
        n_possible = len(self.possible_words)
        pf = self.probability_of_feedback(guess)
        t = 0.0
        for feedback, p_feedback in pf.items():
            clue = Clue(guess, feedback)
            t += p_feedback * self.n_eliminated_by_clue(clue)
        rootlog.debug(f"Guess {guess}: "
                      f"weighted_n_eliminated = {t:.2f} "
                      f"(out of {n_possible}, "
                      f"or {convert_sf(100 * t / n_possible)}%)")
        return t


# -----------------------------------------------------------------------------
# Scoring potential guesses
# -----------------------------------------------------------------------------

@total_ordering
class WordScore:
    """
    Class to represent the score for a potential word guess.
    """
    # Override this to provide the first suggestion quickly:
    INITIAL_GUESS = None

    def __init__(self, word: str, state: State,
                 sig_fig: Optional[int] = DEFAULT_SIG_FIGURES) -> None:
        self.word = word
        self.state = state
        self.sig_fig = sig_fig
        self._score = None  # type: Optional[WORDSCORE_TYPE]

    def __str__(self) -> str:
        if self.sig_fig is not None:
            score_sf = convert_sf(self.score, self.sig_fig)
        else:
            score_sf = self.score
        return f"{self.word} ({score_sf})"

    def __eq__(self, other: "WordScore") -> bool:
        return self.score == other.score

    def __lt__(self, other: "WordScore") -> bool:
        return self.score < other.score

    @property
    def score(self) -> WORDSCORE_TYPE:
        """
        Caches the calculated score. Caching is important as these calculations
        can be very slow, and sorting by score involves re-retrieving scores
        several times.
        """
        if self._score is None:
            self._score = self.get_score()
        return self._score

    def get_score(self) -> WORDSCORE_TYPE:
        """
        Overridden to implement a specific scoring method.

        The key "thinking" algorithm. Returns a score for this word: how good
        would it be to use this as the next guess?

        Can return a float, or a tuple of floats, etc.

        Return None for suggestions so dreadful they should not be considered.
        """
        raise NotImplementedError


class UnknownLetterExplorerMindingGuessCount(WordScore):
    """
    Scores guesses for the letter frequency (among possible words on a
    letter-once-per-word basis), for letters already not known to be present.
    Thus, explores unknown letters, preferring the most common.
    Tie-breaker (second part of tuple): whether a word is a candidate or not.

    Performance across our 5905 words: 98.7% success.
    """
    INITIAL_GUESS = "AROSE"

    def get_score(self) -> Optional[Tuple[int, int]]:
        state = self.state
        possible = int(state.is_possible(self.word))
        if state.guesses_remaining <= 1 and not possible:
            # If we have only one guess left, we must make a stab at the word
            # itself. So eliminate words that are impossible.
            return None
        s = state.sum_letter_count_unknown_for_word(self.word)
        return s, possible


class UnknownLetterExplorerAmongstPossible(WordScore):
    """
    As for UnknownLetterExplorerMindingGuessCount, but only explores possible
    words.

    Performance across our 5905 words: 94.6% success.
    """
    INITIAL_GUESS = "AROSE"

    def get_score(self) -> Optional[float]:
        state = self.state
        if not state.is_possible(self.word):
            return None
        return state.sum_letter_count_unknown_for_word(self.word)


class PositionalExplorerAmongstPossible(WordScore):
    """
    Likes unknown letters that are common in positions we don't know, amongst
    possible words.

    Performance across our 5905 words: 94.8% success.
    """
    INITIAL_GUESS = "CARES"

    def get_score(self) -> Optional[int]:
        state = self.state
        if not state.is_possible(self.word):
            return None
        return state.sum_letter_count_positional_unknown_for_word(self.word)


class PositionalExplorerMindingGuessCount(WordScore):
    """
    Likes unknown letters that are common in positions we don't know, across
    all words (except insisting on a possible word for our last guess).

    Performance across our 5905 words: 94.8% success.

    ? problem -- exactly the same as PositionalExplorerAmongstPossible
    """
    INITIAL_GUESS = "CARES"

    def get_score(self) -> Optional[Tuple[int, int]]:
        state = self.state
        possible = int(state.is_possible(self.word))
        if state.guesses_remaining <= 1 and not possible:
            return None
        return (
            state.sum_letter_count_positional_unknown_for_word(self.word),
            possible
        )


class Eliminator(WordScore):
    """
    Likes guesses that eliminate the most possibilities, in a
    probability-weighted way.

    Slow, but good.
    Performance across our 5905 words: 99.966% success, with a mean of 3.82.
    (The only two it didn't achieve within 6 guesses were HAZES and WAXES,
    which it got on 7.)
    """
    INITIAL_GUESS = "RATES"
    # - AIRES and ARIES both score top at 5780 (out of 5905), but neither are
    #   in the Wordle list.
    # - TARES, SANER, RATES, LANES, TALES, and REALS all come next at 5770.

    def get_score(self) -> Optional[Tuple[float, int]]:
        state = self.state
        possible = int(state.is_possible(self.word))
        if state.guesses_remaining <= 1 and not possible:
            return None
        return (
            state.weighted_n_eliminated(self.word),
            possible
        )


class EliminatorAmongstPossible(WordScore):
    """
    As for Eliminator, but from within current possibilities.

    Much quicker (especially given that we predefine our first guess, thus
    restricting the number of possibilities from the outset).
    """
    INITIAL_GUESS = "RATES"  # as for Eliminator

    def get_score(self) -> Optional[float]:
        state = self.state
        if not state.is_possible(self.word):
            return None
        return state.weighted_n_eliminated(self.word)


ALGORITHMS = {
    "UnknownLetterExplorerMindingGuessCount":
        UnknownLetterExplorerMindingGuessCount,
    "UnknownLetterExplorerAmongstPossible":
        UnknownLetterExplorerAmongstPossible,
    "PositionalExplorerAmongstPossible": PositionalExplorerAmongstPossible,
    "PositionalExplorerMindingGuessCount": PositionalExplorerMindingGuessCount,
    "Eliminator": Eliminator,
    "EliminatorAmongstPossible": EliminatorAmongstPossible,
}  # type: Dict[str, Type[WordScore]]

DEFAULT_ALGORITHM = "Eliminator"  # it's the best

DEFAULT_ALGORITHM_CLASS = ALGORITHMS[DEFAULT_ALGORITHM]


# -----------------------------------------------------------------------------
# Using a particular scoring method, calculate a best guess
# -----------------------------------------------------------------------------

@ray.remote
def suggest_ray(algorithm_class: Type[WordScore],
                test_words: Iterable[str],
                state: State) -> List[WordScore]:
    """
    Helper function for a parallel version. This worker task scores a bunch of
    words (a subset of the full set).
    """
    return [algorithm_class(word=w, state=state) for w in test_words]


def flatten(x: Iterable[Any]) -> Iterable[Any]:
    """
    Flatten, for example, a list of lists to an iterable of the items.
    """
    for y in x:
        if isinstance(y, list):
            for item in y:
                yield item
        else:
            yield y


def filter_consider_suggestion(suggestion: WordScore) -> bool:
    """
    Filter to reject awful suggestions.
    """
    return suggestion.score is not None


def suggest(state: State,
            algorithm_name: str = DEFAULT_ALGORITHM,
            top_n: int = 5,
            silent: bool = False,
            log: logging.Logger = None,
            nproc: int = 1) -> Tuple[str, bool]:
    """
    Show advice to the user: what word should be guessed next?

    Returns the best guess (or, the first of the equally good guesses) for
    automatic checking.

    Returns: guess, certain

    Parallelization doesn't really help, but the framework code is done.
    """
    log = log or rootlog
    n_possible = state.n_possible
    if n_possible == 1:
        answer = state.first_possible
        if not silent:
            log.info(f"Word is: {answer}")
        return answer, True
    elif n_possible == 0:
        raise ValueError("Word is not in our dictionary!")

    if not silent:
        log.info(f"State:\n{state}")

    algorithm_class = ALGORITHMS[algorithm_name]
    if state.this_guess == 1 and algorithm_class.INITIAL_GUESS:
        # Speedup
        suggestion = algorithm_class.INITIAL_GUESS
        if not silent:
            log.info(f"Suggestion algorithm: {algorithm_name}\n"
                     f"- Initial suggestion: {suggestion}")
    else:
        # Any word may be a candidate for a guess, not just the possibilities
        # -- for example, if we know 4/5 letters in the correct positions early
        # on, we might be better off with a guess that has lots of options for
        # that final letter, rather than guessing them sequentially in a single
        # position. Therefore, we generate a score for every word in all_words.
        if nproc > 1:
            chunks_per_task = 1
            words_per_chunk = len(state.all_words) // (nproc * chunks_per_task)
            wordgen = flatten(ray.get([
                suggest_ray.remote(algorithm_class, test_words, state)
                for test_words in chunks(state.all_words, words_per_chunk)
            ]))
        else:
            wordgen = (
                algorithm_class(word=w, state=state) for w in state.all_words
            )
        options = sorted(
            filter(filter_consider_suggestion, wordgen),
            reverse=True  # from high to low scores
        )

        # The thinking is done. Now we just need to present them nicely.
        top_n_options_str = prettylist(options[:top_n])
        best_score = options[0].score
        # Find the equal best suggestion(s)
        top_words = []  # type: List[str]
        for o in options:
            if o.score < best_score:
                break
            top_words.append(o.word)
        if not silent:
            log.info(f"Suggestion algorithm: {algorithm_name}\n"
                     f"- Top {top_n} suggestions: {top_n_options_str}\n"
                     f"- Best suggestion(s): {prettylist(top_words)}")
        suggestion = top_words[0]

    return suggestion, False


# -----------------------------------------------------------------------------
# Interactive solver
# -----------------------------------------------------------------------------

def solve_interactive(
        wordlist_filename: str,
        debug_nwords: int = None,
        show_threshold: int = DEFAULT_SHOW_THRESHOLD,
        advice_top_n: int = DEFAULT_ADVICE_TOP_N,
        algorithm_name: str = DEFAULT_ALGORITHM) -> None:
    """
    Solve in a basic way using user guesses.
    """
    rootlog.info("Wordle Assistant. By Rudolf Cardinal <rudolf@pobox.com>.")
    all_words = read_words(wordlist_filename, max_n=debug_nwords)
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while guesses_remaining > 0:
        # Show the current situation
        state = State(all_words, clues, guesses_remaining,
                      show_threshold=show_threshold)

        # Provide advice
        _, _ = suggest(state, algorithm_name, top_n=advice_top_n)

        # Read the results of a guess
        clue = Clue.read_from_user()
        clues.append(clue)
        if clue.correct():
            this_guess = N_GUESSES - guesses_remaining + 1
            rootlog.info(f"Success in {this_guess} guesses.")
            return

        guesses_remaining -= 1

    rootlog.info("Out of guesses!")
    guess_words = [c.clue_word for c in clues]
    if len(guess_words) != len(set(guess_words)):
        rootlog.info("You are an idiot.")


# -----------------------------------------------------------------------------
# Autosolver and performance testing framework to compare algorithms
# -----------------------------------------------------------------------------

def autosolve(target: str,
              all_words: np.array,
              algorithm_name: str,
              allow_beyond_guess_limit: int = 100,
              log: logging.Logger = None) -> List[Clue]:
    """
    Automatically solves, and returns the clues from each guess (including the
    final successful one). (This can go over the Wordle guess limit; avoid
    sharp edges for comparing algorithms.)
    """
    log = log or rootlog
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    state = None
    while True:
        log.debug(f"... for word {target}, "
                  f"guesses_remaining={guesses_remaining}...")
        state = State(all_words, clues, guesses_remaining,
                      previous_state=state)
        guess, certain = suggest(
            state,
            algorithm_name=algorithm_name,
            top_n=DEFAULT_ADVICE_TOP_N,
            silent=True,
            nproc=1  # parallelize over words instead
        )
        clue = Clue.get_from_known_word(guess=guess, target=target)
        clues.append(clue)
        if clue.correct():
            log.info(f"Word is: {guess}. Guesses: {state.pretty_clues}")
            return clues
        guesses_remaining -= 1
        if guesses_remaining < -allow_beyond_guess_limit:
            log.warning(f"Abandoning word {target}")
            return clues


def autosolve_single_arg(args: Tuple[str, np.array, str]) -> int:
    """
    Version of :func:`autosolve` that takes a single argument, which is
    necessary for some of the parallel processing map functions.

    The argument is a tuple: target, wordlist_filename, score_class.

    Returns a tuple: word, n_guesses_taken.
    """
    target, all_words, algorithm_name = args
    clues = autosolve(target, all_words, algorithm_name)
    n_guesses = len(clues)
    return n_guesses


@ray.remote
def autosolve_ray(targets: List[str],
                  all_words: np.array,
                  algorithm_name: str,
                  dummy_run: bool = False,
                  loglevel: int = logging.INFO) -> List[Tuple[str, int]]:
    """
    Ray version! Batched.
    """
    # Beware using logs in subprocesses?
    # https://stackoverflow.com/questions/55272066/how-can-i-use-the-python-logging-in-ray  # noqa
    # This works, but every time the process is launched with a new chunk,
    # we get an additional logger!
    raylog = logging.getLogger(__name__)
    configure_logger_for_colour(raylog, level=loglevel)
    results = []  # type: List[Tuple[str, int]]
    for target in targets:
        with time_section("Word"):
            if dummy_run:
                raylog.warning(f"dummy: {target}")
                n_guesses = -1
            else:
                clues = autosolve(target, all_words, algorithm_name,
                                  log=raylog)
                n_guesses = len(clues)
        results.append((target, n_guesses))
    return results


def measure_algorithm_performance(
        wordlist_filename: str,
        output_filename: str,
        nwords: int = None,
        nproc: int = DEFAULT_NPROC,
        algorithm_name: str = DEFAULT_ALGORITHM,
        chunks_per_worker: int = 5,
        loglevel: int = logging.INFO,
        use_ray: bool = True) -> None:
    """
    Test a guess algorithm and report its performance statistics.
    """
    all_words = read_words(wordlist_filename)
    test_words = list(read_words(wordlist_filename, max_n=nwords))
    n_words = len(test_words)
    guess_counts = []  # type: List[int]
    with open(output_filename, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "word", "n_guesses"])

        if use_ray:
            # -----------------------------------------------------------------
            # Ray method
            # -----------------------------------------------------------------
            rootlog.info("Starting Ray")
            ray.init(num_cpus=nproc)
            words_per_chunk = n_words // (nproc * chunks_per_worker)
            pending_jobs = [
                autosolve_ray.remote(targets, all_words, algorithm_name,
                                     loglevel=loglevel)
                for targets in chunks(test_words, words_per_chunk)
            ]
            rootlog.info(f"Submitted {len(pending_jobs)} jobs, aiming for "
                         f"{words_per_chunk} words per job")
            while len(pending_jobs):
                rootlog.debug(f"Waiting for a job to complete "
                              f"({len(pending_jobs)} running)...")
                done_jobs, pending_jobs = ray.wait(pending_jobs)
                for done_job in done_jobs:
                    results = ray.get(done_job)
                    rootlog.debug(f"Retrieved {len(results)} results")
                    for word, n_guesses in results:
                        writer.writerow([algorithm_name, word, n_guesses])
                        f.flush()  # nice to be able to follow the output live
                        guess_counts.append(n_guesses)

            # Well, that's more like it. Sensible ability to pass lots of
            # arguments in and get tuples or other Python structures back,
            # back, it yields results helpfully, and it's elegant. Also, it
            # has commands like "ray status" (potentially "ray memory", too,
            # but I haven't got that to work -- dependencies and then a "no
            # module named 'aioredis.pubsub'" crash).
            #
            # Note:
            # - We use np.array(words, object="U5") for 5-character Unicode
            #   strings. Ray is optimized to pass Numpy arrays as read-only
            #   objects without copying them. See
            #   https://docs.ray.io/en/master/ray-core/serialization.html.

        else:
            # -----------------------------------------------------------------
            # ProcessPoolExecutor method
            # -----------------------------------------------------------------
            # ThreadPoolExecutor is slow -- likely limited by Python GIL.
            #
            # General use of ProcessPoolExecutor:
            # - https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor-example  # noqa
            # - https://superfastpython.com/processpoolexecutor-in-python/
            # - https://stackoverflow.com/questions/42074501/python-concurrent-futures-processpoolexecutor-performance-of-submit-vs-map  # noqa
            #
            # Intermittent deadlocks (Ctrl-C shows stuck at
            # "waiter.acquire()"):
            # - https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/  # noqa
            # - https://pythonspeed.com/articles/python-multiprocessing/
            # Argh, frustrating. Neither fork nor spawn make it simple/clear.
            # Fork is the default under Linux.
            # With spawn, logs are not re-enabled.
            #
            # ... was actually due to the underlying code being buggy at the
            # time and taking infinite guesses. Cap added and bug fixed.

            # Workaround to pass a single argument:
            arglist = (
                (target, all_words, algorithm_name)
                for target in test_words
            )
            n_chunks = nproc * chunks_per_worker
            chunksize = max(1, n_words // n_chunks)
            rootlog.debug(
                f"Aiming for {chunks_per_worker} chunks/worker with {nproc} "
                f"workers and thus {n_chunks} chunks: for {n_words} words, "
                f"chunksize = {chunksize} words/chunk"
            )
            with ProcessPoolExecutor(nproc) as executor:
                for word, n_guesses in zip(
                        test_words,
                        executor.map(autosolve_single_arg, arglist,
                                     chunksize=chunksize)):
                    writer.writerow([algorithm_name, word, n_guesses])
                    f.flush()  # nice to be able to follow the output live
                    guess_counts.append(n_guesses)

    n_tests = len(guess_counts)
    assert n_tests > 0, "No words!"
    tested = (
        f"all {n_tests} known" if nwords is None
        else f"the first {n_tests}"
    )
    rootlog.info(
        f"Across {tested} words, method {algorithm_name} took: "
        f"min {min(guess_counts)}, "
        f"median {median(guess_counts)}, "
        f"mean {mean(guess_counts)}, "
        f"max {max(guess_counts)} guesses"
    )


# =============================================================================
# Self-testing
# =============================================================================

class TestClues(unittest.TestCase):
    @staticmethod
    def _testclue(target: str, guess: str, wordle_feedback: str) -> None:
        # Setup
        wordle_clue = Clue.get_from_strings(guess, wordle_feedback)

        # Test 1: do we generate the right clue?
        our_clue = Clue.get_from_known_word(guess, target)
        assert our_clue == wordle_clue, (
            f"For target {target} and guess {guess}, our code produces the "
            f"clue {our_clue.colourful_str} ({our_clue.feedback_str}), but "
            f"the correct clue from Wordle is {wordle_clue.colourful_str} "
            f"({wordle_clue.feedback_str})."
        )

        # Test 2: does our code agree that the correct answer is compatible
        # with the clue?
        cluegroup = ClueGroup([wordle_clue])
        rootlog.critical(f"cluegroup: {cluegroup}")
        assert cluegroup.compatible(target), (
            f"Bug in ClueGroup.compatible(): it thinks {target} is "
            f"incompatible with the clue {wordle_clue.colourful_str}, but "
            f"it must be."
        )

    def test_duplicate_letters(self) -> None:
        self._testclue(
            target="HUMOR",  # Wordle 2022-02-09
            guess="HONOR",
            wordle_feedback="=__=="
            # note in particular that the first O is given a "no" code, not a
            # "somewhere else" code.
        )
        self._testclue(
            # Guess with duplicate letters
            target="HUMOR",  # Wordle 2022-02-09
            guess="HONOR",
            wordle_feedback="=__=="
            # note in particular that the first O is given a "no" code, not a
            # "somewhere else" code.
        )
        self._testclue(
            # What about three?
            # grep "E.*E.*E" five_letter_words.txt
            target="PAUSE",  # Wordle 2022-02-10
            guess="EERIE",
            wordle_feedback="____="
        )
        self._testclue(
            # Two, both in the wrong place
            target="PAUSE",
            guess="LEPER",
            wordle_feedback="_--__"
            # only the first E gets the "wrong place" marker
        )

    def test_possible(self) -> None:
        s1 = State(
            all_words=make_np_array_words(["COINS", "SCION", "PAPER"]),
            clues=[Clue.get_from_strings("COINS", "--=--")],
            guesses_remaining=N_GUESSES,
        )
        assert s1.possible_words == {"SCION"}

    def test_specific(self) -> None:
        # These days, if something goes wrong, it should be because the user
        # mistyped the clue feedback! That was the case here.
        s1 = State(
            all_words=make_np_array_words(["TACIT"]),
            clues=[
                Clue.get_from_strings("RATES", "_=-__"),
                Clue.get_from_strings("TYING", "=_-__"),
            ],
            guesses_remaining=N_GUESSES,
        )
        assert "TACIT" in s1.possible_words


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
    # parser.add_argument(
    #     "--freqlist_filename", default=DEFAULT_FREQLIST,
    #     help=f"File containing all {WORDLEN}-letter words in upper case and "
    #          f"associated frequencies"
    # )
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

    # cmd_mkfreq = "make_frequencylist"
    # parser_mkfreq = subparsers.add_parser(
    #     cmd_mkfreq,
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser_mkfreq.add_argument(
    #     "--freq_url",
    #     default="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_full.txt",  # noqa
    #     help="URL for word/frequency file"
    # )

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
    parser_solve.add_argument(
        "--algorithm", type=str, choices=ALGORITHMS.keys(),
        default=DEFAULT_ALGORITHM,
        help="Algorithm to use"
    )
    parser_solve.add_argument(
        "--debug_nwords", type=int,
        help="Number of words to load (debugging only)"
    )

    cmd_test_performance = "test_performance"
    parser_test_performance = subparsers.add_parser(
        cmd_test_performance,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_test_performance.add_argument(
        "--output", type=str, default=None,
        help="File for CSV-format output (if unspecified, a sensible default "
             "will be created based on the algorithm chosen)"
    )
    parser_test_performance.add_argument(
        "--nwords", type=int,
        help="Number of words to test (if unspecified, will test all)"
    )
    parser_test_performance.add_argument(
        "--nproc", type=int, default=DEFAULT_NPROC,
        help="Number of parallel processes"
    )
    parser_test_performance.add_argument(
        "--algorithm", type=str, choices=ALGORITHMS.keys(),
        default=DEFAULT_ALGORITHM,
        help="Algorithm to use"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    main_only_quicksetup_rootlogger(level=loglevel)

    # -------------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------------
    if args.command == cmd_make:
        make_wordlist(args.source_dict, args.wordlist_filename)
    # elif args.command == cmd_mkfreq:
    #     make_frequency_list(args.freq_url, args.freqlist_filename)
    elif args.command in (cmd_solve, cmd_test_performance):
        if args.command == cmd_solve:
            solve_interactive(
                wordlist_filename=args.wordlist_filename,
                debug_nwords=args.debug_nwords,
                show_threshold=args.show_threshold,
                advice_top_n=args.advice_top_n,
                algorithm_name=args.algorithm,
            )
        elif args.command == cmd_test_performance:
            output_filename = (
                args.output or f"out_{args.algorithm}.csv"
            )
            measure_algorithm_performance(
                wordlist_filename=args.wordlist_filename,
                output_filename=output_filename,
                nwords=args.nwords,
                nproc=args.nproc,
                algorithm_name=args.algorithm,
                loglevel=loglevel,
            )
    else:
        raise AssertionError("argument-parsing bug")


if __name__ == '__main__':
    main()
