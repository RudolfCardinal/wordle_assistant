#!/usr/bin/env python
"""
Wordle solver.

By Rudolf Cardinal <rudolf@pobox.com>, from 2022-02-08.

Run self-tests with:

.. code:: bash

    pip install pytest
    pytest wordle.py

"""

# =============================================================================
# Imports
# =============================================================================

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import reduce, total_ordering
import logging
from multiprocessing import cpu_count
from operator import or_
import os
import re
from statistics import median, mean
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import unittest
from urllib.request import urlopen

from colors import color  # pip install ansicolors
from cardinal_pythonlib.logs import main_only_quicksetup_rootlogger

log = logging.getLogger(__name__)


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
ALL_LETTERS = tuple(
    chr(_letter_ascii_code)
    for _letter_ascii_code in range(ord('A'), ord('Z') + 1)
)

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

# Colours and styles for displaying guesses
COLOUR_FG = "white"
COLOUR_BG_ABSENT_REDUNDANT = "black"
COLOUR_BG_PRESENT_WRONG_LOCATION = "orange"
COLOUR_BG_PRESENT_RIGHT_LOCATION = "green"
STYLE = "bold"

# Defaults
DEFAULT_SHOW_THRESHOLD = 100
DEFAULT_ADVICE_TOP_N = 10
DEFAULT_NPROC = cpu_count()


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

def colourful_char(c: str, feedback: CharFeedback) -> str:
    """
    Returns a string with ANSI codes to colour the character according to the
    feedback (and then reset afterwards).
    """
    if feedback == CharFeedback.ABSENT_OR_REDUNDANT:
        bg = COLOUR_BG_ABSENT_REDUNDANT
    elif feedback == CharFeedback.PRESENT_WRONG_LOCATION:
        bg = COLOUR_BG_PRESENT_WRONG_LOCATION
    elif feedback == CharFeedback.PRESENT_RIGHT_LOCATION:
        bg = COLOUR_BG_PRESENT_RIGHT_LOCATION
    else:
        raise AssertionError("bug")
    return color(c, fg=COLOUR_FG, bg=bg, style=STYLE)


def prettylist(words: Iterable[Any]) -> str:
    """
    Formats a wordlist.
    """
    return ", ".join(str(x) for x in words)


# -----------------------------------------------------------------------------
# Reading word lists
# -----------------------------------------------------------------------------

def make_wordlist(from_filename: str,
                  to_filename: str) -> None:
    """
    Reads a dictionary file and creates a list of 5-letter words.
    """
    log.info(f"Reading from {from_filename}")
    log.info(f"Writing to {to_filename}")
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
    log.info(f"Wrote {n_written} ({WORDLEN}-letter) words to {to_filename}")


def read_words_frequencies(wordlist_filename: str,
                           frequencylist_filename: str = None,
                           default_frequency: int = 1,
                           max_n: int = None) -> Dict[str, int]:
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
                log.warning(f"Reading only {n_read} words")
                break
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


# -----------------------------------------------------------------------------
# Frequency analysis
# -----------------------------------------------------------------------------

def get_letter_frequencies(words: Set[str]) -> Dict[str, float]:
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
        letter: 0.0
        for letter in ALL_LETTERS
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
    def __init__(self, word: str, feedback: List[CharFeedback]):
        """
        Args:

            word: the word being guessed
            feedback: character-by-character feedback
        """
        assert len(word) == WORDLEN
        assert len(feedback) == WORDLEN
        self.clue_word = word.upper()
        self.feedback = feedback
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
        word = ""
        while not WORD_REGEX.match(word):
            word = input(f"{prefix1}> Enter the five-letter word: ")
            word = word.strip().upper()
        feedback_str = ""
        while not FEEDBACK_REGEX.match(feedback_str):
            feedback_str = input(
                f"Enter the feedback ({CHAR_ABSENT_OR_REDUNDANT!r} absent, "
                f"{CHAR_PRESENT_WRONG_LOC!r} present but wrong location, "
                f"{CHAR_CORRECT!r} correct location): "
            )
            feedback_str = feedback_str.strip().upper()
        clue = cls.get_from_typed_pair(word, feedback_str)
        print(f"{prefix2} You have entered this clue: {clue}")
        return clue

    @classmethod
    def get_from_typed_pair(cls, guess: str, feedback_str: str) -> "Clue":
        """
        Use our internal format to create a clue object.
        """
        assert WORD_REGEX.match(guess)
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
                # guess, IN ADDITION to any identical letter that have already
                # been matched for position.
                cluechar = self.clue_word[cluepos]
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

    @property
    def feedback_str(self) -> str:
        """
        Feedback in our plain string format.
        """
        return "".join(f.plain_str for f in self.feedback)

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
        Is a guess compatable with a bunch of clues?
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


# -----------------------------------------------------------------------------
# StateInfo
# -----------------------------------------------------------------------------

class StateInfo:
    """
    Represents summary information about where we stand.
    """
    def __init__(self,
                 all_words: Dict[str, int],
                 clues: List[Clue],
                 guesses_remaining: int,
                 show_threshold: int = DEFAULT_SHOW_THRESHOLD) -> None:
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

        # Derived.
        self.this_guess = N_GUESSES - guesses_remaining + 1
        self.possible_words = set(
            w
            for w in all_words.keys()
            if self.cluegroup.compatible(w)
        )
        self.letter_freq = get_letter_frequencies(self.possible_words)
        self.letter_freq_unknown = self.letter_freq.copy()
        for x in self.cluegroup.present:
            self.letter_freq_unknown[x] = 0.0

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """
        String representation.
        """
        cg = self.cluegroup
        return "\n".join([
            (
                f"- This is guess {self.this_guess}. "
                f"Guesses remaining: {self.guesses_remaining}. "
                f"Number of possible words: {self.n_possible}."
            ),
            f"- {cg}",
            (
                f"- Possibilities: {self.pretty_possibilities}"
                if self.n_possible <= self.show_threshold
                else f"- Not yet showing possibilities (>{self.show_threshold})"
            ),
            (
                f"- Letter frequencies in remaining possible words: "
                f"{self.letter_freq}"
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

    def sum_letter_freq_unknown_for_word(self, guess: str) -> float:
        """
        Returns the sum of our "unknown" letter frequencies for unique letters
        in the candidate word. Used by some algorithms.
        """
        return sum(
            self.letter_freq_unknown[letter]
            for letter in set(guess)
        )


# -----------------------------------------------------------------------------
# Scoring potential guesses
# -----------------------------------------------------------------------------

@total_ordering
class WordScore:
    def __init__(self, word: str, state: StateInfo) -> None:
        self.word = word
        self.state = state

    def __str__(self) -> str:
        return f"{self.word} ({self.score})"

    def __eq__(self, other: "WordScore") -> bool:
        return self.score == other.score

    def __lt__(self, other: "WordScore") -> bool:
        return self.score < other.score

    @property
    def score(self) -> Union[None, float, Iterable[float]]:
        """
        The key "thinking" algorithm. Returns a score for this word: how good
        would it be to use this as the next guess?

        Can return a float, or a tuple of floats, etc.

        Return None for suggestions so dreadful they should not be considered.
        """
        raise NotImplementedError


class UnknownLetterExplorer(WordScore):
    """
    Scores guesses for the letter frequency (among possible words on a
    letter-once-per-word basis), for letters already not known to be present.
    Thus, explores unknown letters, preferring the most common.
    Tie-breaker (second part of tuple): whether a word is a candidate or not.

    Performance across our 5905 words:

    - XXX
    """
    @property
    def score(self) -> Tuple[float, int]:
        state = self.state
        possible = int(state.is_possible(self.word))
        s = state.sum_letter_freq_unknown_for_word(self.word)
        return s, possible


class UnknownLetterExplorerMindingGuessCount(WordScore):
    """
    Scores guesses for the letter frequency (among possible words on a
    letter-once-per-word basis), for letters already not known to be present.
    Thus, explores unknown letters, preferring the most common.
    Tie-breaker (second part of tuple): whether a word is a candidate or not.

    Performance across our 5905 words:

    - XXX
    """
    @property
    def score(self) -> Optional[Tuple[float, int]]:
        state = self.state
        possible = int(state.is_possible(self.word))
        if state.guesses_remaining <= 1 and not possible:
            # If we have only one guess left, we must make a stab at the word
            # itself. So eliminate words that are impossible.
            return None
        s = state.sum_letter_freq_unknown_for_word(self.word)
        return s, possible


class UnknownLetterExplorerAmongstPossible(WordScore):
    """
    As for UnknownLetterExplorer, but only explores possible words.

    Can be especially dumb.

    Performance across our 5905 words:

    - XXX
    """
    @property
    def score(self) -> Optional[float]:
        state = self.state
        if not state.is_possible(self.word):
            return None
        return state.sum_letter_freq_unknown_for_word(self.word)


ALGORITHMS = {
    "UnknownLetterExplorer": UnknownLetterExplorer,
    "UnknownLetterExplorerMindingGuessCount":
        UnknownLetterExplorerMindingGuessCount,
    "UnknownLetterExplorerAmongstPossible":
        UnknownLetterExplorerAmongstPossible,
}  # type: Dict[str, Type[WordScore]]
DEFAULT_ALGORITHM = "UnknownLetterExplorerAmongstPossible"
DEFAULT_ALGORITHM_CLASS = ALGORITHMS[DEFAULT_ALGORITHM]


# -----------------------------------------------------------------------------
# Using a particular scoring method, calculate a best guess
# -----------------------------------------------------------------------------

def filter_consider_suggestion(suggestion: WordScore) -> bool:
    """
    Filter to reject awful suggestions.
    """
    return suggestion.score is not None


def suggest(state: StateInfo,
            score_class: Type[WordScore] = DEFAULT_ALGORITHM,
            top_n: int = 5,
            silent: bool = False) -> Tuple[str, bool]:
    """
    Show advice to the user: what word should be guessed next?

    Returns the best guess (or, the first of the equally good guesses) for
    automatic checking.

    Returns: guess, certain
    """
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

    # Any word may be a candidate for a guess, not just the possibilities --
    # for example, if we know 4/5 letters in the correct positions early on, we
    # might be better off with a guess that has lots of options for that final
    # letter, rather than guessing them sequentially in a single position.
    # Therefore, we generate a score for every word in all_words.
    options = list(filter(
        filter_consider_suggestion,
        (score_class(word=w, state=state) for w in state.all_words)
    ))
    options.sort(reverse=True)  # from high to low scores

    # The thinking is done. Now we just need to present them nicely.
    top_n_options_str = prettylist(options[:top_n])
    if not silent:
        log.info(f"Top {top_n} suggestions: {top_n_options_str}")
    best_score = options[0].score
    # Find the equal best suggestion(s)
    top_words = []  # type: List[str]
    for o in options:
        if o.score < best_score:
            break
        top_words.append(o.word)
    if not silent:
        log.info(f"Best suggestion(s): {prettylist(top_words)}")
    return top_words[0], False


# -----------------------------------------------------------------------------
# Interactive solver
# -----------------------------------------------------------------------------

def solve_interactive(
        wordlist_filename: str,
        show_threshold: int = DEFAULT_SHOW_THRESHOLD,
        advice_top_n: int = DEFAULT_ADVICE_TOP_N,
        algorithm_class: Type[WordScore] = DEFAULT_ALGORITHM_CLASS) -> None:
    """
    Solve in a basic way using user guesses.
    """
    log.info("Wordle Assistant. By Rudolf Cardinal <rudolf@pobox.com>.")
    all_words = read_words_frequencies(wordlist_filename)
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while guesses_remaining > 0:
        # Show the current situation
        state = StateInfo(all_words, clues, guesses_remaining,
                          show_threshold=show_threshold)

        # Provide advice
        _, _ = suggest(
            state,
            score_class=algorithm_class,
            top_n=advice_top_n
        )

        # Read the results of a guess
        clue = Clue.read_from_user()
        clues.append(clue)
        if clue.correct():
            this_guess = N_GUESSES - guesses_remaining + 1
            log.info(f"Success in {this_guess} guesses.")
            return

        guesses_remaining -= 1

    log.info("Out of guesses!")
    guess_words = [c.clue_word for c in clues]
    if len(guess_words) != len(set(guess_words)):
        log.info("You are an idiot.")


# -----------------------------------------------------------------------------
# Autosolver and performance testing framework to compare algorithms
# -----------------------------------------------------------------------------

def autosolve(
        target: str,
        wordlist_filename: str,
        algorithm_class: Type[WordScore] = DEFAULT_ALGORITHM_CLASS) \
        -> List[Clue]:
    """
    Automatically solves, and returns the clues from each guess (including the
    final successful one). (This can go over the Wordle guess limit; avoid
    sharp edges for comparing algorithms.)
    """
    all_words = read_words_frequencies(wordlist_filename)
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while True:
        state = StateInfo(all_words, clues, guesses_remaining)
        guess, certain = suggest(
            state,
            score_class=algorithm_class,
            top_n=DEFAULT_ADVICE_TOP_N,
            silent=True
        )
        clue = Clue.get_from_known_word(guess=guess, target=target)
        clues.append(clue)
        if clue.correct():
            log.info(f"Word is: {guess}. Guesses: {state.pretty_clues}")
            return clues
        guesses_remaining -= 1


def autosolve_single_arg(args: Tuple[str, str, Type[WordScore]]) -> int:
    """
    Version of :func:`autosolve` that takes a single argument, which is
    necessary for some of the parallel processing map functions.

    The argument is a tuple: target, wordlist_filename, score_class.

    Returns the number of guesses taken.
    """
    target, wordlist_filename, score_class = args
    clues = autosolve(target, wordlist_filename, score_class)
    n_guesses = len(clues)
    return n_guesses


def measure_algorithm_performance(
        wordlist_filename: str,
        nwords: int = None,
        nproc: int = DEFAULT_NPROC,
        algorithm_class: Type[WordScore] = DEFAULT_ALGORITHM_CLASS) -> None:
    """
    Test a guess algorithm and report its performance statistics.
    """
    test_words = read_words_frequencies(wordlist_filename, max_n=nwords)
    # Workaround to pass a single argument:
    arglist = (
        (target, wordlist_filename, algorithm_class)
        for target in test_words
    )
    with ProcessPoolExecutor(nproc) as executor:
        guess_counts = list(executor.map(autosolve_single_arg, arglist))
    n_tests = len(guess_counts)
    assert n_tests > 0, "No words!"
    tested = (
        f"all {n_tests} known" if nwords is None
        else f"the first {n_tests}"
    )
    log.info(
        f"Across {tested} words, method {algorithm_class} took: "
        f"min {min(guess_counts)}, "
        f"median {median(guess_counts)}, "
        f"mean {mean(guess_counts)}, "
        f"max {max(guess_counts)} guesses"
    )

# *** frequency by position, and match to that


# =============================================================================
# Self-testing
# =============================================================================

class TestClues(unittest.TestCase):
    @staticmethod
    def _testclue(target: str, guess: str, wordle_feedback: str) -> None:
        # Setup
        wordle_clue = Clue.get_from_typed_pair(guess, wordle_feedback)

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
        log.critical(f"cluegroup: {cluegroup}")
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
        "--freqlist_filename", default=DEFAULT_FREQLIST,
        help=f"File containing all {WORDLEN}-letter words in upper case and "
             f"associated frequencies"
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

    cmd_mkfreq = "make_frequencylist"
    parser_mkfreq = subparsers.add_parser(
        cmd_mkfreq,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_mkfreq.add_argument(
        "--freq_url",
        default="https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_full.txt",  # noqa
        help="URL for word/frequency file"
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
    parser_solve.add_argument(
        "--algorithm", type=str, choices=ALGORITHMS.keys(),
        default=DEFAULT_ALGORITHM,
        help="Algorithm to use"
    )

    cmd_test_performance = "test_performance"
    parser_test_performance = subparsers.add_parser(
        cmd_test_performance,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    main_only_quicksetup_rootlogger(level=logging.DEBUG if args.verbose
                                    else logging.INFO)

    # -------------------------------------------------------------------------
    # Act
    # -------------------------------------------------------------------------
    if args.command == cmd_make:
        make_wordlist(args.source_dict, args.wordlist_filename)
    elif args.command == cmd_mkfreq:
        make_frequency_list(args.freq_url, args.freqlist_filename)
    elif args.command in (cmd_solve, cmd_test_performance):
        algorithm_class = ALGORITHMS[args.algorithm]
        if args.command == cmd_solve:
            solve_interactive(
                wordlist_filename=args.wordlist_filename,
                show_threshold=args.show_threshold,
                advice_top_n=args.advice_top_n,
                algorithm_class=algorithm_class,
            )
        elif args.command == cmd_test_performance:
            measure_algorithm_performance(
                wordlist_filename=args.wordlist_filename,
                nwords=args.nwords,
                nproc=args.nproc,
                algorithm_class=algorithm_class
            )
    else:
        raise AssertionError("argument-parsing bug")


if __name__ == '__main__':
    main()
