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
import multiprocessing
import os
import re
from statistics import median, mean
from typing import Any, Dict, Iterable, List, Set, Tuple, Type, Union

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
CHAR_PRESENT = "-"
CHAR_CORRECT = "="
FEEDBACK_REGEX = re.compile(
    rf"^[\{CHAR_ABSENT}\{CHAR_PRESENT}\{CHAR_CORRECT}]{{{WORDLEN}}}$",  # noqa
    re.IGNORECASE
)

COLOUR_FG = "white"
COLOUR_BG_ABSENT = "black"
COLOUR_BG_PRESENT_WRONG_LOCATION = "orange"
COLOUR_BG_PRESENT_RIGHT_LOCATION = "green"
STYLE = "bold"

N_GUESSES = 6
ALL_LETTERS = tuple(
    chr(_letter_ascii_code)
    for _letter_ascii_code in range(ord('A'), ord('Z') + 1)
)

DEFAULT_SHOW_THRESHOLD = 100
DEFAULT_ADVICE_TOP_N = 10
DEFAULT_CPU_COUNT = multiprocessing.cpu_count()


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

# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Reading word lists
# -----------------------------------------------------------------------------

def make_wordlist(from_filename: str,
                  to_filename: str) -> None:
    """
    Reads a dictionary file and creates a list of 5-letter words.
    """
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

    # *** Not yet done: if we know there's an "E", this continues to give
    #     points for guessing E.
    # *** Not yet done: formal "reduction of possibility space" measure.


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
        print(f"{prefix2} You have entered this clue: {clue}")
        return clue

    @classmethod
    def get_from_known_word(cls, guess: str, target: str) -> "Clue":
        """
        For automatic testing: given that we know the word, return the clue.
        """
        feedback = []  # type: List[CharFeedback]
        for g_pos, g_char in enumerate(guess):
            if target[g_pos] == g_char:
                f = CharFeedback.PRESENT_RIGHT_LOCATION
            elif g_char in target:
                f = CharFeedback.PRESENT_WRONG_LOCATION
            else:
                f = CharFeedback.ABSENT
            feedback.append(f)
        return cls(guess, feedback)

    def __str__(self) -> str:
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

    def letter_known_absent(self, letter: str) -> bool:
        """
        Is the specified letter known to be absent from the target word, from
        this clue?
        """
        for c, f in self.char_feedback_pairs:
            if c == letter and f == CharFeedback.ABSENT:
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
        Was the clue correct?
        """
        return all(
            c == CharFeedback.PRESENT_RIGHT_LOCATION for c in self.feedback
        )


# -----------------------------------------------------------------------------
# Alphabet
# -----------------------------------------------------------------------------

class Alphabet:
    """
    Represents summary information about every letter, from clues.
    """
    def __init__(self, clues: List[Clue]) -> None:
        self.absent = set(
            x
            for x in ALL_LETTERS
            for c in clues
            if c.letter_known_absent(x)
        )
        self.present = set(
            x
            for x in ALL_LETTERS
            for c in clues
            if c.letter_known_present(x)
        )
        self.present_known_location = set(
            x
            for x in ALL_LETTERS
            for c in clues
            if c.letter_known_location(x)
        )
        self.present_unknown_location = (
            self.present
            - self.present_known_location
        )
        self.unknown = (
            set(ALL_LETTERS)
            - self.absent
            - self.present
        )

    def __str__(self) -> str:
        pk = "".join(sorted(self.present_known_location))
        pu = "".join(sorted(self.present_unknown_location))
        a = "".join(sorted(self.absent))
        u = "".join(sorted(self.unknown))
        return (
            f"Present, known location: {pk}. "
            f"Present, unknown location: {pu}. "
            f"Absent: {a}. "
            f"Unknown: {u}."
        )


# -----------------------------------------------------------------------------
# Scoring potential guesses
# -----------------------------------------------------------------------------

@total_ordering
class WordScore:
    def __init__(self,
                 word: str,
                 possible_words: Set[str],
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

    @property
    def score(self) -> Union[float, Iterable[float]]:
        """
        The key "thinking" algorithm. Returns a score for this word: how good
        would it be to use this as the next guess?

        Can return a float, or a tuple of floats, etc.
        """
        raise NotImplementedError


class WordScoreExplore(WordScore):
    """
    Performance across our 5905 words:

    - min 1, median 5, mean 5.449110922946655, max 14 guesses
    """
    @property
    def score(self) -> Tuple[float, int]:
        s = 0.0
        possible = int(self.word in self.possible_words)
        if self.n_guesses_left <= 1 and not possible:
            # If we have only one guess left, we must make a stab at the word
            # itself. So eliminate words that are impossible.
            return s, possible
        for letter in set(self.word):
            s += self.letter_freq[letter]
        return s, possible


DEFAULT_SCORE_CLASS = WordScoreExplore


# -----------------------------------------------------------------------------
# Using a particular scoring method, calculate a best guess
# -----------------------------------------------------------------------------

def suggest(all_words: List[str],
            clues: List[Clue],
            guesses_remaining: int,
            score_class: Type[WordScore] = DEFAULT_SCORE_CLASS,
            show_threshold: int = DEFAULT_SHOW_THRESHOLD,
            top_n: int = 5,
            silent: bool = False) -> Tuple[str, bool]:
    """
    Show advice to the user: what word should be guessed next?

    Returns the best guess (or, the first of the equally good guesses) for
    automatic checking.

    Returns: guess, certain
    """
    possible_words = set(
        w
        for w in all_words
        if all(c.compatible(w) for c in clues)
    )
    if len(possible_words) == 1:
        answer = next(iter(possible_words))
        if not silent:
            log.info(f"Word is: {answer}")
        return answer, True
    elif len(possible_words) == 0:
        raise ValueError("Word is not in our dictionary!")

    letter_freq = get_letter_frequencies(possible_words)
    if not silent:
        this_guess = N_GUESSES - guesses_remaining + 1
        log.info(f"This is guess {this_guess}. "
                 f"Guesses remaining: {guesses_remaining}. "
                 f"Guesses so far: {prettylist(clues)}")
        log.info(f"Number of possible words: {len(possible_words)}")
        if len(possible_words) <= show_threshold:
            log.info(f"Possibilities: {prettylist(sorted(possible_words))}")
        else:
            log.info(f"Not yet showing possibilities (>{show_threshold})")
        log.info(f"Letter frequencies in remaining possible words: "
                 f"{letter_freq}")

    # Any word may be a candidate for a guess, not just the possibilities --
    # for example, if we know 4/5 letters in the correct positions early on, we
    # might be better off with a guess that has lots of options for that final
    # letter, rather than guessing them sequentially in a single position.
    # Therefore, we generate a score for every word in all_words.
    options = [
        score_class(
            word=w,
            possible_words=possible_words,
            letter_frequencies_in_possible_words=letter_freq,
            clues=clues,
            n_guesses_left=guesses_remaining
        )
        for w in all_words
    ]
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

def solve_interactive(wordlist_filename: str,
                      show_threshold: int = DEFAULT_SHOW_THRESHOLD,
                      advice_top_n: int = DEFAULT_ADVICE_TOP_N) -> None:
    """
    Solve in a basic way using user guesses.
    """
    log.info("Wordle Assistant. By Rudolf Cardinal <rudolf@pobox.com>.")
    all_words = read_words(wordlist_filename)
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while guesses_remaining > 0:
        # Show the current situation

        # Provide advice
        _, _ = suggest(
            all_words=all_words,
            clues=clues,
            guesses_remaining=guesses_remaining,
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
    guess_words = [c.word for c in clues]
    if len(guess_words) != len(set(guess_words)):
        log.info("You are an idiot.")


# -----------------------------------------------------------------------------
# Autosolver and performance testing framework to compare algorithms
# -----------------------------------------------------------------------------

def autosolve(target: str,
              wordlist_filename: str,
              score_class: Type[WordScore] = DEFAULT_SCORE_CLASS) -> List[Clue]:
    """
    Automatically solves, and returns the clues from each guess (including the
    final successful one). (This can go over the Wordle guess limit; avoid
    sharp edges for comparing algorithms.)
    """
    all_words = read_words(wordlist_filename)
    guesses_remaining = N_GUESSES
    clues = []  # type: List[Clue]
    while True:
        guess, certain = suggest(
            all_words=all_words,
            clues=clues,
            guesses_remaining=guesses_remaining,
            score_class=score_class,
            top_n=DEFAULT_ADVICE_TOP_N,
            silent=True
        )
        clue = Clue.get_from_known_word(guess=guess, target=target)
        clues.append(clue)
        if certain:
            log.info(f"Word is: {guess}. "
                     f"Guesses: {prettylist(clues)}")
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


def test_performance(
        wordlist_filename: str,
        score_class: Type[WordScore] = DEFAULT_SCORE_CLASS,
        nproc: int = DEFAULT_CPU_COUNT) -> None:
    """
    Test a guess algorithm and report its performance statistics.
    """
    all_words = read_words(wordlist_filename)
    # Workaround to pass a single argument:
    arglist = (
        (target, wordlist_filename, score_class)
        for target in all_words
    )
    with multiprocessing.Pool(nproc) as pool:
        guess_counts = pool.map(autosolve_single_arg, arglist)
    n_tests = len(guess_counts)
    assert n_tests > 0, "No words!"
    log.info(
        f"Across all {n_tests} known words, method {score_class} took: "
        f"min {min(guess_counts)}, "
        f"median {median(guess_counts)}, "
        f"mean {mean(guess_counts)}, "
        f"max {max(guess_counts)} guesses"
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

    cmd_test_performance = "test_performance"
    parser_test_performance = subparsers.add_parser(
        cmd_test_performance,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_test_performance.add_argument(
        "--nproc", type=int,
        help="Number of parallel processes"
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
        solve_interactive(
            wordlist_filename=args.wordlist_filename,
            show_threshold=args.show_threshold,
            advice_top_n=args.advice_top_n
        )
    elif args.command == cmd_test_performance:
        test_performance(
            wordlist_filename=args.wordlist_filename,
            nproc=args.nproc
        )
    else:
        raise AssertionError("argument-parsing bug")


if __name__ == '__main__':
    main()
