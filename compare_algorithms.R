#!/usr/bin/env Rscript

library(data.table)
library(tidyverse)
library(viridis)

MAX_GUESSES <- 6

algodata <- data.table(rbind(
    read.csv("out_UnknownLetterExplorerAmongstPossible.csv"),
    read.csv("out_UnknownLetterExplorerMindingGuessCount.csv"),
    read.csv("out_PositionalExplorerAmongstPossible.csv"),
    read.csv("out_PositionalExplorerMindingGuessCount.csv"),
    read.csv("out_Eliminator.csv"),
    read.csv("out_EliminatorAmongstPossible.csv")
))

algosummary <- (
    algodata %>%
    mutate(
        success = n_guesses <= MAX_GUESSES
    ) %>%
    group_by(algorithm) %>%
    summarize(
        min = min(n_guesses),
        median = median(n_guesses),
        mean = mean(n_guesses),
        max = max(n_guesses),
        prop_success = mean(success)
    )
)

fig <- (
    ggplot(algodata, aes(x = n_guesses, fill = algorithm)) +
    # geom_density(alpha = 0.5)
    geom_histogram(position = "dodge") +
    scale_fill_viridis(discrete = TRUE) +
    geom_vline(xintercept = MAX_GUESSES + 0.5) +
    scale_x_continuous(
        name = "Number of guesses",
        breaks = 1:max(algodata$n_guesses),
        limits = c(1, NA)
    ) +
    xlab("Number of guesses") +
    ylab("Number of words")
)
print(fig)
