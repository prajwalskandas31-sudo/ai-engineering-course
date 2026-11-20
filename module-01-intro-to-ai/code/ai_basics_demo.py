"""
===============================================================================
Module 1: Introduction to AI: Hands-on Demo
Rules vs learning, and a tiny next-word predictor
===============================================================================

Two ideas from Module 1, in code you can run and change:

  Part 1  Traditional programming vs machine learning.
          We write spam rules by hand, then let a model learn its own rules
          from labelled examples, and compare both on messages neither has seen.
  Part 2  How a language model "thinks", in miniature.
          A next-word predictor that counts which word follows which, the same
          basic idea (predict the next word) behind ChatGPT, Gemini and Claude,
          just billions of times smaller.

Run:  python ai_basics_demo.py      (needs: pip install scikit-learn)
===============================================================================
"""

from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -----------------------------------------------------------------------------
# The data: short messages labelled spam (1) or not spam (0)
# -----------------------------------------------------------------------------
TRAIN = [
    ("WIN a FREE iPhone now, click here", 1),
    ("Congratulations! You won a cash prize, claim today", 1),
    ("Limited offer: free gift card, reply YES", 1),
    ("Urgent: your account will be blocked, verify now", 1),
    ("Earn money fast from home, no experience needed", 1),
    ("Exclusive deal just for you, buy now and save 90%", 1),
    ("You have been selected for a reward, click the link", 1),
    ("Cheap loans approved instantly, apply now", 1),
    ("Are we still meeting for lunch tomorrow?", 0),
    ("Please send me the notes from today's class", 0),
    ("The project review is moved to Friday at 3 pm", 0),
    ("Happy birthday! Have a great day", 0),
    ("Can you share the slides for Module 1?", 0),
    ("I will be late, stuck in traffic", 0),
    ("Mom asked if you are coming home this weekend", 0),
    ("Thanks for the help with the Python assignment", 0),
]

TEST = [
    ("Claim your free prize now", 1),
    ("Verify your bank account urgently or lose access", 1),
    ("Get rich quick, limited seats, reply now", 1),
    ("Lunch at 1 pm in the canteen?", 0),
    ("Can you send the assignment file?", 0),
    ("Free entry to the college fest this Friday for students", 0),
]


# -----------------------------------------------------------------------------
# Part 1a: traditional programming: a human writes the rules
# -----------------------------------------------------------------------------
SPAM_WORDS = {"free", "win", "won", "prize", "click", "offer", "cash"}


def rule_based_is_spam(message):
    """Data + Rules -> Answers. The rule: any 'spammy' word means spam."""
    words = {w.strip(".,!?:").lower() for w in message.split()}
    return int(len(words & SPAM_WORDS) > 0)


def part1_rules_vs_learning():
    print("=" * 70)
    print("PART 1: TRADITIONAL PROGRAMMING vs MACHINE LEARNING")
    print("=" * 70)

    # Part 1a: the hand-written rules
    print(f"Hand-written rule: spam if the message contains any of {sorted(SPAM_WORDS)}\n")

    # Part 1b: machine learning: the model finds its own rules from examples
    vectorizer = CountVectorizer(lowercase=True)
    X_train = vectorizer.fit_transform([m for m, _ in TRAIN])
    y_train = [label for _, label in TRAIN]
    model = MultinomialNB().fit(X_train, y_train)

    X_test = vectorizer.transform([m for m, _ in TEST])
    ml_predictions = model.predict(X_test)

    print(f"{'Unseen message':<58} {'Truth':>5} {'Rules':>6} {'ML':>4}")
    rules_right = ml_right = 0
    for (message, truth), ml_pred in zip(TEST, ml_predictions):
        rule_pred = rule_based_is_spam(message)
        rules_right += rule_pred == truth
        ml_right += ml_pred == truth
        label = lambda v: "spam" if v else "ok"
        print(f"{message:<58} {label(truth):>5} {label(rule_pred):>6} {label(ml_pred):>4}")

    print(f"\nRules got {rules_right}/{len(TEST)} right; the learned model got {ml_right}/{len(TEST)}.")
    print("The rules miss spam that avoids their exact words, and wrongly flag")
    print("'Free entry to the college fest' just because it says 'free'.")

    # What did the model learn? The words that push hardest towards 'spam'
    words = vectorizer.get_feature_names_out()
    spam_lean = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    top = sorted(zip(spam_lean, words), reverse=True)[:8]
    print("\nWords the model learned are most 'spammy' (nobody wrote these rules):")
    print("  " + ", ".join(w for _, w in top))
    print()


# -----------------------------------------------------------------------------
# Part 2: a next-word predictor (a language model in miniature)
# -----------------------------------------------------------------------------
CORPUS = """
the quick brown fox jumps over the lazy dog .
the lazy dog sleeps in the sun .
the quick cat jumps over the wall .
a lazy dog dreams of the quick fox .
the brown dog jumps over the lazy cat .
""".split()


def build_next_word_model(words):
    """Count, for every word, which words came right after it."""
    following = defaultdict(Counter)
    for current, nxt in zip(words, words[1:]):
        following[current][nxt] += 1
    return following


def predict_next(model, word, k=3):
    counts = model[word]
    total = sum(counts.values())
    return [(w, c / total) for w, c in counts.most_common(k)]


def generate(model, start, length=8):
    out = [start]
    for _ in range(length):
        options = model[out[-1]]
        if not options:
            break
        out.append(options.most_common(1)[0][0])   # always pick the likeliest word
    return " ".join(out)


def part2_next_word_prediction():
    print("=" * 70)
    print("PART 2: PREDICTING THE NEXT WORD")
    print("=" * 70)
    model = build_next_word_model(CORPUS)
    print(f"Learned from {len(CORPUS)} words of text.\n")
    for word in ["the", "lazy", "quick", "jumps"]:
        guesses = ", ".join(f"'{w}' ({p:.0%})" for w, p in predict_next(model, word))
        print(f"After '{word}', the model expects: {guesses}")
    print(f"\nGenerated text starting from 'the': \"{generate(model, 'the')}\"")
    print("\nA real LLM does the same job (predict the next word) but learns from")
    print("trillions of words with a deep neural network, so it can use far more")
    print("than one previous word of context.")


def main():
    part1_rules_vs_learning()
    part2_next_word_prediction()


if __name__ == "__main__":
    main()
