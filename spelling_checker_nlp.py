import nltk
import tkinter as tk
from tkinter import Text, messagebox
from nltk.util import ngrams
from collections import Counter
import re
import pandas as pd
import joblib

# Ensure the necessary NLTK resources are downloaded
# nltk.download("punkt")

dict = pd.read_csv('scisumm.csv')

word_bank = []
grammar_bank = []

# Preprocess the corpus
for index in range(dict.shape[0]):
    text = dict['text'][index].lower()
    # Clear Number and Symbol
    re_processed = re.sub(r"[^ \w-]|\d","",text)
    re_processed = re.sub(r"-|  |_"," ",re_processed)
    grammar_bank.extend(nltk.tokenize.word_tokenize(text))
    word_bank.extend(nltk.tokenize.word_tokenize(re_processed))    

# Revmove duplicates
word_bank = pd.unique(word_bank)


corpus_tokens = grammar_bank
bigrams = list(ngrams(corpus_tokens, 2))
bigram_counts = Counter(bigrams)
unigram_counts = Counter(word_bank)

# load model
model = joblib.load('lm.pkl')

def levenshtein_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize the table
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    
    return dp[m][n]

# Function to calculate bigram probabilities
def bigram_prob_v2(w1, w2):
    return model.score(w2,[w1]) / unigram_counts[w1] if unigram_counts[w1] > 0 else 1e-6

# Function to calculate minimum edit distance and suggest corrections
def suggest_corrections(word):
    suggestions = []
    for candidate in unigram_counts.keys():
        dist = levenshtein_distance(word, candidate)
        suggestions.append((candidate, dist))
    suggestions.sort(key=lambda x: x[1])
    return [s[0] for s in suggestions[:5]]

# Generate possible corrections
def generate_candidates(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    inserts = [L + c + R for L, R in splits for c in letters]
    substitutes = [L + c + R[1:] for L, R in splits if R for c in letters]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

    return set(deletes + inserts + substitutes + transposes) & set(unigram_counts.keys())

# Noisy Channel Model using Bigram Probabilities
def correct_word(prev_word, word):
    candidates = generate_candidates(word)
    
    if not candidates:
        return word  # Return original word if no candidates found

    ranked_candidates = sorted(
        candidates, 
        key=lambda w: bigram_prob_v2(w, prev_word) / (levenshtein_distance(word, w) + 1), 
        reverse=True
    )

    return ranked_candidates[:5]

# Function to check spelling
misspelled_words = []
grammar_issue = []

def check_spelling(input_text):
    global misspelled_words
    misspelled_words = []
    input_tokens = nltk.word_tokenize(input_text.lower())
    for word in input_tokens:
        if word not in unigram_counts and not re.search(r"[^ \w-]|\d",word):
            misspelled_words.append(word)
    return misspelled_words

# Function to check grammar using bigrams
def check_grammar_with_bigram_v2(input_text):
    """
    Use bigrams from the corpus to identify uncommon or incorrect word combinations.
    """
    global bigram_counts, grammar_issue
    input_tokens = nltk.word_tokenize(input_text.lower())
    input_bigrams = list(ngrams(input_tokens, 2))

    for bigram in input_bigrams:
        if bigram_counts[bigram] == 0 and bigram[1] in unigram_counts and bigram[0] not in misspelled_words and bigram[1] not in misspelled_words:
            grammar_issue.append(bigram)
            
    return grammar_issue



# GUI setup
def highlight_misspelled():
    text_area.tag_remove("misspelled", "1.0", tk.END)
    for word in misspelled_words:
        start = "1.0"
        while True:
            start = text_area.search(word, start, stopindex=tk.END, nocase=True)
            if not start:
                break
            end = f"{start}+{len(word)}c"
            text_area.tag_add("misspelled", start, end)
            start = end

def highlight_grammar():
    text_area.tag_remove("grammar_issue", "1.0", tk.END)
    text_area.tag_remove("grammar_issue", "1.0", tk.END)
    for word in grammar_issue:
        joined_word = ' '.join(word)
        start = "1.0"
        while True:
            start = text_area.search(word[0], start, stopindex=tk.END, nocase=True)
            if not start:
                break
            end = f"{start}+{len(joined_word)}c"
            text_area.tag_add("grammar_issue", start, end)
            start = end

# On-click correction
selected_word = None

def on_word_click(event):
    global selected_word

    cursor_index = text_area.index(tk.CURRENT)
    word_start = text_area.search(r"\m\w+\M", cursor_index, backwards=True, regexp=True)
    word_end = text_area.search(r"\M", word_start, forwards=True, regexp=True)
    if word_start and word_end:
        selected_word = text_area.get(word_start, word_end)
        if selected_word in misspelled_words:
            suggestions = suggest_corrections(selected_word)
            suggestion_popup(suggestions)
        for word in grammar_issue:
            if selected_word in word[1]:
                suggestions = correct_word(word[0],word[1])
                suggestion_popup(suggestions)

# Suggestion popup
def suggestion_popup(suggestions):
    def replace_word(selected_suggestion):
        global selected_word, misspelled_words, grammar_issue
        content = text_area.get("1.0", tk.END)
        content = content.replace(selected_word, selected_suggestion, 1)
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, content)
        selected_word = None
        popup.destroy()
        misspelled_words = []
        grammar_issue = []
        check_text_v2()

    popup = tk.Toplevel(root)
    popup.title("Suggestions")
    for suggestion in suggestions:
        btn = tk.Button(popup, text=suggestion, command=lambda s=suggestion: replace_word(s))
        btn.pack()

# Check text for spelling errors
def check_text_v2():
    input_text = text_area.get("1.0", tk.END)
    check_spelling(input_text)
    highlight_misspelled()
    check_grammar_with_bigram_v2(input_text)
    highlight_grammar()

# Search functionality
def search_word():
    search_term = search_entry.get().lower()
    word_list.delete(0, tk.END)
    for word in unigram_counts.keys():
        if search_term in word:
            word_list.insert(tk.END, word)

# GUI components
root = tk.Tk()
root.title("Spelling and Grammar Checker")

frame = tk.Frame(root)
frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

text_area = Text(frame, wrap=tk.WORD, font=("Arial", 14))
text_area.pack(expand=True, fill=tk.BOTH)

check_button = tk.Button(root, text="Check Text", command=check_text_v2)
check_button.pack()

text_area.tag_config("misspelled", foreground="red", underline=1)
text_area.tag_config("grammar_issue", foreground="blue", underline=1)
text_area.bind("<Button-1>", on_word_click)

# Word list panel
word_panel = tk.Frame(root)
word_panel.pack(side=tk.RIGHT, fill=tk.Y)

search_label = tk.Label(word_panel, text="Search Word:")
search_label.pack()
search_entry = tk.Entry(word_panel)
search_entry.pack()
search_button = tk.Button(word_panel, text="Search", command=search_word)
search_button.pack()

word_list = tk.Listbox(word_panel, height=20)
word_list.pack(fill=tk.BOTH, expand=True)
for word in unigram_counts.keys():
    word_list.insert(tk.END, word)

root.mainloop()
