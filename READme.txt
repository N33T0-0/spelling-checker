Methods:
https://www.kaggle.com/datasets/jawakar/scisummnet-corpus?resource=download
Scientific text data from kaggle. with 1009 scientific articles/ journals.

Text data is tokenized, Lemmatized and Cleaned using Regular Expression to create two sets of data.
Word_Bank -> Used for Spelling Correction (Only Contains Unique Words)
Grammar_Bank -> Used for Grammar Correction

Levenshtein Distance -> Used to determine the edit distance between the misspelled word and target word
Noisy Channel + Bigram Probability / Levenshtein Distance -> Used to determine the best word for grammatical error

Unigram -> Detect Spelling error (Non-Real Word Error).
Bigram -> Detect Grammar Error (Real Word Error).

Program will check Mispelled word and grammatical error. When both have error, program will first point out mispelled word, only it will point out possible grammar errors.

FOR Record DEMO:
They have used varoius lexical clustering algorithms in their text classification model. (Use this as example, Perfect demonstration)

Limitations
He is check if the room is empty. <-- Correct would be: He is checking if the room is empty.
If the a set of words (bigram) are not in the corpus, it will detect as grammar error.

I have too apples. <-- Correct would be: I have two apples.
only can change the second word. Program assumes that only the second word has issue.