import io
import collections
import matplotlib.pyplot as plt
import nltk
import enchant
  ​
words = []
with io.open('corpus.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        words.extend(line.split())
  ​
vocab = collections.Counter(words)
vocab.most_common(10)

''' 
#output
[('i', 174639),
('to', 127111),
('my', 84886),
('is', 69504),
('me', 67741),
('the', 63488),
('not', 51194),
('you', 50830),
('for', 47846),
('?', 45599)]
'''

list(reversed(vocab.most_common()[-10:]))
''' 
#output
[('酒店在haridwar', 1),
  ('谢谢', 1),
  ('谈', 1),
  ('看不懂', 1),
  ('的人##', 1),
  ('现在呢', 1),
  ('王建', 1),
  ('火大金一女', 1),
  ('李雙鈺', 1),
  ('拜拜', 1)]
'''
#learning fasttext
# $ fasttext skipgram -input corpus.txt -output model -minCount 1 -minn 3 -maxn 6 -lr 0.01 -dim 100 -ws 3 -epoch 10 -neg 20

from gensim.fasttext import FastText 
model = FastText.load_fasttext_format('model')  
print(model.wv.most_similar('recharge', topn=5))
print(model.wv.most_similar('reminder', topn=5))
print(model.wv.most_similar('thanks', topn=5))


'''
#output
[('rechargecharge', 0.9973811507225037),
 ('rechargea', 0.9964320063591003),
 ('rechargedd', 0.9945225715637207),
 ('erecharge', 0.9935820698738098),
 ('rechargw', 0.9932199716567993)]
  
 [("reminder'⏰", 0.992865264415741),
 ('sk-reminder', 0.9927705526351929),
 ('myreminder', 0.992688775062561),
 ('reminderw', 0.9921447038650513),
 ('ofreminder', 0.992128312587738)]
  
 [('thanksd', 0.996020495891571),
 ('thanksll', 0.9954444169998169),
 ('thankseuy', 0.9953703880310059),
 ('thankss', 0.9946843385696411),
'''

word_to_mistakes = collections.defaultdict(list)
nonalphabetic = re.compile(r'[^a-zA-Z]')
  
for word, freq in vocab.items():
    if freq < 500 or len(word) <= 3 or nonalphabetic.search(word) is not None:
        #  To keep this task simple, we will not try finding
        #  spelling mistakes for words that occur less than 500 times
        #  or have length less than equal to 3 characters
        #  or have anything other than English alphabets
        continue
  
    # Query the fasttext model for 50 closest neighbors to the word
    similar_words = model.wv.most_similar(word, topn=50)
    for similar_word in results:
        if include_spell_mistake(word, similar_word, similarity_score):
            word_to_mistakes[word].append(similar_word)


enchant_us = enchant.Dict('en_US')
spell_mistake_min_frequency = 5
fasttext_min_similarity = 0.96
def include_spell_mistake(word, similar_word, score):
    """
    Check if similar word passes some rules to be considered a spelling mistake
    
    Rules:
       1. Similarity score should be greater than a threshold
       2. Length of the word with spelling error should be greater than 3.
       3. spelling mistake must occur at least some N times in the corpus
       4. Must not be a correct English word.
       5. First character of both correct spelling and wrong spelling should be same.
       6. Has edit distance less than 2
    """
    edit_distance_threshold = 1 if len(word) <= 4 else 2
    return (score > fasttext_min_similarity
            and len(similar_word) > 3
            and vocab[similar_word] >= spell_mistake_min_frequency
            and not enchant_us.check(similar_word)
            and word[0] == similar_word[0]
            and nltk.edit_distance(word, similar_word) <= edit_distance_threshold)

'''
Some rules are straightforward:

Spelling mistake word vector must have high vector similarity with correct word’s vector,
Spelling mistake word must occur at least 5 times in our corpus,
It must have more than three characters
It should not be a legit English word (we use Enchant which has a convenient dictionary check function).
'''
#At this point, most of our work is done, let’s check word_to_mistakes:

'''
print(list(word_to_mistakes.items())[:10])
 
[
 ('want', ['wann', 'wanto', 'wanr', 'wany']),
 ('have', ['havea', 'havr']),
 ('this', ['thiss', 'thise']),
 ('please', ['pleasee', 'pleasr', 'pleasw', 'pleaseee', 'pleae', 'pleaae']),
 ('number', ['numbe', 'numbet', 'numbee', 'numbr']),
 ('call', ['calll']),
 ('will', ['willl', 'wiill']),
 ('account', ['aaccount', 'acccount', 'accouny', 'accoun', 'acount', 'accout', 'acoount']),
 ('match', ['matche', 'matchs', 'matchh', 'matcj', 'matcg', 'matc', 'matcha']), ('recharge', ['rechargr', 'recharg', 'rechage', 'recharege', 'recharje', 'recharhe', 'rechare'])
 ]
'''

#an inverted index for fast lookup:
inverted_index = {}
for word, mistakes in word_to_mistakes.items():
    for mistake in mistakes:
        if mistake != word:
            inverted_index[mistake] = word

'''
However, this method is not entirely accurate.

Very common proper nouns can still slip through the rules and end up being corrected when they shouldn’t be.
A manual inspection must still be done once to remove errors.
Another drawback is spelling mistakes that never occurred in the corpus will not have a correction in the index. Nevertheless, this was a fun experiment.
'''
