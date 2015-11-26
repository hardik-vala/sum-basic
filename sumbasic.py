"""
My implementation of SUMBASIC (Solution to Question 1 for COMP 599 Assignment
#4)

author: Hardik
"""

import argparse
import nltk
import os
import re
import sys

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def count_words(texts):
	"""
	Counts the words in the given texts, ignoring puncuation and the like.

	@param texts - Texts (as a single string or list of strings)
	@return Word count of texts
	"""

	if type(texts) is list:
		return sum(len(t.split()) for t in texts)

	return len(texts.split())


class Preprocessor(object):
	"""
	Preprocesses a document by performing lowercasing, sentence segmentation,
	lemmatization, and stopword removal.
	"""

	# List of English stop words.
	STOPS = stopwords.words('english')

	def __init__(self):
		pass

	def sent_seg(self, doc_path):
		"""
		Performs sentence segmentation on a document.

		@param doc_path - Filepath to document
		@return List of sentences
		"""

		with open(doc_path) as f:
			return sent_tokenize(f.read())

	def preprocess_sent(self, sent):
		"""
		Preprocesses a sentence by performing lowercasing, lemmatization, and
		stopword removal.

		@param sent - Sentence as a string
		@return Preprocessed sentence as a list of words
		"""

		lmtzr = WordNetLemmatizer()

		# Lowercase and tokenize words.
		words = word_tokenize(sent.lower())
		# Assign POS tags to words.
		words_pos = nltk.pos_tag(words)
		# Lemmatize.
		lm_words = []
		for w, p in words_pos:
			try:
				lm_words.append(lmtzr.lemmatize(w, pos=p[0].lower()))
			except KeyError:
				lm_words.append(lmtzr.lemmatize(w))

		# Remove stop words.
		return [w for w in lm_words if w not in self.STOPS]

	def preprocess(self, doc_path):
		"""
		Preprocesses a document by performing lowercasing, sentence
		segmentation, lemmatization, and stopword removal.

		@param doc_path - Filepath to document
		@return List of sentences, with each sentence represented as a list of
			words
		"""

		with open(doc_path) as f:
			# Segment sentences and preprocess each individually.
			return [self.preprocess_sent(sent)
					for sent in sent_tokenize(f.read())]


class Summarizer(object):
	"""
	Multi-Document Auto-Summarizer interface.
	"""

	def __init__(self, limit=100):
		"""
		@param limit - Word length limit (Default is 100)
		"""

		self.limit = 100

		self.preprocessor = Preprocessor()

	def summarize(self, doc_paths):
		"""
		Summarizes documents.

		@param doc_path - Filepaths to documents
		@return Summary as a string
		"""

		raise NotImplementedError


class LeadingSummarizer(Summarizer):
	"""
	Leading baseline summarizer.
	"""

	def summarize(self, doc_paths):
		"""
		Summarizes the documents by taking leading sentences in the first
		document until the word limit is reached.

		@param doc_path - Filepaths to documents
		@return Summary as a string
		"""

		sentences = self.preprocessor.sent_seg(doc_paths[0])

		# If the document already has word count at most the limit, than return
		# the whole document text as the summary.
		if sum(count_words(sent) for sent in sentences) <= self.limit:
			return ' '.join(sentences)

		summary, i = [], 0
		# Add sentences to the summary until word limit is exceeded.
		while count_words(summary) <= self.limit:
			summary.append(sentences[i])
			i += 1

		# Exclude the last sentence in the summary so the resulting summary is
		# in the word limit.
		return ' '.join(summary[:-1])


def sumbasic_summarize(doc_paths, limit, preprocessor, update):
	"""
	Summarizes the documents using the SUMBASIC algorithm.

	@param doc_path - Filepaths to documents
	@param limit - Word limit of summary
	@param preprocessor - Preprocessor
	@param update - If true, then the non-redundancy update is applied to word
		scores, otherwise not
	@return Summary as a string
	"""

	# Preprocess the documents texts.
	sents_processed = [s for p in doc_paths for s in preprocessor.preprocess(p)]

	# Counter for all words.
	cnts = Counter()
	for sent in sents_processed:
		cnts += Counter(sent)

	# Number of tokens.
	N = float(sum(cnts.values()))
	# Unigram probabilities.
	probs = {w: cnt / N for w, cnt in cnts.iteritems()}

	# List of all sentences in all documents.
	sentences = [s for p in doc_paths for s in preprocessor.sent_seg(p)]

	summary = []
	# Add sentences to the summary until there are no more sentences or word
	# limit is exceeded.
	while len(sentences) > 0 and count_words(summary) < limit:
		# Track the max probability of a sentence with corresponding
		# sentence.
		max_prob, max_sent = 0.0, None
		for i, sent in enumerate(sentences):
			# Calculate the probability of the sentence by simply
			# multiplying all unigram probabilities in the preprocessed
			# sentence.
			prob = reduce(lambda x, y: x * y,
				[probs[w] for w in sents_processed[i]])

			if max_prob < prob:
				max_prob, max_sent = prob, sent

		summary.append(max_sent)
		# Remove the sentence added to the summary so it doesn't get
		# selected again.
		sentences.remove(max_sent)

		if update:
			# Apply the update for non-redundancy.
			for w in sents_processed[i]:
				probs[w] = probs[w] ** 2

	# Exclude the last sentence in the summary if its over the word limit.
	if count_words(summary) > limit:
		return ' '.join(summary[:-1])

	return ' '.join(summary)


class SimplifiedSUMBASICSummarizer(Summarizer):
	"""
	Simplified SUMBASIC summarizer that holds word scores constant and does not
	incorporate the non-redundancy update.
	"""

	def summarize(self, doc_paths):
		"""
		Summarizes the documents using the simpilified SUMBASIC algorithm.

		@param doc_path - Filepaths to documents
		@return Summary as a string
		"""

		return sumbasic_summarize(doc_paths, self.limit, self.preprocessor,
			update=False)


class OriginalSUMBASICSummarizer(Summarizer):
	"""
	Original SUMBASIC summarizer that incorporates the non-redundancy update for
	word scores.
	"""

	def summarize(self, doc_paths):
		"""
		Summarizes the documents using the original SUMBASIC algorithm.

		@param doc_path - Filepaths to documents
		@return Summary as a string
		"""

		return sumbasic_summarize(doc_paths, self.limit, self.preprocessor,
			update=True)


def main():
	"""
	Runs the Leading, simpilified SUMBASIC, or original SUMBASIC summarizer on a
	collection of documents.
	"""

	parser_description = ("Runs the Leading, simpilified SUMBASIC, or "
		"original SUMBASIC summarizer on a collection of documents.")
	parser = argparse.ArgumentParser(description=parser_description)

	parser.add_argument('method_name', help="Method name ('leading', "
		"'simplified', or 'orig')")

	parser.add_argument('file_n', help="Filepaths to documents to summarize",
		nargs='+')

	args = parser.parse_args()

	if args.method_name == 'leading':
		summarizer = LeadingSummarizer()
	elif args.method_name == 'simplified':
		summarizer = SimplifiedSUMBASICSummarizer()
	elif args.method_name == 'original':
		summarizer = OriginalSUMBASICSummarizer()
	else:
		ValueError("method_name must be 'leading', 'simplified', or "
			"'original'")

	print summarizer.summarize(args.file_n)


if __name__ == '__main__':
	main()
