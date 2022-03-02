import math
import re

class Bayes_Classifier:

    stop_words = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
    'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 
    'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has',
     "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 
     'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in',
      'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 
      'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
       'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 
       'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 
       'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
        "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 
        'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 
        'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's",
        'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 
        'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 
        'yours', 'yourself', 'yourselves'} 
    words = {}
    by_class = [{}, {}]
    word_like_by_class = [{}, {}]
    class_probs = [0, 0]
    def __init__(self):
        pass

    def train(self, lines):
        for line in lines:
            score, line_id, line_text = line.split("|")
            score = self.conv_score(score)
            line_words = self.parse_line(line_text)
            for word in line_words:
                self.add_word(word, score)

        print("Lines Read")
        self.generate_class_probs()
        print("class probabilities generated")
        self.generate_word_likes()
        print("Training Done")
    
    def generate_class_probs(self):
        total = sum([len(self.by_class[x]) for x in range(2)])
        for i in range(2):
            this_class = self.get_num_words_by_class(i)
            self.class_probs[i] = math.log(this_class / total)

    def generate_word_likes(self):
        for i in range(2):
            for word in self.words.keys():
                self.word_like_by_class[i][word] = self.get_likelihood_word(word, i)
            print("Class {} done".format(i))

    def add_word(self, word, score):
        if word not in self.by_class[score].keys():
            self.by_class[score][word] = 1
            self.by_class[(score + 1) % 2][word] = 1
        self.by_class[score][word] += 1

        if word not in self.words:
            self.words[word] = [1, 1]
        self.words[word][score] += 1

    def conv_score(self, score):
        if score == "1":
            return 0
        else:
            return 1
    
    def conv_score_back(self, score):
        if score == 0:
            return "1"
        else:
            return "5"

    def get_word_like_by_class(self, word, score):
        if word in self.word_like_by_class[score].keys():
            return self.word_like_by_class[score][word]
        else:
            return 0

    def parse_line(self, line):
        line = line.strip()
        line = line.lower()
        line = re.sub(r'[^\w\s]','',line)
        line = [word for word in line.split(" ") if word not in self.stop_words]
        return line

    def get_num_words_by_class(self, class_id):
        words = self.by_class[class_id]
        return sum([words[word] for word in words.keys()])

    def get_likelihood_word(self, word, score):
        class_length = self.get_num_words_by_class(score)
        result = self.by_class[score][word]
        result /= class_length
        return math.log(result)

    def get_likelihood_sentence(self, sentence, score):
        line_sum = 0
        for word in sentence:
            line_sum += self.get_word_like_by_class(word, score)
        return line_sum

    def get_prob_line(self, line, score):
        prob_score = self.class_probs[score]
        line_like = self.get_likelihood_sentence(line, score)
        return prob_score + line_like

    def classify_line(self, line):
        score, line_id, line_text = line.split("|")
        line_text = self.parse_line(line_text)
        probabilities = [self.get_prob_line(line_text, score) for score in range(2)]
        max_i = -1
        max_val = float('-inf')
        for i in range(2):
            if probabilities[i] > max_val:
                max_i = i
                max_val = probabilities[i]
        return max_i

    def classify(self, lines):
        classes = [self.conv_score_back(self.classify_line(line)) for line in lines]
        return classes


    def debug_line(self, line):
        score, line_id, line_text = line.split("|")

        probabilities = [self.get_prob_line(line_text, score) for score in range(2)]
        print(probabilities)

        line_text = self.parse_line(line_text)
        
        for word in line_text:
            print("{} : {}".format(word, [self.get_word_like_by_class(word, num) for num in range(2)]))
        
        print(self.class_probs)
