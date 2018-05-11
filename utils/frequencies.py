from collections import Counter
import pickle
from nltk.corpus import brown, cess_esp
from nltk.util import ngrams
from collections import defaultdict

# Map words and the trigrams contained in them to their absolute frequency in the
# Brown Corpus (English) or the CESS_ESP corpus (Spanish)
if __name__=='__main__':
    brown_words = brown.words()
    english_freqs = Counter(brown_words)
    esp_words = cess_esp.words()
    spanish_freqs = Counter(esp_words)
    with open( 'spanish_freqs.pkl', 'wb') as esp_f:
            pickle.dump(spanish_freqs, esp_f)
    with open( 'english_freqs.pkl', 'wb') as en_f:
            pickle.dump(english_freqs, en_f)
            
    en_trigrams = defaultdict(int)
    for word in brown_words:
        for trigram in ['{}{}{}'.format(t[0],t[1], t[2]) for t in list(ngrams(word, 3))]:
            en_trigrams[trigram] += 1
    with open( 'english_trigram_freqs.pkl', 'wb') as en_t_f:
            pickle.dump(en_trigrams, en_t_f)
            
    esp_trigrams = defaultdict(int)
    for word in esp_words:
        for trigram in ['{}{}{}'.format(t[0],t[1], t[2]) for t in list(ngrams(word, 3))]:
            esp_trigrams[trigram] += 1
    with open( 'spanish_trigram_freqs.pkl', 'wb') as esp_t_f:
            pickle.dump(en_trigrams, esp_t_f)
    
        
        
       
            
        
    
    
    
    
    