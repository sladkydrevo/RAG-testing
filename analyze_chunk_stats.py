import numpy as np
import pandas as pd
import rag_functions as rf


class Chunker:
    def __init__(self, texts_path, chunk_size, overlap):
        self.texts = rf.load_texts(texts_path)
        self.chunk_counts, self.chunk_data = rf.chunk_texts(self.texts, chunk_size=chunk_size, overlap=overlap)
        self.chunk_names, self.text_chunks = rf.split_dict_data(self.chunk_data)

    def get_stats_for_chunks(self):
        counts = np.array(self.chunk_counts)
        stats = {
            "Celkový počet chunků" : sum(counts), # celkový počet chunků
            "Průměrný počet chunků na text" : counts.mean(), # průměrný počet chunků z jednoho textu (znaky)
        }
        return stats
    
    def get_stats_chars(self):
        lengths = np.array([len(text) for text in self.text_chunks])
        chars_stats = {
            "Průměrná délka" : lengths.mean(), # průměrná délka chunku (znaky)
            "Medián" : np.median(lengths), # medián délek chunku (znaky)
            "Směrodatná odchylka" : lengths.std(), # SD délek chunku (znaky)
            "Minimální délka" : min(lengths), # nejmenší délka chunku (znaky)
            "Maximální délka" : max(lengths), # největší délka chunku (znaky)
        }
        return chars_stats
    
    def get_stats_words(self):
        words_counts = np.array([len(chunk.split()) for chunk in self.text_chunks])
        words_stats = {
            "Průměrná délka" : words_counts.mean(), # průměrný počet chunků z jednoho textu (slova)
            "Medián" : np.median(words_counts), # medián délek chunku (slova)
            "Směrodatná odchylka" : words_counts.std(), # SD délek chunku (znaky)
            "Minimální délka" : min(words_counts), # nejmenší délka chunku (slova)
            "Maximální délka" : max(words_counts) # největší délka chunku (slova)
        }
        return words_stats
    
    def get_all_stats(self):
        all_stats = {
            "Chunky" : self.get_stats_for_chunks(), 
            "Znaky" : self.get_stats_chars(), 
            "Slova" : self.get_stats_words()
        }
        self.all_stats = all_stats
        return all_stats


def get_stats_table(stats):
    df = pd.DataFrame(stats.items())
    return df
        

texts_path = "/Users/sladkydrevo/opt/baka/dataset/texts"
chunker = Chunker(texts_path, chunk_size=128, overlap=10)

stats = chunker.get_all_stats()
i = 1
for s in stats.values():
    df = get_stats_table(s)
    path = "/Users/sladkydrevo/opt/baka/chunk_stats"
    df.to_csv(f"{path}{i}")
    i += 1



