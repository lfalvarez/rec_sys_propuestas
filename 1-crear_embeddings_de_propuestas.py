import torch
from transformers import BertForMaskedLM, BertTokenizer
import pandas as pd
import json
from numpy import asarray, save

class BETOSearchEngine():
    start_with_keyword = '[CLS]'
    end_with_keyword = '[SEP]'
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained("pytorch/")

    def get_tensor_of_entire_text(self, text):
        phrases = text.split('/n')
        list_of_tensors = []
        for phrase in phrases:
            tensor = self.get_entire_phrase_vectorized(phrase)
            list_of_tensors.append(tensor)
        return torch.stack(list_of_tensors)[0][0].mean(0)

    def get_entire_phrase_vectorized(self, phrase):
        phrase = phrase[0:2300]
        if not phrase.startswith(self.start_with_keyword):
            phrase = self.start_with_keyword + phrase
        if not phrase.endswith(self.end_with_keyword):
            phrase += self.end_with_keyword
        tokens = self.tokenizer.tokenize(phrase)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        predictions = self.model(tokens_tensor)[0]
        return predictions[0].mean(0)

    def vectorize(self, text):
        all_words = self.get_entire_phrase_vectorized(text)
        return all_words.detach().numpy()


def write_as_json_file(file_name, content):
    jsonFile = open(f"{file_name}.json", "w")
    jsonFile.write(json.dumps(content))
    jsonFile.close()

if __name__ == '__main__':
    search_engine = BETOSearchEngine()
    df = pd.read_csv('propuestas.csv')
    idx2propid = {i: id_ for i, id_ in enumerate(df.id)}
    write_as_json_file("paso1/idx2propid", idx2propid)
    propid2idx = {id_: i for i, id_ in enumerate(df.id)}
    write_as_json_file("paso1/propid2idx", propid2idx)
    data_title = asarray([search_engine.vectorize(t) for t in df.title])
    save('paso1/embeddings_based_on_title.npy', data_title)
    #data_description = asarray([search_engine.vectorize(t) for t in df.descripción])
    #save('embeddings_based_on_description.npy', data_description)

