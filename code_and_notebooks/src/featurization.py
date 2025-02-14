import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, Autotokenizer

class LLMsEncoding:

    def featurize_ChemBERTa(smiles_list, padding=True):
        chemberta = AutoModelForMaskedLM.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
        tokenizer = Autotokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

        embeddings_mean = torch.zeros(len(smiles_list), 600)

        with torch.no_grad():
            for i, smiles in enumerate(tqdm(smiles_list)):
                encoded_input = tokenizer(smiles, return_tensors='pt', padding=padding, truncation=True)
                model_output = chemberta(**encoded_input)

                embedding = torch.mean(model_output[0], 1)
                embeddings_mean[i] = embedding

        return embeddings_mean.numpy()