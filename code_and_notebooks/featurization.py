import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, Autotokenizer

class LLMsEncoding:

    def featurize_ChemBERTa(smiles_list, padding=True):
        """
    Featurizes a list of SMILES strings using the ChemBERTa-77M-MTR model.

    Parameters:
    -----------
    smiles_list : list of str
        A list of SMILES strings representing molecular structures.
    padding : bool, optional (default=True)
        Whether to pad the tokenized inputs to the same length. If True, the tokenizer will
        pad the inputs to the maximum sequence length in the batch. If False, no padding
        will be applied.

    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array of shape (number_of_smiles, 600), where each row corresponds to
        the mean embedding of a SMILES string from the input list.

        """

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
    
