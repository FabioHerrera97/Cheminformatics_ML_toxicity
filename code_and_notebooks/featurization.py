import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, Autotokenizer

class LLMsEncoding:
    
    def featurize(smiles_list, model_name, padding=True):
        """
    Featurizes a list of SMILES strings using the pretrained models.

    Parameters:
    -----------
    smiles_list : list of str
        A list of SMILES strings representing molecular structures.
    model_name : str
            The name of the pre-trained model to use for featurization.
    padding : bool, optional (default=True)
        Whether to pad the tokenized inputs to the same length. If True, the tokenizer will
        pad the inputs to the maximum sequence length in the batch. 

    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array of shape (number_of_smiles, 600), where each row corresponds to
        the embedding of a SMILES string from the input list.

        """

        tokenizer = Autotokenizer.from_pretrained(model_name)
        chemberta = AutoModelForMaskedLM.from_pretrained(model_name)

        embeddings_cls = torch.zeros(len(smiles_list), 768)

        with torch.no_grad():
            for i, smiles in enumerate(tqdm(smiles_list)):
                encoded_input = tokenizer(smiles, return_tensors='pt', padding=padding, truncation=True)
                model_output = chemberta(**encoded_input)

                embedding = model_output[0][::, 0, ::]
                embeddings_cls[i] = embedding

        return embeddings_cls.numpy()