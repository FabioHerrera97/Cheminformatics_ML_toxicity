import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, Autotokenizer

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
        pad the inputs to the maximum sequence length in the batch. 

    Returns:
    --------
    numpy.ndarray
        A 2D NumPy array of shape (number_of_smiles, 600), where each row corresponds to
        the embedding of a SMILES string from the input list.

        """

        tokenizer = Autotokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MTR')
        chemberta = AutoModelForMaskedLM.from_pretrained('DeepChem/ChemBERTa-77M-MTR')

        embeddings_cls = torch.zeros(len(smiles_list), 768)

        with torch.no_grad():
            for i, smiles in enumerate(tqdm(smiles_list)):
                encoded_input = tokenizer(smiles, return_tensors='pt', padding=padding, truncation=True)
                model_output = chemberta(**encoded_input)

                embedding = model_output[0][::, 0, ::]
                embeddings_cls[i] = embedding

        return embeddings_cls.numpy()
    
    def featurize_MolFormer(smiles_list, padding=True):
        
        tokenizer = Autotokenizer.from_pretrained('MolFormer/MolFormer')
        molformer = AutoModel.from_pretrained('MolFormer/MolFormer')

        embeddings_cls = torch.zeros(len(smiles_list), 768)

        with torch.no_grad():
            for i, smiles in enumerate(tqdm(smiles_list)):

                encoded_input = tokenizer(smiles_list, return_tensors='pt', padding=padding, truncation=True)
                model_outputs = molformer (**encoded_input)

                embedding = model_outputs[0][::, 0, ::]
                embeddings_cls[i] = embedding

            return embeddings_cls.numpy()
        
    

