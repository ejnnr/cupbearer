import torch
from torch import nn


from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPast

from transformers.models.codegen.modeling_codegen import CodeGenForCausalLM
from transformers.models.codegen.tokenization_codegen_fast import CodeGenTokenizerFast
from transformers.models.gpt_neox.tokenization_gpt_neox_fast import GPTNeoXTokenizerFast
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig


from cupbearer.models import HookedModel

from typing import TypedDict

class TokenDict(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

AUTO_MODELS = {
    "code-gen": "Salesforce/codegen-350M-mono",
}
NEOX_MODELS= {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-14m": "EleutherAI/pythia-14m"
}

HF_MODELS = {**AUTO_MODELS, **NEOX_MODELS}

def load_transformer(name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, int, int]:
    # TODO (odk) add others, generalize (all models you can get from AutoModel)
    if name in AUTO_MODELS:
        checkpoint = AUTO_MODELS[name]
        model: CodeGenForCausalLM = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer: CodeGenTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
        transformer = model.transformer 
        tokenizer.pad_token = tokenizer.eos_token
        emb_dim = transformer.embed_dim 
        max_len = tokenizer.model_max_length
    elif name in NEOX_MODELS:
        checkpoint = NEOX_MODELS[name]
        model: GPTNeoXForCausalLM = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer: GPTNeoXTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
        transformer = model.gpt_neox
        tokenizer.pad_token = tokenizer.eos_token
        config: GPTNeoXConfig = transformer.config
        emb_dim = config.hidden_size
        max_len = config.max_position_embeddings
    else:
        raise ValueError(f"unsupported model {name}")
    
    return transformer, tokenizer, emb_dim, max_len


class TransformerBaseHF(HookedModel):
    def __init__(
            self,
            model: PreTrainedModel, 
            tokenizer: PreTrainedTokenizerBase,
            embed_dim: int,
            max_length: int, #TODO: find attribute in model,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.embed_dim = embed_dim

        # setup
        self.tokenizer.pad_token = tokenizer.eos_token
    
    
    @property
    def defeault_names(self) -> list[str]:
        return ["final_layer_embeddings"]
    
    def process_input(self, x) -> TokenDict:
        tokens = self.tokenizer(
            x, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )
        # load to device
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        return tokens
    
    def get_single_token(self, x):
        tokens: TokenDict = self.tokenizer(x)
        return tokens["input_ids"][0]
    
    def get_embeddings(self, tokens: TokenDict) -> torch.Tensor:
        b, s = tokens["input_ids"].shape
        out: BaseModelOutputWithPast = self.model(**tokens)
        embeddings = out.last_hidden_state
        assert embeddings.shape == (b, s, self.embed_dim), embeddings.shape
        self.store("last_hidden_state", embeddings)
        return embeddings

#TODO: test
class ClassifierTransformerHF(TransformerBaseHF):

    def __init__(
        self,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase,
        embed_dim: int,
        max_length: int, 
        num_classes: int
    ):
        super().__init__(
            model=model, tokenizer=tokenizer, embed_dim=embed_dim, max_length=max_length
        )
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        
    def forward(self, x: str | list[str]):
        # get tokens
        tokens = self.process_input(x)
        # get embeddings
        embeddings = self.get_embeddings(tokens)

        #TODO (odk) se store (doesn't this slow down training?)
        
        # take mean across non-padded dimensions
        mask = tokens["input_ids"] != self.tokenizer.pad_token_id
        mask = mask.unsqueeze(-1)
        assert mask.shape == tokens["input_ids"] + (1,)
        assert embeddings.shape == tokens["input_ids"] + (self.embed_dim,)
        embeddings = embeddings * mask
        embeddings = embeddings.sum(dim=1) / mask.sum(dim=1)

        # compute logits
        logits = self.classifier(embeddings)
        return logits

class TamperingPredictionTransformer(TransformerBaseHF): 
    #TODO: factor out token processing, create interface for using tokenizer in dataset
    def __init__(
        self,
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase,
        embed_dim: int,
        max_length: int,
        n_sensors: int, 
        sensor_token: str = " omit"
    ):
        super().__init__(
            model=model, tokenizer=tokenizer, embed_dim=embed_dim, max_length=max_length
        )
        self.n_sensors = n_sensors 
        self.n_probes = self.n_sensors + 1 # +1 for aggregate measurements
        self.sensor_token_id = self.get_single_token(sensor_token)

        self.probes = nn.ModuleList(
            [nn.Linear(self.embed_dim, 1) for _ in range(self.n_probes)]
        )
    
    def forward(self, x: str | list[str]):
        tokens = self.process_input(x)
        embeddings = self.get_embeddings(tokens)
        b = embeddings.shape[0]

        #TODO (odk) use store
        
        # sensor embeddings
        batch_inds, seq_ids = torch.where(tokens["input_ids"] == self.sensor_token_id) #TODO: pre-specify that its always 3?
        sensor_embs = embeddings[batch_inds, seq_ids].reshape(b, self.n_sensors, self.embed_dim)
        # last token embedding (for aggregate measurement)
        last_token_ind = tokens["attention_mask"].sum(dim=1) - 1
        last_embs = embeddings[torch.arange(b), last_token_ind]
        probe_embs = torch.concat([sensor_embs, last_embs.unsqueeze(dim=1)], axis=1)
        assert probe_embs.shape == (b, self.n_probes, self.embed_dim)
        logits = torch.concat([
            probe(emb) for probe, emb in zip(self.probes, torch.split(probe_embs, 1, dim=1))
        ], dim=1)
        logits = logits.squeeze(dim=-1)
        assert logits.shape == (b, self.n_probes)
        return logits
