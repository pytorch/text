import torch


class TextVocab(torch.jit.ScriptModule):
    def __init__(self, tokens, unk_token='<unk>'):
        super(TextVocab, self).__init__()
        self.textvocab = torch.classes.torchtext.TextVocab(tokens, unk_token)

    @torch.jit.script_method
    def forward(self, token: str) -> int:
        return self.textvocab.stoi(token)

    @torch.jit.script_method
    def stoi(self, token: str) -> int:
        return self.textvocab.stoi(token)

    @torch.jit.script_method
    def itos(self, idx: int) -> str:
        return self.textvocab.itos(idx)
