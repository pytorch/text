import sentencepiece as spm


class SentencePieceTransform(object):
    def __init__(self, spm_filename):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_filename)

    def __call__(self, txt_str):
        return self.sp_model.EncodeAsIds(txt_str)
