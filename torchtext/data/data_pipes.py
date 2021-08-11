
from torch.utils.data import IterDataPipe


class ParseSQuADQAData(IterDataPipe):
    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for _, stream in self.source_datapipe:
            raw_json_data = stream['data']
            for layer1 in raw_json_data:
                for layer2 in layer1['paragraphs']:
                    for layer3 in layer2['qas']:
                        _context, _question = layer2['context'], layer3['question']
                        _answers = [item['text'] for item in layer3['answers']]
                        _answer_start = [item['answer_start'] for item in layer3['answers']]
                        if len(_answers) == 0:
                            _answers = [""]
                            _answer_start = [-1]
                        yield (_context, _question, _answers, _answer_start)
