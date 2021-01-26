import glob
import torch
import logging
from torchtext.data.utils import get_tokenizer
import random
from torchtext.experimental.datasets import LanguageModelingDataset
from torch.utils.data.datasets import ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset


###################################################################
# Set up dataset for book corpus
###################################################################
def BookCorpus(vocab, tokenizer=get_tokenizer("basic_english"),
               data_select=('train', 'valid', 'test'), removed_tokens=[],
               min_sentence_len=None):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset(set(('train', 'test', 'valid'))):
        raise TypeError('data_select is not supported!')

    extracted_files = glob.glob('/datasets01/bookcorpus/021819/*/*.txt')
    random.seed(1000)
    random.shuffle(extracted_files)

    num_files = len(extracted_files)
    _path = {'train': extracted_files[:(num_files // 20 * 17)],
             'test': extracted_files[(num_files // 20 * 17):(num_files // 20 * 18)],
             'valid': extracted_files[(num_files // 20 * 18):]}

    data = {}
    for item in _path.keys():
        data[item] = []
        logging.info('Creating {} data'.format(item))
        tokens = []
        for txt_file in _path[item]:
            with open(txt_file, 'r', encoding="utf8", errors='ignore') as f:
                for line in f.readlines():
                    _tokens = tokenizer(line.strip())
                    if min_sentence_len:
                        if len(_tokens) >= min_sentence_len:
                            tokens.append([vocab.stoi[token] for token in _tokens])
                    else:
                        tokens += [vocab.stoi[token] for token in _tokens]
        data[item] = tokens

    for key in data_select:
        if data[key] == []:
            raise TypeError('Dataset {} is empty!'.format(key))
    if min_sentence_len:
        return tuple(LanguageModelingDataset(data[d], vocab, lambda x: x, False)
                     for d in data_select)
    else:
        return tuple(LanguageModelingDataset(torch.tensor(data[d]).long(), vocab, lambda x: x, False)
                     for d in data_select)


class CC100(torch.utils.data.IterableDataset):
    def __init__(self, data_directory, languages, start_line=0, num_lines=16, reset_iterator=False):
        """
        Args:
            data_directory: directory to save the data files
            languages: a unix style filter string set for filtering filename
            start_line: the starting line
            num_lines: the number of output lines
            reset_iterator: reset file handle and continuously read text until the number of lines

        Examples:
            >>> from data import CC100
            >>> dataset = CC100('/datasets01/cc100/031720/', {'as_IN.txt', 'om_KE.txt', 'su_ID.txt'}, start_line=300, num_lines=10)
            >>> for rec in dataset:
            >>>     print(rec)
        """

        file_paths = ListDirFilesIterableDataset(data_directory, languages)
        self.dataset_list = [item for item in LoadFilesFromDiskIterableDataset(file_paths)]
        self.start_line = start_line
        self.num_lines = num_lines
        self.reset_iterator = reset_iterator
        self._count = 0
        self._current_dataset = 0

    def __iter__(self):
        for i, (filename, dataset_handle) in enumerate(self.dataset_list):
            language_id = self.find_language_id(filename)
            self.setup_dataset(dataset_handle)
            for _count in range(self.num_lines):
                _text = self.readline(dataset_handle)
                if not _text:
                    if self.reset_iterator:
                        dataset_handle.seek(0)
                        _text = self.readline(dataset_handle)
                    else:
                        continue
                yield language_id, _text.decode('utf-8')

    def __len__(self):
        return self.num_lines * len(self.dataset_list)

    def setup_dataset(self, dataset_handle):
        for _count in range(self.start_line):
            _text = self.readline(dataset_handle)

    # readline skips newline
    def readline(self, dataset_handle):
        _text = dataset_handle.readline()
        while _text == b'\n':
            _text = dataset_handle.readline()
        return _text

    def find_language_id(self, file_name):
        for idx in range(len(language_list)):
            if language_list[idx] in file_name:
                return idx
        return -1


language_list = ['af_ZA', 'am_ET', 'ar_AR', 'as_IN', 'az_AZ',
                 'be_BY', 'bg_BG', 'bn_IN', 'bn_IN_rom', 'br_FR',
                 'bs_BA', 'ca_ES', 'cs_CZ', 'cy_GB', 'da_DK',
                 'de_DE', 'el_GR', 'en_XX', 'eo_EO', 'es_XX',
                 'et_EE', 'eu_ES', 'fa_IR', 'ff_NG', 'fi_FI',
                 'fr_XX', 'fy_NL', 'ga_IE', 'gd_GB', 'gl_ES',
                 'gn_PY', 'gu_IN', 'ha_NG', 'he_IL', 'hi_IN',
                 'hi_IN_rom', 'hr_HR', 'ht_HT', 'hu_HU', 'hy_AM',
                 'id_ID', 'ig_NG', 'is_IS', 'it_IT', 'ja_XX',
                 'jv_ID', 'ka_GE', 'kk_KZ', 'km_KH', 'kn_IN',
                 'ko_KR', 'ku_TR', 'ky_KG', 'la_VA', 'lg_UG',
                 'li_NL', 'ln_CD', 'lo_LA', 'lt_LT', 'lv_LV',
                 'mg_MG', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN',
                 'ms_MY', 'my_MM', 'my_MM_zaw', 'ne_NP', 'nl_XX',
                 'no_XX', 'ns_ZA', 'om_KE', 'or_IN', 'pa_IN',
                 'pl_PL', 'ps_AF', 'pt_XX', 'qu_PE', 'rm_CH',
                 'ro_RO', 'ru_RU', 'sa_IN', 'sc_IT', 'sd_PK',
                 'si_LK', 'sk_SK', 'sl_SI', 'so_SO', 'sq_AL',
                 'sr_RS', 'ss_SZ', 'su_ID', 'sv_SE', 'sw_KE',
                 'ta_IN', 'ta_IN_rom', 'te_IN', 'te_IN_rom', 'th_TH',
                 'tl_XX', 'tn_BW', 'tr_TR', 'ug_CN', 'uk_UA',
                 'ur_PK', 'ur_PK_rom', 'uz_UZ', 'vi_VN', 'wo_SN',
                 'xh_ZA', 'yi_DE', 'yo_NG', 'zh_CN', 'zh_TW']
